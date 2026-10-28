"""Concurrency on WorkBench: the numbers behind the analysis page (analysis/concurrency.html).

    python analysis/concurrency.py              # writes analysis/concurrency.json; reads every trace (a few minutes)
    python analysis/build_concurrency_page.py   # writes analysis/concurrency-final.html

Needs WorkBench (re-scoring) and the hard-slice runs: the 10-run baselines (tags rep01..rep10), the write-ahead
log repeats (wal01..wal10) and the WAL pilot (wal-p1, for one example). Conflicts are counted only where agents
share a workspace (the agent-driven teams); in the paper's topologies each worker has a private copy.
"""
import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import workbench as wb  # noqa: E402
from conflicts import insert_key, record_of  # noqa: E402
from wal_compare import BASELINES, TASKS, permutation, system  # noqa: E402
from compare import load  # noqa: E402

RUNS = ROOT / "runs"
DRIVEN = {"none": "No channels", "text-coord": "Text + coordinate", "all-coord": "All channels + coordinate"}
KIND = {"email": "email", "reply": "email reply", "forward": "email forward", "event": "calendar event",
        "customer": "CRM customer", "task": "project task", "plot": "analytics plot"}
ID = re.compile(r"\b\d{8}\b")


def runs(pattern):
    return sorted(d for d in glob.glob(str(RUNS / pattern)) if Path(d, "results.jsonl").exists())


def rows_and_traces(dirs):
    for d in dirs:
        for line in open(Path(d, "results.jsonl")):
            r = json.loads(line)
            f = Path(d, "traces", f"{r['id']}.json")
            if f.exists():
                yield d, r, json.loads(f.read_text())


def raw(shown):
    """'email.send_email(...)' -> 'email.send_email.func(...)', WorkBench's action string."""
    i = shown.index("(")
    return shown[:i] + ".func" + shown[i:]


def instances(trace):
    """Every concurrency conflict in one agent-driven task-run, in time order."""
    tools = [e for e in trace["events"] if e["kind"] == "tool"]
    out, created, touched, field_sets = [], defaultdict(list), defaultdict(list), defaultdict(list)
    for e in tools:
        key = insert_key(e["tool"], e["args"])
        if key and e["changes"]:
            created[key].append(e)
        rec = record_of(e["args"]) if ("update" in e["tool"] or "delete" in e["tool"]) else None
        if rec is None:
            continue
        if e["changes"]:
            touched[rec].append(e)
            if "update" in e["tool"] and e["args"].get("field"):
                field_sets[(rec, e["args"]["field"])].append(e)
        else:
            prior = [p for p in touched[rec] if p["agent"] != e["agent"] and p["t"] <= e["t"]]
            if prior:
                out.append({"type": "gone" if "not found" in e["out"].lower() else "redundant", "kind": rec[0],
                            "first": prior[-1], "second": e})
    for key, es in created.items():
        es = sorted(es, key=lambda e: e["t"])
        for second in es[1:]:
            if second["agent"] != es[0]["agent"]:
                out.append({"type": "duplicate", "kind": key[0], "first": es[0], "second": second})
                break
    for (rec, field), es in field_sets.items():
        if len({e["agent"] for e in es}) > 1 and len({str(e["args"].get("new_value")) for e in es}) > 1:
            es = sorted(es, key=lambda e: e["t"])
            out.append({"type": "lost", "kind": rec[0], "first": es[0], "second": es[-1]})
    return sorted(out, key=lambda c: c["second"]["t"])


def without_duplicates(actions):
    """The actions with the later copy of every duplicated insert removed (what a uniqueness check would leave)."""
    seen, kept = set(), []
    for a in actions:
        m = re.match(r"([\w.]+)\.func\((.*)\)$", a, re.S)
        if m:
            tool = m.group(1).replace(".", "_")
            args = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(2)))
            key = insert_key(tool, args)
            if key in seen:
                continue
            if key:
                seen.add(key)
        kept.append(a)
    return kept


def conflict_table():
    """Conflicts before and after the log, per agent-driven setup."""
    out = {}
    for tag, label in DRIVEN.items():
        for wal in (False, True):
            dirs = runs(f"*-agent-driven-{tag}-wal-workbench-*-wal[0-9][0-9]") if wal else \
                runs(f"*-agent-driven-{tag}-workbench-gpt-oss-20b-rep??")
            n = correct = 0
            per_type, task_runs_with, kinds, gaps = Counter(), Counter(), Counter(), []
            dup_runs = dup_fixed = 0
            for d, r, t in rows_and_traces(dirs):
                n += 1
                correct += bool(r["correct"])
                cs = instances(t)
                for c in cs:
                    per_type[c["type"]] += 1
                    if c["type"] == "duplicate":
                        kinds[KIND[c["kind"]]] += 1
                        gaps.append(round(c["second"]["t"] - c["first"]["t"], 1))
                for typ in {c["type"] for c in cs}:
                    task_runs_with[typ] += 1
                if any(c["type"] == "duplicate" for c in cs):
                    dup_runs += 1
                    dup_fixed += wb.score(without_duplicates(r["answer"]), r["expected"])["correct"]
            out[f"{tag}{'+wal' if wal else ''}"] = {
                "label": label, "wal": wal, "runs": len(dirs), "task_runs": n, "accuracy": correct / n,
                "instances": dict(per_type), "task_runs_with": dict(task_runs_with), "duplicate_kinds": dict(kinds),
                "duplicate_gap_median": st.median(gaps) if gaps else None,
                "duplicate_runs": dup_runs, "duplicate_runs_pass_without": dup_fixed}
            print("conflicts", tag, "wal" if wal else "base", per_type, flush=True)
    return out


def remaining():
    """Every conflict left in the WAL runs, with what the second agent had been shown and declared."""
    items = []
    for tag in DRIVEN:
        for d, r, t in rows_and_traces(runs(f"*-agent-driven-{tag}-wal-workbench-*-wal[0-9][0-9]")):
            ev, log = t["events"], t["wal_log"]
            changes = [e for e in log if e["kind"] == "change"]
            intents = {e["id"]: e for e in log if e["kind"] == "intent"}

            def entry(e):  # the log entry a tool call produced
                c = [x for x in changes if x["agent"] == e["agent"] and x["t"] <= e["t"] + 0.05]
                return c[-1] if c else None
            for c in instances(t):
                first, second = c["first"], c["second"]
                ce, fe = entry(second), entry(first)
                intent = intents.get(ce["intent"]) if ce else None
                if intent is None and second.get("changes") == []:  # a failed write has no change entry
                    mine = [x for x in intents.values() if x["agent"] == second["agent"] and x["t"] <= second["t"]]
                    intent = mine[-1] if mine else None
                targets = " ".join(x["target"] for x in intent["resources"]).lower() if intent else ""
                rec = record_of(first["args"])
                items.append({
                    "setup": tag, "task": r["id"], "run": d[-5:], "type": c["type"], "kind": KIND.get(c["kind"], c["kind"]),
                    "gap": round(second["t"] - first["t"], 1),
                    "first": {"agent": first["agent"], "t": first["t"],
                              "change": (first["changes"] or [""])[0]},
                    "second": {"agent": second["agent"], "t": second["t"],
                               "call": (second["changes"] or [f"{second['tool']}({json.dumps(second['args'])})"])[0],
                               "out": second["out"][:120]},
                    "identical": first["args"] == second["args"],
                    "shown_before": fe is not None and intent is not None and intent["id"] > fe["id"],
                    "named_in_intent": bool(rec and rec[1] in targets),
                    "intent": intent["plan"] if intent else None,
                    "intent_targets": targets[:300],
                    "retried_after_refusal": any(x["kind"] == "wal_refused" and x["agent"] == second["agent"]
                                                 and x["tool"] == second["tool"] and x["args"] == second["args"]
                                                 and first["t"] <= x["t"] <= second["t"] for x in ev),
                    "followup": bool(intent and re.search(r"follow.?up|updated|correct|complete|again|new email|resend|full",
                                                          intent["plan"], re.I))})
    by_task = Counter(i["task"] for i in items if i["type"] == "duplicate")
    print("remaining", Counter(i["type"] for i in items), flush=True)
    return {"items": items, "duplicate_tasks": by_task.most_common(10),
            "task_runs": len({(i["setup"], i["run"], i["task"]) for i in items}),
            "total_task_runs": sum(1 for tag in DRIVEN
                                   for d in runs(f"*-agent-driven-{tag}-wal-workbench-*-wal[0-9][0-9]")
                                   for _ in open(Path(d, "results.jsonl")))}


def cost_and_usage():
    """Tokens, turns and time per task, with and without the log, and how agents used the log."""
    out = {}
    pairs = [(f"agent-driven-{tag}", f"agent-driven-{tag}-wal", f"Agent-driven, {DRIVEN[tag].lower()}") for tag in DRIVEN]
    pairs += [(t, f"{t}-wal", t.capitalize()) for t in ("independent", "centralized", "decentralized", "hybrid")]
    for base, wal, label in pairs:
        side = {}
        for key, pat in (("base", f"*-{base}-workbench-gpt-oss-20b-rep??"), ("wal", f"*-{wal}-workbench-*-wal[0-9][0-9]")):
            rows = [json.loads(line) for d in runs(pat) for line in open(Path(d, "results.jsonl"))]
            side[key] = {"task_runs": len(rows),
                         "prompt_tokens": st.mean(r["prompt_tokens"] for r in rows),
                         "completion_tokens": st.mean(r["completion_tokens"] for r in rows),
                         "turns": st.mean(r["turns"] for r in rows), "tool_calls": st.mean(r["tool_calls"] for r in rows),
                         "seconds": st.mean(r["seconds"] for r in rows)}
            if key == "wal":
                comm = Counter()
                for r in rows:
                    comm.update({k: v for k, v in r.get("comm", {}).items() if k.startswith("wal_")})
                side[key]["log"] = {k[4:]: v / len(rows) for k, v in comm.items()}
        out[wal] = {"label": label, **side}
    return out


def log_behaviour():
    """Per WAL setup: how agents start, how specific intents are, and who holds back."""
    out = {}
    for name, pat in [(f"agent-driven-{t}-wal", f"*-agent-driven-{t}-wal-workbench-*-wal[0-9][0-9]") for t in DRIVEN] + \
                     [(f"{t}-wal", f"*-{t}-wal-workbench-*-wal[0-9][0-9]")
                      for t in ("independent", "centralized", "decentralized", "hybrid")]:
        first, targets, agree, res = Counter(), Counter(), Counter(), []
        agents = holding_back = 0
        for d, r, t in rows_and_traces(runs(pat)):
            events = t.get("wal_events") or [e for e in t["events"] if e["kind"].startswith("wal")]
            log = t["wal_log"]
            for a in sorted({e["agent"] for e in events} | {e["agent"] for e in log}):
                ks = [e["kind"] for e in events if e["agent"] == a]
                first[ks[0] if ks else "none"] += 1
                agents += 1
                holding_back += "wal_read" in ks and not any(e["agent"] == a for e in log)
            intents = {e["id"]: e for e in log if e["kind"] == "intent"}
            for e in intents.values():
                res.append(len(e["resources"]))
                for x in e["resources"]:
                    targets["record id" if ID.search(x["target"]) else "description" if x["target"] else "none"] += 1
            for e in log:
                m = re.search(r'(customer_id|event_id|task_id|email_id)="(\d+)"', e.get("call", ""))
                if e["kind"] == "change" and m and e["intent"] in intents:
                    named = m.group(2) in " ".join(x["target"] for x in intents[e["intent"]]["resources"])
                    agree["named" if named else "not named"] += 1
        tt = sum(targets.values())
        out[name] = {"agents": agents, "first_action": {k.replace("wal_", ""): v / agents for k, v in first.items()},
                     "holding_back": holding_back / agents, "resources_per_intent": st.mean(res),
                     "targets": {k: v / tt for k, v in targets.items()},
                     "change_named_in_intent": agree["named"] / max(1, sum(agree.values()))}
        print("behaviour", name, flush=True)
    return out


def accuracy():
    systems = {}
    for kind in BASELINES:
        dirs = runs(f"*-{kind}-workbench-gpt-oss-20b-rep??")
        if dirs:
            systems[load(dirs[0])[0]] = system(dirs)
    wal = defaultdict(list)
    for d in runs("*-wal-workbench-*-wal[0-9][0-9]"):
        wal[load(d)[0]].append(d)
    rows = []
    for label, dirs in wal.items():
        w, b = system(dirs), systems.get(label.removesuffix("+wal"))
        if not (w and b):
            continue
        above, below = permutation(w["per_run"], b["per_run"])
        rows.append({"setup": label.removesuffix("+wal"), "base": b["acc"]["all"], "wal": w["acc"]["all"],
                     "base_range": b["range"], "wal_range": w["range"], "p_better": above, "p_worse": below})
    return {"rows": rows, "single": systems["single"]["acc"]["all"], "decentralized": systems["decentralized"]["acc"]["all"]}


def vote():
    """Decentralized: what the majority vote kept, with and without the log, and what keeping every change would score."""
    out = {}
    for key, pat in (("base", "*-decentralized-workbench-gpt-oss-20b-rep??"), ("wal", "*-decentralized-wal-workbench-*-wal[0-9][0-9]")):
        n = acc = nomaj = empty = union_ok = 0
        changed, share = [], []
        for d, r, t in rows_and_traces(runs(pat)):
            v = t["vote"]
            answers = list(v["answers"].values())
            n += 1
            acc += bool(r["correct"])
            nomaj += v["votes"] == 1
            empty += not v["answers"][v["winner"]]
            changed.append(sum(bool(a) for a in answers))
            union = list(dict.fromkeys(raw(x) for a in answers for x in a))
            union_ok += wb.score(union, r["expected"])["correct"]
            share.append(len(v["answers"][v["winner"]]) / len(union) if union else 1)
        out[key] = {"task_runs": n, "accuracy": acc / n, "no_majority": nomaj / n, "winner_empty": empty / n,
                    "workers_changing": st.mean(changed), "winner_share": st.mean(share), "union_accuracy": union_ok / n}
        print("vote", key, flush=True)
    return out


def examples():
    """Two logs: the pilot's no-channel team coordinating, and a decentralized team voting its work away."""
    ex = {}
    d = runs("*-agent-driven-none-wal-workbench-*-wal-p1")[0]
    t = json.loads(Path(d, "traces", "customer_relationship_manager-75.json").read_text())
    ex["coordination"] = {"task": t["task"], "log": t["wal_log"],
                          "other": [e for e in t["events"] if e["kind"] in ("wal_read", "wal_append_refused", "wal_refused")]}
    d = runs("*-decentralized-wal-workbench-*-wal01")[0]
    t = json.loads(Path(d, "traces", "customer_relationship_manager-46.json").read_text())
    # mas.py traces don't store the task; it opens each worker's first message.
    first = next(m["content"] for m in t["conversations"]["agent_1"] if m["role"] == "user")
    task = first.split("Here is the task for the team to work on:\n", 1)[-1].split("\n\nTo start,")[0].strip()
    ex["vote"] = {"task": task, "log": t["wal_log"],
                  "other": [e for e in t["wal_events"] if e["kind"] in ("wal_read", "wal_append_refused", "wal_refused")],
                  "vote": {a: len(x) for a, x in t["vote"]["answers"].items()}, "winner": t["vote"]["winner"]}
    return ex


def main():
    data = {"tasks": len(TASKS), "conflicts": conflict_table(), "remaining": remaining(), "cost": cost_and_usage(),
            "behaviour": log_behaviour(), "accuracy": accuracy(), "vote": vote(), "examples": examples()}
    (ROOT / "analysis" / "concurrency.json").write_text(json.dumps(data, indent=1, default=str))
    print("written analysis/concurrency.json")


if __name__ == "__main__":
    main()
