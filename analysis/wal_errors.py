"""Why the write-ahead log removes most conflicts but barely moves accuracy: what the failed task-runs got wrong.

    python analysis/wal_errors.py      # prints the breakdown and writes analysis/wal_errors.json

Runs: the single agent and the three agent-driven teams without the log (10 hard-slice repeats each, rep01..) and
with it (wal01..), on the 64 hard tasks. WorkBench scores the final state of the databases, so a task-run is
compared with the correct answer by the changes it made (writes that changed the databases, replayed on fresh
databases), as multisets of WorkBench action strings:
- a missing change and an extra change on the same record (or recipient, event name...) are one change with
  wrong values; the argument that differs is recorded;
- other missing changes are omitted; other extra changes went to records the task didn't ask to change.
A failed task-run first gets the counterfactual of removing teammates' duplicate inserts (the only conflict that
changes the final state: the others are writes that changed nothing), then one primary category from the rest.
For omitted changes to existing records: whether any agent's WorkBench tool results ever showed the record, and
(with the log) whether an intent named it.
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from concurrency import insert_key  # noqa: E402
from hard_common import DOTTED, main_tool, target  # noqa: E402
from wal_compare import TASKS  # noqa: E402

import workbench as wb  # noqa: E402
from src.evals.actions import _parse_and_dispatch  # noqa: E402

SETUPS = {"none": "No channels", "text-coord": "Text + coordinate", "all-coord": "All channels + coordinate"}
CONDITIONS = [("single", "Single agent", "*-sas-workbench-gpt-oss-20b-rep??")] + [
    c for s, label in SETUPS.items() for c in (
        (s, label, f"*-agent-driven-{s}-workbench-gpt-oss-20b-rep??"),
        (s + "+wal", label + " + log", f"*-agent-driven-{s}-wal-workbench-gpt-oss-20b-wal[0-9][0-9]"))]
CATEGORIES = ["Duplicate insert by a teammate", "Nothing changed", "Some required changes missing",
              "Right records, wrong values", "Extra changes to other records", "Wrong records: missed some, changed others",
              "Mixed", "Other"]
ID_KEYS = ("email_id", "event_id", "task_id", "customer_id")
RAW_CALL = re.compile(r"<\|call\|>|to=functions\.|<\|channel\|>")
# a final reply that mentions a teammate: from an agent that changed nothing, it left the work to them
TEAMMATE = re.compile(r"\bagent[ _]?\d\b|teammate|another agent|other agents?\b|the team\b", re.I)


def args_of(action):
    return dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', action))


def tool_of(action):
    return action.split(".func(")[0]


def changes(actions):
    """The write actions that changed the databases, replayed in order on fresh databases."""
    wb._state._local.tool_state = wb._state._pristine_state().copy()
    st = wb._state.get_state()
    out = []
    for a in actions:
        if not wb.is_write(a):
            continue
        before = {f: getattr(st, f).copy() for f in wb._MUTABLE}
        try:
            _parse_and_dispatch(a)
        except Exception:  # noqa: BLE001 - a call that raised changed nothing
            continue
        if any(not getattr(st, f).equals(before[f]) for f in wb._MUTABLE):
            out.append(a)
    wb._state._local.tool_state = None
    return out


def without_teammate_duplicates(actions, keys):
    """The actions with the later copy of every insert whose key two agents created removed."""
    seen, kept = set(), []
    for a in actions:
        k = insert_key(tool_of(a).replace(".", "_"), args_of(a))
        if k in keys and k in seen:
            continue
        seen.add(k)
        kept.append(a)
    return kept


def classify(made, expected):
    """(primary category, details) for a failed task-run's changes against the correct ones."""
    M, G = Counter(a.lower() for a in made), Counter(a.lower() for a in expected)
    missing, extra = list((G - M).elements()), list((M - G).elements())
    wrong, args_diff = [], Counter()
    for g in list(missing):
        e = next((e for e in extra if target(e) == target(g) and tool_of(e) == tool_of(g)), None)
        if e is not None:
            missing.remove(g)
            extra.remove(e)
            wrong.append((g, e))
            ga, ea = args_of(g), args_of(e)
            for k in sorted(set(ga) | set(ea)):
                if ga.get(k) != ea.get(k):
                    args_diff[f"{tool_of(g).split('.')[-1]}: {k}"] += 1
    repeats = [e for e in extra if M[e] > 1 and G[e] >= 1]  # a required change made again (same text)
    extra = [e for e in extra if e not in repeats]
    if not made:
        cat = "Nothing changed"
    elif missing and not extra and not wrong:
        cat = "Some required changes missing"
    elif wrong and not missing and not extra:
        cat = "Right records, wrong values"
    elif extra and not missing and not wrong:
        cat = "Extra changes to other records"
    elif missing and extra and not wrong:
        cat = "Wrong records: missed some, changed others"
    elif missing or extra or wrong:
        cat = "Mixed"
    else:
        cat = "Other"
    return cat, {"omitted": missing, "extra": extra, "wrong": wrong, "repeats": repeats, "args_diff": args_diff}


def tool_texts(trace):
    """Per agent, the text of its WorkBench tool results (not the log, messages or shared state)."""
    convs = trace["conversations"] if isinstance(trace, dict) else {"agent": trace}
    return {ag: " ".join(str(m.get("content", "")) for m in conv if m.get("role") == "tool" and m.get("name") in DOTTED)
            for ag, conv in convs.items()}


def raw_endings(trace):
    """Agents whose last reply is a tool call written out as text (the model meant to act and the call was lost)."""
    if not isinstance(trace, dict):
        return 0
    return sum(1 for e in trace["events"] if e["kind"] == "idle" and RAW_CALL.search(str(e.get("reply", ""))))


def claims(trace):
    """What the agents told each other they would do or had done: intents, messages, A2A tasks, shared state."""
    if not isinstance(trace, dict):
        return ""
    parts = [f"{e.get('plan', '')} {json.dumps(e.get('resources', []))}" for e in trace.get("wal_log", [])
             if e.get("kind") == "intent"]
    parts += [str(e.get(k, "")) for e in trace["events"] if e["kind"] in ("message", "a2a_send", "a2a_update", "state_write")
              for k in ("text", "instruction", "value", "data")]
    return " ".join(parts)


def deferrers(trace):
    """Agents that changed nothing and ended by pointing at a teammate ('Agent 2 is handling it')."""
    if not isinstance(trace, dict):
        return 0
    acted = {e["agent"] for e in trace["events"] if e["kind"] == "tool" and e["changes"]}
    last = {e["agent"]: str(e.get("reply", "")) for e in trace["events"] if e["kind"] == "idle"}
    return sum(1 for ag, reply in last.items() if ag not in acted and TEAMMATE.search(reply))


def task_run(d, r, trace):
    made = changes(r["answer"])
    out = {"task": r["id"], "correct": bool(r["correct"]), "n_made": len(made), "n_expected": len(r["expected"]),
           "raw_endings": raw_endings(trace), "deferrers": deferrers(trace),
           "unknown_tool": sum(1 for conv in (trace["conversations"].values() if isinstance(trace, dict) else [trace])
                               for m in conv if m.get("role") == "tool" and "unknown tool" in str(m.get("content", "")))}
    if out["correct"]:
        return out
    if isinstance(trace, dict):  # agent-driven: inserts created by two agents
        by_key = defaultdict(set)
        for e in trace["events"]:
            if e["kind"] == "tool" and e["changes"]:
                k = insert_key(e["tool"], e["args"])
                if k:
                    by_key[k].add(e["agent"])
        dup_keys = {k for k, ags in by_key.items() if len(ags) > 1}
        if dup_keys:
            out["had_duplicate"] = True
            kept = without_teammate_duplicates(r["answer"], dup_keys)
            if wb.score(kept, r["expected"])["correct"]:
                out["category"] = CATEGORIES[0]
                return out
            made = changes(kept)
    cat, det = classify(made, r["expected"])
    out["category"] = cat
    out["omitted"], out["extra"], out["wrong"], out["repeats"] = (len(det["omitted"]), len(det["extra"]),
                                                                  len(det["wrong"]), len(det["repeats"]))
    out["args_diff"] = dict(det["args_diff"])
    # why each omitted change to an existing record was omitted
    texts = tool_texts(trace)
    claimed = claims(trace)
    why = Counter()
    for g in det["omitted"]:
        ids = [v for k, v in args_of(g).items() if k in ID_KEYS]
        if not ids or tool_of(g).endswith(("reply_email", "forward_email")):
            why["a new item (email, event, customer, task) never created"] += 1
            continue
        rid = ids[0]
        if not any(rid in t for t in texts.values()):
            why["record never shown by any search"] += 1
        elif rid in claimed:
            why["shown, and named in an intent or message, but never changed"] += 1
        else:
            why["shown, but never changed"] += 1
    out["omitted_why"] = dict(why)
    tool = main_tool(r["expected"])
    out["cap"] = TASKS[r["id"]]["cap"] and sum(tool_of(a) == tool for a in made) == 5
    return out


def load(pattern):
    rows = []
    for d in sorted(glob.glob(str(ROOT / "runs" / pattern))):
        res = [json.loads(line) for line in open(Path(d, "results.jsonl"))]
        if not set(TASKS) <= {r["id"] for r in res}:
            continue  # complete runs only, as in wal_compare.py
        for r in res:
            if r["id"] in TASKS:
                trace = json.loads(Path(d, "traces", f"{r['id']}.json").read_text())
                rows.append(task_run(d, r, trace))
    return rows


def summary(rows):
    n, failed = len(rows), [r for r in rows if not r["correct"]]
    cats = Counter(r["category"] for r in failed)
    why = Counter()
    args = Counter()
    for r in failed:
        why.update(r.get("omitted_why", {}))
        args.update(r.get("args_diff", {}))
    return {"task_runs": n, "accuracy": 1 - len(failed) / n, "failed": len(failed),
            "categories": {c: cats.get(c, 0) for c in CATEGORIES},
            "with_duplicate": sum(bool(r.get("had_duplicate")) for r in failed),
            "omitted": sum(r.get("omitted", 0) for r in failed), "extra": sum(r.get("extra", 0) for r in failed),
            "wrong": sum(r.get("wrong", 0) for r in failed), "omitted_why": dict(why), "args_diff": dict(args.most_common(12)),
            "cap": sum(bool(r.get("cap")) for r in failed),
            "raw_endings": sum(r["raw_endings"] > 0 for r in rows), "raw_endings_failed": sum(r["raw_endings"] > 0 for r in failed),
            "unknown_tool": sum(r["unknown_tool"] > 0 for r in rows),
            "deferred": sum(r["deferrers"] > 0 for r in rows), "deferred_failed": sum(r["deferrers"] > 0 for r in failed),
            "deferred_omitting": sum(r["deferrers"] > 0 and r.get("omitted", 0) + (r.get("category") == "Nothing changed") > 0
                                     for r in failed),
            "made_vs_expected": {"fewer": sum(r["n_made"] < r["n_expected"] for r in failed),
                                 "same": sum(r["n_made"] == r["n_expected"] for r in failed),
                                 "more": sum(r["n_made"] > r["n_expected"] for r in failed)}}


def main():
    out, per_task = {"_categories": CATEGORIES, "_conditions": [(k, label) for k, label, _ in CONDITIONS]}, {}
    for key, label, pattern in CONDITIONS:
        rows = load(pattern)
        if not rows:
            continue
        out[key] = {"label": label, **summary(rows)}
        per_task[key] = defaultdict(list)
        for r in rows:
            per_task[key][r["task"]].append(r["correct"])
        s = out[key]
        print(f"\n{label}: {s['task_runs']} task-runs, accuracy {s['accuracy']:.0%}, {s['failed']} failed")
        for c, v in s["categories"].items():
            print(f"  {c:46s} {v:4d}  ({v / s['task_runs']:.1%} of task-runs)")
        print(f"  omitted changes {s['omitted']}, extra {s['extra']}, wrong values {s['wrong']}; why omitted: {s['omitted_why']}")
        print(f"  differing args: {s['args_diff']}")
        print(f"  failed with a teammate duplicate {s['with_duplicate']}; stopped at the 5-result cap {s['cap']}; "
              f"task-runs with a tool call written as text at the end {s['raw_endings']} ({s['raw_endings_failed']} failed); "
              f"made fewer/same/more changes than needed {s['made_vs_expected']}")
        print(f"  task-runs where an agent changed nothing and left the work to a teammate: {s['deferred']} "
              f"({s['deferred_failed']} failed, {s['deferred_omitting']} of them with required changes missing)", flush=True)
    # tasks: how often each is solved, per condition
    rates = {k: {t: sum(v) / len(v) for t, v in pt.items()} for k, pt in per_task.items()}
    driven = [k for k in rates if k != "single"]
    never = [t for t in TASKS if all(rates[k].get(t, 0) == 0 for k in rates)]
    always = [t for t in TASKS if all(rates[k].get(t, 0) >= 0.9 for k in rates)]
    out["_tasks"] = {"rates": rates, "never_solved": never, "nearly_always_solved": always,
                     "solved_rarely": [t for t in TASKS if t not in never and max(rates[k].get(t, 0) for k in driven) <= 0.3]}
    print(f"\n{len(never)} of {len(TASKS)} tasks are never solved by any of these setups; {len(always)} are solved in 90%+ "
          f"of runs by all; {len(out['_tasks']['solved_rarely'])} more are solved in at most 30% of runs by every team")
    (ROOT / "analysis" / "wal_errors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
