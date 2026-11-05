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
from conflicts import conflicts  # noqa: E402
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
# What contributed to a failed task-run (several can apply), for the per-task table: conflict-related first.
FACTORS = [("dup_caused", "A teammate's duplicate insert, the only error", "conflict"),
           ("dup_other", "A teammate's duplicate insert, with other errors", "conflict"),
           ("lost_update", "A lost update (two agents set one field differently)", "conflict"),
           ("never_found", "A required record that no search showed", "failure"),
           ("found_not_changed", "A required record shown but not changed", "failure"),
           ("claimed", "A required record named in an intent or message, never changed", "failure"),
           ("not_created", "A required new item (email, event, reply...) not created", "failure"),
           ("extra", "Changes to records the task didn't ask for", "failure"),
           ("wrong_values", "Wrong values on a required record", "failure"),
           ("nothing", "Nothing changed at all", "failure"),
           ("deferred", "An agent changed nothing and left the work to a teammate", "behaviour"),
           ("lost_call", "An agent ended on a garbled or empty tool call", "behaviour")]
REASON = {"record never shown by any search": "never_found", "shown, but never changed": "found_not_changed",
          "shown, and named in an intent or message, but never changed": "claimed",
          "a new item (email, event, customer, task) never created": "not_created"}
KIND = {"email_id": "email", "event_id": "event", "task_id": "task", "customer_id": "customer"}


def record(action):
    """A short name for what an action changes: 'event 00000205', 'new email to x: \"subject\"', ..."""
    a, tool = args_of(action), tool_of(action).split(".")[-1]
    if tool in ("reply_email", "forward_email"):
        return f"{tool.split('_')[0]} of email {a.get('email_id')}" + (f" to {a['recipient']}" if a.get("recipient") else "")
    for k, kind in KIND.items():
        if k in a:
            return f"{kind} {a[k]}"
    if tool == "send_email":
        return f"new email to {a.get('recipient', '?')}: \u201c{a.get('subject', '')}\u201d"
    if tool == "create_event":
        return f"new event \u201c{a.get('event_name', '')}\u201d at {a.get('event_start', '?')}"
    if tool == "add_customer":
        return f"new customer {a.get('customer_name', '?')}"
    if tool == "create_task":
        return f"new task \u201c{a.get('task_name', '')}\u201d"
    if tool == "create_plot":
        return f"plot of {a.get('value_to_plot', '?')} ({a.get('plot_type', '?')})"
    return tool


def describe(label, st):
    """What an existing record is, from WorkBench's starting databases (empty for new items)."""
    kind, _, rid = label.partition(" ")
    frame = {"event": "calendar_events", "email": "emails", "task": "project_tasks", "customer": "crm_data"}.get(kind)
    if not frame:
        return ""
    df = getattr(st, frame)
    row = df[df[kind + "_id"] == rid]
    if row.empty:
        return ""
    x = row.iloc[0]
    who = lambda e: str(e).split("@")[0]  # noqa: E731 - every address is @atlas.com or a customer's
    if kind == "event":
        return f"\u201c{x.event_name}\u201d with {who(x.participant_email)}, {x.event_start[:16]}"
    if kind == "email":
        way = "from" if x["inbox/outbox"] == "inbox" else "to"
        return f"{way} {who(x['sender/recipient'])}: \u201c{x.subject}\u201d, {x.sent_datetime[:10]}"
    if kind == "task":
        return f"\u201c{x.task_name}\u201d ({x.board}): {x.list_name}, {who(x.assigned_to_email)}, due {x.due_date}"
    return f"{x.customer_name}: {x.status}, {x.product_interest}, {who(x.assigned_to_email)}, last contact {x.last_contact_date}"


def lost_calls(trace):
    """Agents whose last reply is a tool call written out as text, or empty: the call never ran."""
    convs = trace["conversations"].values() if isinstance(trace, dict) else [trace]
    n = 0
    for conv in convs:
        last = [m for m in conv if m.get("role") == "assistant"]
        if last and not last[-1].get("tool_calls"):
            text = str(last[-1].get("content") or "")
            n += bool(RAW_CALL.search(text) or not text.strip() or text == "None")
    return n


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
           "raw_endings": raw_endings(trace), "deferrers": deferrers(trace), "lost_calls": lost_calls(trace),
           "conflicts": conflicts(trace) if isinstance(trace, dict) else None,
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
    out["extra_list"] = [record(e) for e in det["extra"]]
    out["wrong_list"] = [f"{record(g)}: {k}" for g, e in det["wrong"]
                         for k in sorted(set(args_of(g)) | set(args_of(e))) if args_of(g).get(k) != args_of(e).get(k)]
    out["omitted_list"] = []
    # why each omitted change to an existing record was omitted
    texts = tool_texts(trace)
    claimed = claims(trace)
    why = Counter()
    for g in det["omitted"]:
        ids = [v for k, v in args_of(g).items() if k in ID_KEYS]
        if not ids or tool_of(g).endswith(("reply_email", "forward_email")):
            why["a new item (email, event, customer, task) never created"] += 1
            out["omitted_list"].append([record(g), "not_created"])
            continue
        rid = ids[0]
        if not any(rid in t for t in texts.values()):
            reason = "record never shown by any search"
        elif rid in claimed:
            reason = "shown, and named in an intent or message, but never changed"
        else:
            reason = "shown, but never changed"
        why[reason] += 1
        out["omitted_list"].append([record(g), REASON[reason]])
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


def factors(r):
    """What contributed to a failed task-run (keys of FACTORS)."""
    c = r.get("conflicts") or {}
    f = {reason for _, reason in r.get("omitted_list", [])}
    if r.get("category") == CATEGORIES[0]:
        f.add("dup_caused")
    elif r.get("had_duplicate"):
        f.add("dup_other")
    if c.get("lost_updates"):
        f.add("lost_update")
    if r.get("extra_list"):
        f.add("extra")
    if r.get("wrong_list"):
        f.add("wrong_values")
    if r.get("category") == "Nothing changed":
        f.add("nothing")
    if r["deferrers"]:
        f.add("deferred")
    if r["lost_calls"]:
        f.add("lost_call")
    return f


def task_summary(rows):
    """One task under one condition: how often solved, what contributed to its failures, which records."""
    failed = [r for r in rows if not r["correct"]]
    f, omitted, reason, extra, wrong = Counter(), Counter(), defaultdict(Counter), Counter(), Counter()
    for r in failed:
        f.update(factors(r))
        for lab, why in r.get("omitted_list", []):
            omitted[lab] += 1
            reason[lab][why] += 1
        extra.update(set(r.get("extra_list", [])))
        wrong.update(set(r.get("wrong_list", [])))
    conf = [r["conflicts"] for r in rows if r.get("conflicts")]
    total = lambda c: c["dup_inserts"] + c["gone"] + c["redundant"] + c["lost_updates"]  # noqa: E731
    return {"runs": len(rows), "solved": len(rows) - len(failed), "factors": dict(f),
            "conflict_runs": sum(total(c) > 0 for c in conf), "conflicts": sum(total(c) for c in conf),
            "harmless_runs": sum(c["gone"] + c["redundant"] > 0 for c in conf),
            "omitted": [[lab, n, reason[lab].most_common(1)[0][0]] for lab, n in omitted.most_common(3)],
            "extra": [[lab, n] for lab, n in extra.most_common(3)], "wrong": [[lab, n] for lab, n in wrong.most_common(2)]}


def main():
    out, per_task, by_task = {"_categories": CATEGORIES, "_conditions": [(k, label) for k, label, _ in CONDITIONS]}, {}, {}
    for key, label, pattern in CONDITIONS:
        rows = load(pattern)
        if not rows:
            continue
        out[key] = {"label": label, **summary(rows)}
        per_task[key] = defaultdict(list)
        by_task[key] = defaultdict(list)
        for r in rows:
            per_task[key][r["task"]].append(r["correct"])
            by_task[key][r["task"]].append(r)
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
    out["_factors"] = FACTORS
    st = wb._state._pristine_state()
    labels = set()
    for key, pt in by_task.items():
        for t, rows in pt.items():
            agg = task_summary(rows)
            out[key].setdefault("tasks", {})[t] = agg
            labels |= {x[0] for k in ("omitted", "extra", "wrong") for x in agg[k]}
    out["_records"] = {lab: describe(lab.split(":")[0], st) for lab in sorted(labels)}
    info = {t["id"]: t for t in wb.load_tasks()}
    out["_task_info"] = {t: {"question": info[t]["question"], "changes": TASKS[t]["changes"], "cap": TASKS[t]["cap"]}
                         for t in TASKS}
    (ROOT / "analysis" / "wal_errors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
