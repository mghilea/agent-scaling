"""Concurrency conflicts in agent-driven WorkBench runs, per condition (e.g. driven/none vs driven/none+wal).

    python analysis/conflicts.py runs/*-agent-driven-none-workbench-*-rep?? runs/*-agent-driven-none-wal-*
    python analysis/conflicts.py --json out.json runs/...      # also write the numbers

Runs are grouped by condition (compare.py's labels); each task-run (one task in one run) is checked for:
- duplicate insert: two agents each created the same thing (an email to the same recipient with the same
  subject, a reply or forward of the same email, an event with the same name and start, a customer with the
  same name, a task with the same name, the same plot). Scored changes, so it can make the task wrong.
- collision: a write that changed nothing because a teammate had already changed that record. Split into
  gone (the record was already deleted by a teammate: "not found") and redundant (a teammate had already set
  it, e.g. the same update twice). Harmless but wasted.
- lost update: two agents set the same field of the same record to different values; the later one wins.
Also accuracy, turns and tokens, and with --wal the log's counts (intents, refused changes and why).
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from compare import load  # noqa: E402

RECORD_ARGS = ("customer_id", "event_id", "task_id", "email_id")


def insert_key(tool, a):
    """What makes two creations 'the same thing', or None if the call doesn't create anything."""
    low = lambda k: str(a.get(k, "")).strip().lower()  # noqa: E731
    if tool == "email_send_email":
        return ("email", low("recipient"), low("subject"))
    if tool == "email_reply_email":
        return ("reply", low("email_id"))
    if tool == "email_forward_email":
        return ("forward", low("email_id"), low("recipient"))
    if tool == "calendar_create_event":
        return ("event", low("event_name"), low("event_start"))
    if tool == "customer_relationship_manager_add_customer":
        return ("customer", low("customer_name"))
    if tool == "project_management_create_task":
        return ("task", low("task_name"))
    if tool == "analytics_create_plot":
        return ("plot", low("time_min"), low("time_max"), low("value_to_plot"), low("plot_type"))
    return None


def record_of(a):
    for k in RECORD_ARGS:
        if a.get(k):
            return (k, str(a[k]))
    return None


def conflicts(trace):
    """Conflict counts for one task-run, from its event log (tool calls in time order)."""
    tools = [e for e in trace["events"] if e["kind"] == "tool"]
    created = defaultdict(list)   # insert key -> [(t, agent)] of creations that changed the workspace
    touched = defaultdict(list)   # record -> [(t, agent)] of changes to it
    field_sets = defaultdict(list)  # (record, field) -> [(t, agent, value)] of updates that changed it
    gone = redundant = 0
    for e in tools:
        tool, a, changed = e["tool"], e["args"], bool(e["changes"])
        key = insert_key(tool, a)
        if key and changed:
            created[key].append((e["t"], e["agent"]))
        rec = record_of(a) if ("update" in tool or "delete" in tool) else None
        if rec is None:
            continue
        if changed:
            touched[rec].append((e["t"], e["agent"]))
            if "update" in tool and a.get("field"):
                field_sets[(rec, a["field"])].append((e["t"], e["agent"], str(a.get("new_value"))))
        elif any(ag != e["agent"] and t <= e["t"] for t, ag in touched[rec]):
            if "not found" in e["out"].lower():
                gone += 1
            else:
                redundant += 1
    dups = [sorted(v) for v in created.values() if len({ag for _, ag in v}) > 1]
    lost = sum(1 for v in field_sets.values()
               if len({ag for _, ag, _ in v}) > 1 and len({val for _, _, val in v}) > 1)
    gaps = [v[1][0] - v[0][0] for v in dups]
    return {"dup_inserts": len(dups), "gone": gone, "redundant": redundant, "lost_updates": lost, "dup_gaps": gaps}


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        i = args.index("--json")
        out_json, args = args[i + 1], args[:i] + args[i + 2:]
    groups = defaultdict(list)
    for d in args:
        d = Path(d)
        if not (d / "results.jsonl").exists():
            continue
        label, rows = load(d)
        groups[label].append((d, rows))
    if not groups:
        sys.exit("no runs given")

    summary = {}
    for label, runs in groups.items():
        rs, cs = [], []
        for d, rows in runs:
            for tid, r in rows.items():
                f = d / "traces" / f"{tid}.json"
                if not f.exists():
                    continue
                rs.append(r)
                cs.append(conflicts(json.loads(f.read_text())))
        n = len(rs)
        any_of = lambda k: sum(c[k] > 0 for c in cs) / n  # noqa: E731
        wal = Counter()
        for r in rs:
            wal.update({k: v for k, v in r.get("comm", {}).items() if k.startswith("wal_")})
        gaps = [g for c in cs for g in c["dup_gaps"]]
        summary[label] = {
            "runs": len(runs), "task_runs": n,
            "accuracy": sum(bool(r["correct"]) for r in rs) / n,
            "side_effects": sum(bool(r.get("side_effects")) for r in rs) / n,
            "dup_insert_rate": any_of("dup_inserts"), "gone_rate": any_of("gone"),
            "redundant_rate": any_of("redundant"), "lost_update_rate": any_of("lost_updates"),
            "dup_inserts_per_task": sum(c["dup_inserts"] for c in cs) / n,
            "acc_with_dup": (sum(bool(r["correct"]) for r, c in zip(rs, cs) if c["dup_inserts"])
                             / max(1, sum(c["dup_inserts"] > 0 for c in cs))),
            "dup_gap_median_s": statistics.median(gaps) if gaps else None,
            "turns": sum(r["turns"] for r in rs) / n,
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
            "wal_per_task": {k: v / n for k, v in sorted(wal.items())},
        }

    cols = [("runs", "runs", "{:d}"), ("task_runs", "task-runs", "{:d}"), ("accuracy", "accuracy", "{:.0%}"),
            ("side_effects", "side eff.", "{:.0%}"), ("dup_insert_rate", "dup insert", "{:.0%}"),
            ("acc_with_dup", "acc|dup", "{:.0%}"), ("gone_rate", "gone", "{:.0%}"),
            ("redundant_rate", "redundant", "{:.0%}"), ("lost_update_rate", "lost upd.", "{:.1%}"),
            ("turns", "turns", "{:.1f}"), ("tokens", "tokens", "{:,.0f}")]
    width = max(len(k) for k in summary) + 2
    print(f"{'condition':<{width}}" + "".join(f"{h:>11}" for _, h, _ in cols))
    for label, s in summary.items():
        print(f"{label:<{width}}" + "".join(f"{f.format(s[k]):>11}" for k, _, f in cols))
    print("\ndup insert, gone, redundant, lost upd.: share of task-runs with at least one. "
          "acc|dup: accuracy of the task-runs with a duplicate insert.")
    for label, s in summary.items():
        if s["dup_gap_median_s"] is not None:
            print(f"{label}: duplicate inserts {s['dup_inserts_per_task']:.2f}/task-run, "
                  f"median {s['dup_gap_median_s']:.1f} s apart")
        if s["wal_per_task"]:
            print(f"{label}: per task-run " + ", ".join(f"{k[4:]} {v:.1f}" for k, v in s["wal_per_task"].items()))
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
