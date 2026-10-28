"""The paper's topologies: how much of the workers' changes would have collided in one shared workspace.

    python analysis/overlaps.py   # writes analysis/overlaps.json (a few minutes; needs WorkBench)

In independent, centralized, decentralized and hybrid each worker changes only its own private copy, so there
are no write conflicts at run time. To compare them with the agent-driven teams, every worker's write calls are
replayed in order on a fresh copy (skipping calls the write-ahead log refused) to get the changes it really made,
and the changes of different workers are compared as if they had shared one workspace:
- duplicate insert: two workers created the same thing (as analysis/conflicts.py defines it);
- overlapping update or delete: two workers changed the same record (in a shared workspace the second would
  have found it deleted, or already changed);
- lost update: two workers set the same field of the same record to different values.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import workbench as wb  # noqa: E402
from conflicts import insert_key, record_of  # noqa: E402
from wal import WRITES  # noqa: E402  (the sandbox's write tools, e.g. calendar_create_event)

TOPOLOGIES = ["independent", "centralized", "decentralized", "hybrid"]



def worker_changes(messages):
    """The changes one worker made in its copy: its write calls replayed in order on a fresh copy."""
    results = {m.get("tool_call_id"): m.get("content", "") for m in messages if m.get("role") == "tool"}
    sandbox, out = wb.Sandbox(), []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            name = tc["function"]["name"].split("<|")[0]
            if name not in WRITES or str(results.get(tc["id"], "")).startswith("[WAL]"):
                continue
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                before = len(sandbox.changes)
                sandbox.tools[name][0](**{k: v for k, v in args.items() if v is not None})
            except Exception:  # a malformed call changed nothing in the original run either
                continue
            if len(sandbox.changes) > before:
                out.append((name, {k: str(v) for k, v in args.items() if v is not None}))
    return out


def overlaps(trace):
    created, touched, fields = defaultdict(set), defaultdict(set), defaultdict(dict)
    for worker, messages in trace["conversations"].items():
        for name, args in worker_changes(messages):
            key = insert_key(name, args)
            if key:
                created[key].add(worker)
            rec = record_of(args) if ("update" in name or "delete" in name) else None
            if rec:
                touched[rec].add(worker)
                if "update" in name and args.get("field"):
                    fields[(rec, args["field"])].setdefault(worker, args.get("new_value"))
    return {"duplicate": sum(len(w) > 1 for w in created.values()),
            "overlap": sum(len(w) > 1 for w in touched.values()),
            "lost": sum(len(v) > 1 and len(set(v.values())) > 1 for v in fields.values())}


def main():
    out = {}
    for topo in TOPOLOGIES:
        for wal in (False, True):
            pat = f"*-{topo}-wal-workbench-*-wal[0-9][0-9]" if wal else f"*-{topo}-workbench-gpt-oss-20b-rep??"
            n, totals, with_any = 0, defaultdict(int), defaultdict(int)
            for d in sorted(glob.glob(str(ROOT / "runs" / pat))):
                for f in sorted(glob.glob(d + "/traces/*.json")):
                    c = overlaps(json.loads(Path(f).read_text()))
                    n += 1
                    for k, v in c.items():
                        totals[k] += v
                        with_any[k] += v > 0
            out[f"{topo}{'+wal' if wal else ''}"] = {"task_runs": n, "instances": dict(totals),
                                                     "task_runs_with": dict(with_any)}
            print(topo, "wal" if wal else "base", n, dict(totals), flush=True)
    (ROOT / "analysis" / "overlaps.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
