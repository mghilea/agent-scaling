"""Write-ahead log with and without piggybacking (--wal-piggyback): latency, model calls, refusals, conflicts.

    python analysis/piggyback.py      # prints the comparison and writes analysis/piggyback.json

The runs come from suites/wal-pb-{none,text,all}.txt (tags pb01..): each job runs one agent-driven setup with
the log, with and without piggybacking, side by side on one GPU, so a task-run is compared with its partner
from the same job (same task, same GPU load). Latency is the team's wall time per task.
"""
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from conflicts import conflicts  # noqa: E402

SETUPS = {"none": "No channels", "text-coord": "Text + coordinate", "all-coord": "All channels + coordinate"}


def load(pattern):
    """{(rep, task): (result row, conflict counts)} for the runs matching pattern."""
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / pattern))):
        rep = d.rsplit("-", 1)[-1]
        for line in open(Path(d, "results.jsonl")):
            r = json.loads(line)
            trace = json.loads(Path(d, "traces", f"{r['id']}.json").read_text())
            out[(rep, r["id"])] = (r, conflicts(trace))
    return out


def ci(diffs, n=4000, seed=0):
    """Mean and a 95% bootstrap interval, resampling task-runs."""
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def side(rows):
    rs = [r for r, _ in rows]
    cs = [c for _, c in rows]
    comm = lambda k: st.mean(r["comm"].get(k, 0) for r in rs)  # noqa: E731
    return {"task_runs": len(rs), "seconds": st.mean(r["seconds"] for r in rs),
            "seconds_median": st.median(r["seconds"] for r in rs),
            "turns": st.mean(r["turns"] for r in rs),
            "tokens": st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
            "accuracy": st.mean(bool(r["correct"]) for r in rs),
            "reads": comm("wal_reads"), "intents": comm("wal_intents"), "post_refused": comm("wal_append_refused"),
            "write_refused": comm("wal_refused_behind") + comm("wal_refused_no_intent") + comm("wal_refused_undeclared"),
            "piggybacked": comm("wal_piggybacked"), "piggybacked_entries": comm("wal_piggybacked_entries"),
            "conflicts": {k: st.mean(c[k] > 0 for c in cs) for k in ("dup_inserts", "gone", "redundant", "lost_updates")},
            "conflict_count": sum(c["dup_inserts"] + c["gone"] + c["redundant"] + c["lost_updates"] for c in cs)}


def main():
    out = {}
    for tag, label in SETUPS.items():
        plain = load(f"*-agent-driven-{tag}-wal-workbench-*-pb[0-9][0-9]")
        piggy = load(f"*-agent-driven-{tag}-wal-pb-workbench-*-pb[0-9][0-9]")
        keys = sorted(set(plain) & set(piggy))
        if not keys:
            continue
        a, b = side([plain[k] for k in keys]), side([piggy[k] for k in keys])
        secs = [piggy[k][0]["seconds"] - plain[k][0]["seconds"] for k in keys]
        turns = [piggy[k][0]["turns"] - plain[k][0]["turns"] for k in keys]
        reps = sorted({k[0] for k in keys})
        per_rep = {r: (st.mean(plain[k][0]["seconds"] for k in keys if k[0] == r),
                       st.mean(piggy[k][0]["seconds"] for k in keys if k[0] == r)) for r in reps}
        out[tag] = {"label": label, "pairs": len(keys), "reps": len(reps), "plain": a, "piggyback": b,
                    "seconds_diff": ci(secs), "turns_diff": ci(turns), "per_rep_seconds": per_rep}
        m, lo, hi = out[tag]["seconds_diff"]
        print(f"\n{label}: {len(keys)} paired task-runs over {len(reps)} runs")
        for name, s in (("WAL", a), ("WAL + piggyback", b)):
            print(f"  {name:16s} {s['seconds']:5.1f} s/task (median {s['seconds_median']:.1f})  {s['turns']:4.1f} calls  "
                  f"{s['tokens'] / 1000:5.0f}k tokens  reads {s['reads']:.1f}  posts refused {s['post_refused']:.2f}  "
                  f"writes refused {s['write_refused']:.2f}  piggybacked {s['piggybacked']:.1f}  accuracy {s['accuracy']:.0%}  "
                  f"conflicts {s['conflict_count']}  dup {s['conflicts']['dup_inserts']:.1%}")
        print(f"  latency change: {m:+.1f} s per task (95% CI {lo:+.1f} to {hi:+.1f}), {m / a['seconds']:+.0%}; "
              f"per run: " + ", ".join(f"{r} {p:.0f}->{q:.0f}" for r, (p, q) in per_rep.items()))
    (ROOT / "analysis" / "piggyback.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
