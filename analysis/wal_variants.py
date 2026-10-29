"""Variants of the write-ahead log run side by side in the same jobs, compared task by task.

    python analysis/wal_variants.py --tag dr   # the log alone, + piggybacking, + piggybacking + drafts
                                                # (suites/wal-dr-*.txt); writes analysis/wal_variants_dr.json
    python analysis/wal_variants.py --tag ab   # ablations: the full log vs changes only, posting while behind,
                                                # nothing enforced (suites/wal-ab-*.txt)

Every task-run is paired with the same task in the same job under each other variant (same GPU load), as in
analysis/piggyback.py, whose loading, summaries and tests this reuses. For each agent-driven setup: latency,
model calls, tokens, log reads, rejected posts and writes (writes by reason), drafts recorded, conflicts and
accuracy per variant, and the paired change in seconds per task between variants.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from piggyback import SETUPS, ci, load, side, sign_flip  # noqa: E402

# Per run tag: the variants (run-name part after the setup, and a label) and the pairs of variants to compare.
TAGS = {
    "dr": ([("wal", "Log"), ("wal-pb", "Log + piggybacking"), ("wal-pb-dr", "Log + piggybacking + drafts")],
           [("wal", "wal-pb"), ("wal-pb", "wal-pb-dr"), ("wal", "wal-pb-dr")]),
    "ab": ([("wal", "Full log"), ("wal-changes-only", "Changes only (no intents)"),
            ("wal-post-anytime", "Posting while behind allowed"), ("wal-advisory", "Nothing enforced")],
           [("wal", "wal-changes-only"), ("wal", "wal-post-anytime"), ("wal", "wal-advisory")]),
}


def conflict_total(c):
    return c["dup_inserts"] + c["gone"] + c["redundant"] + c["lost_updates"]


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "dr"
    VARIANTS, STEPS = TAGS[tag]
    out = {"_variants": VARIANTS, "_steps": STEPS}
    for setup, label in SETUPS.items():
        runs = {v: load(f"*-agent-driven-{setup}-{v}-workbench-*-{tag}[0-9][0-9]") for v, _ in VARIANTS}
        keys = sorted(set.intersection(*(set(r) for r in runs.values())))
        if not keys:
            continue
        stats = {}
        for v, _ in VARIANTS:
            rows = [runs[v][k] for k in keys]
            stats[v] = side(rows)
            stats[v]["drafts_used"] = st.mean(r["comm"].get("wal_drafts_used", 0) for r, _, _ in rows)
            stats[v]["drafts_used_total"] = sum(r["comm"].get("wal_drafts_used", 0) for r, _, _ in rows)
        steps = {}
        for a, b in STEPS:
            secs = [runs[b][k][0]["seconds"] - runs[a][k][0]["seconds"] for k in keys]
            calls = [runs[b][k][0]["turns"] - runs[a][k][0]["turns"] for k in keys]
            confs = [conflict_total(runs[b][k][1]) - conflict_total(runs[a][k][1]) for k in keys]
            steps[f"{a}->{b}"] = {"seconds_diff": ci(secs), "turns_diff": ci(calls), "conflicts_diff": sum(confs),
                                  "conflicts_p_more": sign_flip(confs), "conflicts_p_fewer": sign_flip([-x for x in confs])}
        out[setup] = {"label": label, "pairs": len(keys), "reps": len({k[0] for k in keys}), "variants": stats,
                      "steps": steps}
        print(f"\n{label}: {len(keys)} task-runs per variant, {out[setup]['reps']} runs")
        for v, name in VARIANTS:
            s = stats[v]
            w = s["write_refused_total"]
            print(f"  {name:28s} {s['seconds']:5.1f} s/task  {s['turns']:4.1f} calls  {s['tokens'] / 1000:4.0f}k tokens  "
                  f"reads {s['reads']:.1f}  posts rejected {s['post_refused_total']:4d}  writes rejected "
                  f"{sum(w.values()):3d} (no intent {w['no_intent']}, behind {w['behind']}, undeclared {w['undeclared']})  "
                  f"drafts used {s['drafts_used_total']:3d}  conflicts {s['conflict_count']:3d}  accuracy {s['accuracy']:.0%}")
        for step, d in steps.items():
            m, lo, hi = d["seconds_diff"]
            print(f"  {step:20s} {m:+.1f} s per task (95% CI {lo:+.1f} to {hi:+.1f}); calls {d['turns_diff'][0]:+.1f}; "
                  f"conflicts {d['conflicts_diff']:+d}")
    (ROOT / "analysis" / f"wal_variants_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
