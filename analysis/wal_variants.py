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
    load_cache = {}
    for setup, label in SETUPS.items():
        runs = {v: load(f"*-agent-driven-{setup}-{v}-workbench-*-{tag}[0-9][0-9]") for v, _ in VARIANTS}
        load_cache.update({(setup, v): r for v, r in runs.items()})
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
            dups = [runs[b][k][1]["dup_inserts"] - runs[a][k][1]["dup_inserts"] for k in keys]
            correct = [int(bool(runs[b][k][0]["correct"])) - int(bool(runs[a][k][0]["correct"])) for k in keys]
            steps[f"{a}->{b}"] = {"seconds_diff": ci(secs), "turns_diff": ci(calls), "conflicts_diff": sum(confs),
                                  "conflicts_p_more": sign_flip(confs), "conflicts_p_fewer": sign_flip([-x for x in confs]),
                                  "duplicates_diff": sum(dups), "duplicates_p_more": sign_flip(dups),
                                  "duplicates_p_fewer": sign_flip([-x for x in dups]),
                                  "accuracy_diff": sum(correct) / len(keys), "accuracy_p_up": sign_flip(correct),
                                  "accuracy_p_down": sign_flip([-x for x in correct])}
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
    # pooled over the setups: each variant against the first, paired task-runs from all three
    pooled = {}
    for a, b in STEPS:
        if a != VARIANTS[0][0]:
            continue
        diffs = {"correct": [], "conflicts": [], "dups": []}
        for setup in SETUPS:
            if setup not in out:
                continue
            ra = {k: v for k, v in load_cache[(setup, a)].items()}
            rb = load_cache[(setup, b)]
            for k in set(ra) & set(rb):
                diffs["correct"].append(int(bool(rb[k][0]["correct"])) - int(bool(ra[k][0]["correct"])))
                diffs["conflicts"].append(conflict_total(rb[k][1]) - conflict_total(ra[k][1]))
                diffs["dups"].append(rb[k][1]["dup_inserts"] - ra[k][1]["dup_inserts"])
        if diffs["correct"]:
            pooled[f"{a}->{b}"] = {"pairs": len(diffs["correct"]),
                                   "accuracy_diff": sum(diffs["correct"]) / len(diffs["correct"]),
                                   "accuracy_p_up": sign_flip(diffs["correct"]),
                                   "accuracy_p_down": sign_flip([-x for x in diffs["correct"]]),
                                   "conflicts_diff": sum(diffs["conflicts"]), "duplicates_diff": sum(diffs["dups"]),
                                   "duplicates_p_more": sign_flip(diffs["dups"]),
                                   "duplicates_p_fewer": sign_flip([-x for x in diffs["dups"]])}
    out["_pooled"] = pooled
    print("\npooled:", json.dumps(pooled, indent=1))
    (ROOT / "analysis" / f"wal_variants_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
