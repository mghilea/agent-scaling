"""Variants of the write-ahead log run side by side in the same jobs, compared task by task.

    python analysis/wal_variants.py --tag dr   # the log alone, + piggybacking, + piggybacking + drafts
                                                # (suites/wal-dr-*.txt); writes analysis/wal_variants_dr.json
    python analysis/wal_variants.py --tag ab   # ablations as a ladder: no log, then the log's checks added one at
                                                # a time up to the full log (suites/wal-ab-*.txt)

Every task-run is paired with the same task in the same job under each other variant (same GPU load), as in
analysis/piggyback.py, whose loading and summaries this reuses. For each agent-driven setup: latency, model calls,
tokens, log reads, rejected posts and writes (writes by reason), drafts recorded, conflicts and accuracy per
variant, and the paired change between variants. A reference without the log (the 10-run baselines, other jobs)
is compared task by task instead: each task's mean over its runs.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from piggyback import SETUPS, ci, load, side  # noqa: E402

# Per run tag: the variants (run-name part after the setup, and a label), the pairs of variants to compare, and
# optionally a reference from other jobs (name, label, run-name pattern with {setup}) compared with the first variant.
TAGS = {
    "dr": ([("wal", "Log"), ("wal-pb", "Log + piggybacking"), ("wal-pb-dr", "Log + piggybacking + drafts")],
           [("wal", "wal-pb"), ("wal-pb", "wal-pb-dr"), ("wal", "wal-pb-dr")], None),
    # the ladder: each variant adds one check to the one before (changes only also drops the optional intents);
    # the last pair is what intents and their checks add on top of the catch-up check
    "ab": ([("wal-advisory", "Log, nothing enforced"), ("wal-changes-only", "+ Catch up before writing"),
            ("wal-post-anytime", "+ Declare before writing"), ("wal", "+ Catch up before posting (full log)")],
           [("nolog", "wal-advisory"), ("wal-advisory", "wal-changes-only"), ("wal-changes-only", "wal-post-anytime"),
            ("wal-post-anytime", "wal"), ("wal-changes-only", "wal")],
           ("nolog", "No log", "*-agent-driven-{setup}-workbench-gpt-oss-20b-rep??")),
}


def conflict_total(c):
    return c["dup_inserts"] + c["gone"] + c["redundant"] + c["lost_updates"]


# per task-run: conflicts, whether it had a duplicate insert, whether it was solved, wall time, tokens
METRICS = {"conflicts": lambda r, c: conflict_total(c), "duplicates": lambda r, c: int(c["dup_inserts"] > 0),
           "correct": lambda r, c: int(bool(r["correct"])), "seconds": lambda r, c: r["seconds"],
           "tokens": lambda r, c: r["prompt_tokens"] + r["completion_tokens"]}


def sign_flips(diffs, n=20000, seed=0):
    """Two one-sided p-values for paired differences under random signs: (sum at least as large, at least as small)."""
    d = np.array([x for x in diffs if x], dtype=float)  # zeros add nothing whatever their sign
    if not len(d):
        return 1.0, 1.0
    rng, observed, sums = np.random.default_rng(seed), d.sum(), []
    for i in range(0, n, 2000):
        sums.append((rng.integers(0, 2, size=(min(2000, n - i), len(d)), dtype=np.int8) * 2 - 1) @ d)
    sums = np.concatenate(sums)
    return float(np.mean(sums >= observed - 1e-9)), float(np.mean(sums <= observed + 1e-9))


def units(a, b):
    """The paired units of two variants' {(rep, task): row}: task-runs from the same job, or else tasks (each
    task's runs under one variant against its runs under the other)."""
    common = sorted(set(a) & set(b))
    if common:
        return [([a[k]], [b[k]]) for k in common]
    by_task = defaultdict(lambda: ([], []))
    for k, row in a.items():
        by_task[k[1]][0].append(row)
    for k, row in b.items():
        by_task[k[1]][1].append(row)
    return [g for _, g in sorted(by_task.items()) if g[0] and g[1]]


def compare(pairs):
    """The change from a to b over paired units: mean per task-run, and both one-sided sign-flip p-values."""
    out = {"units": len(pairs)}
    for name, f in METRICS.items():
        diffs = [st.mean(f(*x[:2]) for x in gb) - st.mean(f(*x[:2]) for x in ga) for ga, gb in pairs]
        more, fewer = sign_flips(diffs)
        out[name] = {"diff": st.mean(diffs), "p_more": more, "p_fewer": fewer}
        if name == "seconds":
            out[name]["ci"] = ci(diffs)
    return out


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "dr"
    VARIANTS, STEPS, REF = TAGS[tag]
    out = {"_variants": ([REF[:2]] if REF else []) + VARIANTS, "_steps": STEPS}
    pooled_pairs = defaultdict(list)
    for setup, label in SETUPS.items():
        runs = {v: load(f"*-agent-driven-{setup}-{v}-workbench-*-{tag}[0-9][0-9]") for v, _ in VARIANTS}
        keys = sorted(set.intersection(*(set(r) for r in runs.values())))
        if not keys:
            continue
        runs = {v: {k: r[k] for k in keys} for v, r in runs.items()}  # task-runs every variant has
        if REF:
            runs[REF[0]] = load(REF[2].format(setup=setup))
        stats = {}
        for v, _ in out["_variants"]:
            rows = list(runs[v].values())
            stats[v] = side(rows)
            stats[v]["drafts_used"] = st.mean(r["comm"].get("wal_drafts_used", 0) for r, _, _ in rows)
            stats[v]["drafts_used_total"] = sum(r["comm"].get("wal_drafts_used", 0) for r, _, _ in rows)
        steps = {}
        for a, b in STEPS:
            pairs = units(runs[a], runs[b])
            pooled_pairs[f"{a}->{b}"] += pairs
            steps[f"{a}->{b}"] = compare(pairs)
        out[setup] = {"label": label, "pairs": len(keys), "reps": len({k[0] for k in keys}), "variants": stats,
                      "steps": steps}
        print(f"\n{label}: {len(keys)} task-runs per variant, {out[setup]['reps']} runs")
        for v, name in out["_variants"]:
            s = stats[v]
            w = s["write_refused_total"]
            print(f"  {name:38s} {s['task_runs']:4d} task-runs  {s['seconds']:5.1f} s/task  {s['turns']:4.1f} calls  "
                  f"{s['tokens'] / 1000:4.0f}k tokens  reads {s['reads']:.1f}  intents {s['intents']:.1f}  "
                  f"posts rejected {s['post_refused_total']:4d}  writes rejected {sum(w.values()):3d} "
                  f"(no intent {w['no_intent']}, behind {w['behind']}, undeclared {w['undeclared']})  "
                  f"drafts used {s['drafts_used_total']:3d}  conflicts {s['conflict_count'] / s['task_runs']:.2f}/task-run  "
                  f"dup {s['conflicts']['dup_inserts']:.1%}  accuracy {s['accuracy']:.0%}")
        for step, d in steps.items():
            m, lo, hi = d["seconds"]["ci"]
            print(f"  {step:34s} {d['units']:3d} units  {m:+5.1f} s (95% CI {lo:+.1f} to {hi:+.1f})  "
                  + "  ".join(f"{k} {d[k]['diff']:+.3f} (p {d[k]['p_more']:.3f}/{d[k]['p_fewer']:.3f})"
                              for k in ("conflicts", "duplicates", "correct")))
    # pooled over the setups: every step's paired units from all three
    out["_pooled"] = {step: compare(pairs) for step, pairs in pooled_pairs.items()}
    print("\npooled:")
    for step, d in out["_pooled"].items():
        print(f"  {step:34s} {d['units']:3d} units  {d['seconds']['diff']:+5.1f} s  "
              + "  ".join(f"{k} {d[k]['diff']:+.3f} (p {d[k]['p_more']:.3f}/{d[k]['p_fewer']:.3f})"
                          for k in ("conflicts", "duplicates", "correct", "tokens")))
    (ROOT / "analysis" / f"wal_variants_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
