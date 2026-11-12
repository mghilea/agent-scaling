"""AppWorld enforcement ladder: a 3-agent team without the write-ahead log (L0) and with it at each level (wal2.py).

    python analysis/aw_ladder.py [--tag lad] [--dataset dev]    # writes analysis/aw_ladder.json

Runs come from suites/aw-ladder.txt (one job runs every level side by side on one GPU; tags lad02, lad03, ...), so
each task-run is paired with the same task in the same job at every other level. Per level: task and scenario goal
completion, conflicts between agents (duplicate writes and collisions, i.e. writes that failed because a teammate had
already made them), refusals by rule, claim releases, model calls, tokens, the team's time, and how much the agents
worked at the same time. Between levels: the paired change, with a two-sided sign-flip test (accuracy, conflicts) or a
bootstrap interval (seconds). The single agent on the same split (tag base02) is the reference.
No task contents are read or written (AppWorld's data must not be redistributed unencrypted): only ids and numbers.
"""
import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEVELS = [("L0", "No log"), ("L1", "Log, nothing enforced"), ("L2", "Claims required"),
          ("L3", "+ Catch up on every entry"), ("L4", "Claims + relevant catch-up + conflicts refused"),
          ("L4-strict", "L4 with catch-up on every entry")]
STEPS = [("L0", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4-strict", "L4"), ("L0", "L4"), ("L0", "L3")]
RULES = ["intent", "read_before_post", "read_before_write", "relevant_read", "no_conflict"]


def run_dirs(level, dataset, tag):
    part = "" if level == "L0" else f"-{level}"
    return sorted(d for d in glob.glob(str(ROOT / "runs" / f"*-agent-driven-none{part}-rc-appworld-{dataset}-*-{tag}[0-9][0-9]"))
                  if Path(d, "results.jsonl").exists())


def load(level, dataset, tag):
    """{(rep, task): row} over every run of this level."""
    out = {}
    for d in run_dirs(level, dataset, tag):
        rep = d.rsplit("-", 1)[-1]
        if rep == "lad01":  # a cancelled job with the earlier level definitions
            continue
        for line in open(Path(d, "results.jsonl")):
            r = json.loads(line)
            out[(rep, r["id"])] = r
    return out


def m(r, key):
    """One number from a result row (the team's comm block, or the row)."""
    c = r.get("comm") or {}
    if key == "correct":
        return int(bool(r["correct"]))
    if key == "tokens":
        return r["prompt_tokens"] + r["completion_tokens"]
    if key == "any_conflict":
        return int(c.get("conflicts", 0) > 0)
    if key == "refusals":
        return sum(c.get(f"wal_refused_{x}", 0) for x in RULES)
    if key in c:
        return c[key]
    return r.get(key, 0) or 0


def summary(rows):
    n = len(rows)
    mean = lambda k: st.mean(m(r, k) for r in rows) if rows else 0  # noqa: E731
    by_s = defaultdict(list)
    for r in rows:
        by_s[r["id"].rsplit("_", 1)[0], r.get("_rep")].append(bool(r["correct"]))
    full = [v for v in by_s.values() if len(v) == 3]
    out = {"task_runs": n, "accuracy": mean("correct"),
           "scenarios": sum(all(v) for v in full) / len(full) if full else None,
           "any_conflict": mean("any_conflict"), "conflicts": mean("conflicts"), "duplicates": mean("duplicate_writes"),
           "collisions": mean("collisions"), "writes": mean("writes"), "writers": mean("writers"),
           "seconds": mean("seconds"), "team_seconds": mean("team_seconds"), "turns": mean("turns"),
           "tokens": mean("tokens"), "overlap": mean("overlap"), "parallelism": mean("parallelism"),
           "completed": mean("completed"), "reminded": st.mean(int(bool(r.get("reminded"))) for r in rows) if rows else 0,
           "out_of_calls": st.mean(int(r.get("out_of_calls", 0) > 0) for r in rows) if rows else 0,
           "intents": mean("wal_intents"), "reads": mean("wal_reads"), "refusals": mean("refusals"),
           **{f"refused_{x}": mean(f"wal_refused_{x}") for x in RULES},
           "released_stopped": mean("wal_released_stopped"), "released_expired": mean("wal_released_expired"),
           "delivered": mean("wal_delivered"), "refusal_chains_3plus": mean("refusal_chains_3plus"),
           "refusal_chain_max": max((m(r, "refusal_chain_max") for r in rows), default=0),
           "stopped_after_refusal": mean("stopped_after_refusal"),
           "by_difficulty": {str(dif): {"n": len(g), "accuracy": st.mean(int(bool(r["correct"])) for r in g)}
                             for dif in sorted({r.get("difficulty") for r in rows if r.get("difficulty")})
                             for g in [[r for r in rows if r.get("difficulty") == dif]]}}
    return out


def sign_flips(diffs, n=20000, seed=0):
    d = np.array([x for x in diffs if x], dtype=float)
    if not len(d):
        return 1.0
    rng, obs, sums = np.random.default_rng(seed), d.sum(), []
    for i in range(0, n, 2000):
        sums.append((rng.integers(0, 2, size=(min(2000, n - i), len(d)), dtype=np.int8) * 2 - 1) @ d)
    sums = np.concatenate(sums)
    return float(min(1.0, 2 * min(np.mean(sums >= obs - 1e-9), np.mean(sums <= obs + 1e-9))))


def boot(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return [st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]]


def compare(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    out = {"pairs": len(keys)}
    for k in ("correct", "any_conflict", "conflicts", "refusals", "tokens", "turns"):
        diffs = [m(b[x], k) - m(a[x], k) for x in keys]
        out[k] = {"diff": st.mean(diffs), "p": sign_flips(diffs)}
    out["seconds"] = boot([m(b[x], "seconds") - m(a[x], "seconds") for x in keys])
    return out


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "lad"
    dataset = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "dev"
    data = {lvl: load(lvl, dataset, tag) for lvl, _ in LEVELS}
    # pair only task-runs that every level has (complete jobs and tasks)
    common = set.intersection(*(set(v) for v in data.values() if v)) if any(data.values()) else set()
    for lvl in data:
        data[lvl] = {k: {**r, "_rep": k[0]} for k, r in data[lvl].items() if k in common}
    out = {"dataset": dataset, "levels": LEVELS, "reps": sorted({k[0] for k in common}), "tasks": len({k[1] for k in common}),
           "summary": {lvl: summary(list(rows.values())) for lvl, rows in data.items()},
           "per_rep_accuracy": {lvl: {rep: st.mean(int(bool(r["correct"])) for k, r in rows.items() if k[0] == rep)
                                      for rep in sorted({k[0] for k in rows})} for lvl, rows in data.items()},
           "steps": {f"{a}->{b}": compare(data[a], data[b]) for a, b in STEPS}}
    single = [json.loads(line) for d in glob.glob(str(ROOT / "runs" / f"*-sas-rc-appworld-{dataset}-*-base02"))
              for line in open(Path(d, "results.jsonl"))]
    single = [r for r in single if r["id"] in {k[1] for k in common}]  # the same tasks as the teams
    if single:
        out["single"] = {"task_runs": len(single), "accuracy": st.mean(int(bool(r["correct"])) for r in single),
                         "seconds": st.mean(r["seconds"] for r in single),
                         "tokens": st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in single),
                         "turns": st.mean(r["turns"] for r in single)}
    (ROOT / "analysis" / "aw_ladder.json").write_text(json.dumps(out, indent=1))
    print(f"{dataset}: {out['tasks']} tasks x {len(out['reps'])} runs {out['reps']}")
    print(f"{'level':10s} {'acc':>6s} {'any confl':>9s} {'confl/run':>9s} {'refusals':>8s} {'secs':>6s} {'tokens':>7s} "
          f"{'overlap':>7s} {'writers':>7s}")
    for lvl, label in LEVELS:
        s = out["summary"][lvl]
        if not s["task_runs"]:
            continue
        print(f"{lvl:10s} {s['accuracy']:6.1%} {s['any_conflict']:9.1%} {s['conflicts']:9.2f} {s['refusals']:8.2f} "
              f"{s['seconds']:6.1f} {s['tokens'] / 1000:6.0f}k {s['overlap']:7.2f} {s['writers']:7.2f}")
    for step, c in out["steps"].items():
        if c:
            print(f"  {step:14s} acc {c['correct']['diff']:+.3f} (p {c['correct']['p']:.3f})  any conflict "
                  f"{c['any_conflict']['diff']:+.3f} (p {c['any_conflict']['p']:.3f})  seconds {c['seconds'][0]:+.1f} "
                  f"[{c['seconds'][1]:+.1f}, {c['seconds'][2]:+.1f}]")
    if single:
        print(f"single agent ({len(single)} task-runs): accuracy {out['single']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
