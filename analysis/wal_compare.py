"""Teams with the write-ahead log (--wal) against the hard-slice baselines' 10 runs each.

    python analysis/wal_compare.py                 # the WAL repeats (runs tagged wal01..wal10)
    python analysis/wal_compare.py runs/A runs/B   # these WAL runs (e.g. the pilot, tagged wal-p1)

For each system: mean accuracy over its runs (and the range of single runs), split by task size (3-4 vs 5+
changes) and by whether the task is cap-bound (its main action is needed more than 5 times, so one search,
which returns at most 5 results, can't find everything), plus tokens and seconds per task.

Then each WAL condition against its own system without the log, the single agent and decentralized:
- several WAL runs: a permutation test on per-run accuracy. If the log made no difference, the runs' labels
  (WAL or baseline) would be exchangeable; p(>=) is the share of all relabellings in which the WAL runs' mean
  is at least as far above the baseline's as observed, p(<=) at least as far below.
- one WAL run (e.g. the pilot): against the baseline's per-task solve rates p_i from its 10 runs. If the WAL
  team were no better, each task would be solved with probability p_i, so its solved count would follow the
  Poisson-binomial distribution of the p_i; p(>=) is the chance of solving at least as many tasks, p(<=) at
  most as many.
"""
from itertools import combinations
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from compare import load  # noqa: E402

# The run-directory kinds of the systems with 10 hard-slice runs (labelled as compare.py labels them).
BASELINES = ["sas", "independent", "centralized", "decentralized", "hybrid", "agent-driven-none",
             "agent-driven-text-coord", "agent-driven-all-coord"]
TASKS = json.loads((ROOT / "analysis" / "hard64_repeats.json").read_text())["tasks"]


def poisson_binomial(ps):
    """dist[k] = probability that exactly k of the independent events with probabilities ps happen."""
    dist = [1.0]
    for p in ps:
        dist = [a * (1 - p) + b * p for a, b in zip(dist + [0.0], [0.0] + dist)]
    return dist


def permutation(a, b):
    """One-sided p-values for mean(a) - mean(b) under random relabelling of the runs (exact)."""
    pooled, k, n = a + b, len(a), len(a) + len(b)
    observed = sum(a) / len(a) - sum(b) / len(b)
    total, above, below = sum(pooled), 0, 0
    count = 0
    for idx in combinations(range(n), k):
        x = sum(pooled[i] for i in idx)
        d = x / k - (total - x) / (n - k)
        above += d >= observed - 1e-9
        below += d <= observed + 1e-9
        count += 1
    return above / count, below / count


def system(run_dirs):
    runs = [load(d)[1] for d in run_dirs]
    runs = [r for r in runs if set(TASKS) <= set(r)]  # complete runs only
    if not runs:
        return None
    def acc(ids):
        return sum(sum(bool(r[i]["correct"]) for i in ids) / len(ids) for r in runs) / len(runs)
    groups = {"all": list(TASKS),
              "3-4": [i for i, t in TASKS.items() if t["changes"] <= 4],
              "5+": [i for i, t in TASKS.items() if t["changes"] >= 5],
              "cap": [i for i, t in TASKS.items() if t["cap"]],
              "not cap": [i for i, t in TASKS.items() if not t["cap"]]}
    per_run = [sum(bool(r[i]["correct"]) for i in TASKS) / len(TASKS) for r in runs]
    rows = [r[i] for r in runs for i in TASKS]
    return {"runs": len(runs), "acc": {k: acc(v) for k, v in groups.items()},
            "range": (min(per_run), max(per_run)), "per_run": per_run,
            "rate": {i: sum(bool(r[i]["correct"]) for r in runs) / len(runs) for i in TASKS},
            "tokens": sum(x["prompt_tokens"] + x["completion_tokens"] for x in rows) / len(rows),
            "seconds": sum(x["seconds"] for x in rows) / len(rows)}


def main():
    wal_dirs = sys.argv[1:] or sorted(glob.glob(str(ROOT / "runs" / "*-wal-workbench-*-wal[0-9][0-9]")))
    wal = {}
    for d in wal_dirs:
        if (Path(d) / "results.jsonl").exists():
            wal.setdefault(load(d)[0], []).append(d)
    systems = {}
    for kind in BASELINES:
        dirs = sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-workbench-gpt-oss-20b-rep??")))
        s = system(dirs) if dirs else None
        if s:
            systems[load(dirs[0])[0]] = s
    for label, dirs in wal.items():
        s = system(dirs)
        if s:
            systems[label] = s
    sizes = {k: sum(1 for t in TASKS.values() if f(t)) for k, f in
             {"3-4": lambda t: t["changes"] <= 4, "5+": lambda t: t["changes"] >= 5, "cap": lambda t: t["cap"],
              "not cap": lambda t: not t["cap"]}.items()}
    width = max(map(len, systems)) + 2
    print(f"{len(TASKS)} hard tasks: {sizes['3-4']} with 3-4 changes, {sizes['5+']} with 5+; "
          f"{sizes['cap']} cap-bound, {sizes['not cap']} not\n")
    print(f"{'system':<{width}}{'runs':>5}{'accuracy':>10}{'one run':>12}{'3-4':>7}{'5+':>7}{'cap':>7}"
          f"{'not cap':>9}{'tokens':>10}{'seconds':>9}")
    for name, s in systems.items():
        a, (lo, hi) = s["acc"], s["range"]
        one_run = f"{lo:.0%}-{hi:.0%}"
        print(f"{name:<{width}}{s['runs']:>5}{a['all']:>10.0%}{one_run:>12}"
              f"{a['3-4']:>7.0%}{a['5+']:>7.0%}{a['cap']:>7.0%}{a['not cap']:>9.0%}{s['tokens']:>10,.0f}{s['seconds']:>9.0f}")

    print("\nWAL condition vs baseline: mean solved tasks per run, and how likely a difference that large is if the "
          "log made no difference (p >=: WAL at least that much better; p <=: at least that much worse)")
    for name in wal:
        if name not in systems:
            continue
        w = systems[name]
        solved = round(w["acc"]["all"] * len(TASKS))
        for base in dict.fromkeys([name.removesuffix("+wal"), "single", "decentralized"]):
            if base not in systems:
                continue
            b = systems[base]
            if w["runs"] > 1 and b["runs"] > 1:
                above, below = permutation(w["per_run"], b["per_run"])
                test = f"permutation over {w['runs']}+{b['runs']} runs"
            else:
                dist = poisson_binomial([b["rate"][i] for i in TASKS])
                above, below = sum(dist[solved:]), sum(dist[:solved + 1])
                test = "one WAL run vs the baseline's per-task rates"
            print(f"  {name:<24} vs {base:<16} {w['acc']['all'] * len(TASKS):5.1f} vs {b['acc']['all'] * len(TASKS):5.1f}"
                  f"   p(>=) {above:.3f}   p(<=) {below:.3f}   ({test})")


if __name__ == "__main__":
    main()
