"""Agent-driven teams with the write-ahead log (--wal) against the hard-slice baselines' 10 runs each.

    python analysis/wal_compare.py                 # every run tagged wal-p1 or walNN
    python analysis/wal_compare.py runs/A runs/B   # these WAL runs

For each system: mean accuracy over its runs (and the range of single runs), split by task size (3-4 vs 5+
changes) and by whether the task is cap-bound (its main action is needed more than 5 times, so one search,
which returns at most 5 results, can't find everything), plus tokens and seconds per task.

Then each WAL condition against each baseline. A WAL condition with few runs is compared with a baseline's
per-task solve rates (from its 10 runs): if the WAL team were no better than the baseline, each task would be
solved with the baseline's rate p_i, so the WAL team's number of solved tasks per run would follow the
Poisson-binomial distribution of those p_i. p(>=) is the chance of solving at least as many tasks as the WAL
runs did on average, p(<=) of solving at most as many.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from compare import load  # noqa: E402

BASELINES = {
    "Single agent": "sas",
    "Decentralized": "decentralized",
    "Independent": "independent",
    "Driven, no channels": "agent-driven-none",
    "Driven, text + coord": "agent-driven-text-coord",
    "Driven, all + coord": "agent-driven-all-coord",
}
TASKS = json.loads((ROOT / "analysis" / "hard64_repeats.json").read_text())["tasks"]


def poisson_binomial(ps):
    """dist[k] = probability that exactly k of the independent events with probabilities ps happen."""
    dist = [1.0]
    for p in ps:
        dist = [a * (1 - p) + b * p for a, b in zip(dist + [0.0], [0.0] + dist)]
    return dist


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
            "range": (min(per_run), max(per_run)),
            "rate": {i: sum(bool(r[i]["correct"]) for r in runs) / len(runs) for i in TASKS},
            "tokens": sum(x["prompt_tokens"] + x["completion_tokens"] for x in rows) / len(rows),
            "seconds": sum(x["seconds"] for x in rows) / len(rows)}


def main():
    wal_dirs = sys.argv[1:] or sorted(glob.glob(str(ROOT / "runs" / "*-wal-workbench-*-wal*")))
    wal = {}
    for d in wal_dirs:
        if (Path(d) / "results.jsonl").exists():
            wal.setdefault(load(d)[0], []).append(d)
    systems = {}
    for name, kind in BASELINES.items():
        s = system(sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-workbench-gpt-oss-20b-rep??"))))
        if s:
            systems[name] = s
    for label, dirs in wal.items():
        s = system(dirs)
        if s:
            systems[label.replace("driven/", "Driven, ").replace("+", " + ")] = s
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

    print("\nWAL condition vs baseline: solved tasks per run, and how likely that is if the WAL team were no "
          "better (p >=) or no worse (p <=) than the baseline")
    for label in wal:
        name = label.replace("driven/", "Driven, ").replace("+", " + ")
        if name not in systems:
            continue
        w = systems[name]
        solved = round(w["acc"]["all"] * len(TASKS))
        for base in ("Single agent", "Decentralized", "Driven, no channels", "Driven, text + coord"):
            if base not in systems:
                continue
            dist = poisson_binomial([systems[base]["rate"][i] for i in TASKS])
            expected = sum(k * p for k, p in enumerate(dist))
            print(f"  {name:<28} vs {base:<22} {solved:>3} vs {expected:5.1f} expected   "
                  f"p(>=) {sum(dist[solved:]):.3f}   p(<=) {sum(dist[:solved + 1]):.3f}")


if __name__ == "__main__":
    main()
