"""Compare runs side by side with the paper's metrics (arXiv 2512.08296, section 4.1).

    python compare.py runs/A runs/B ...    # these runs
    python compare.py                      # the latest WorkBench run of each topology

Only tasks that every run completed are compared, so pass runs of the same task sample.
Overhead, error amplification and efficiency are relative to the single-agent run, if there is one.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ["single", "independent", "centralized", "decentralized", "hybrid"]


def load(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in open(run_dir / "results.jsonl")]
    name = config.get("topology", "single")
    if name == "agent-driven":  # one label per channel condition
        name = f"driven/{config.get('comm', 'all')}{'+coord' if config.get('coordinate') else ''}"
    return name, {r["id"]: r for r in rows}


def latest_runs():
    latest = {}
    for d in sorted((ROOT / "runs").glob("*-workbench-*")):
        if (d / "results.jsonl").exists():
            latest[load(d)[0]] = d  # sorted by timestamp, so the last one wins
    return [latest[t] for t in ORDER if t in latest]


def main():
    runs = [load(d) for d in (sys.argv[1:] or latest_runs())]
    if not runs:
        sys.exit("no runs found")
    common = set.intersection(*(set(rows) for _, rows in runs))
    print(f"{len(common)} tasks in common\n")

    stats = {}
    for topology, rows in runs:
        rs = [rows[i] for i in common]
        n = len(rs)
        turns = sum(r["turns"] for r in rs)
        stats[topology] = {
            "accuracy": sum(bool(r["correct"]) for r in rs) / n,
            "side_eff": sum(bool(r.get("side_effects")) for r in rs) / n,
            "turns": turns / n,
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
            "messages": sum(r.get("agent_messages", 0) for r in rs) / n,
            "density": sum(r.get("agent_messages", 0) for r in rs) / max(1, turns),
            "seconds": sum(r["seconds"] for r in rs) / n,
        }
    base = stats.get("single")
    for s in stats.values():
        if base:
            s["turn_ovh"] = (s["turns"] - base["turns"]) / base["turns"]
            s["token_ovh"] = (s["tokens"] - base["tokens"]) / base["tokens"]
            # Task-level error amplification: error rate relative to the single agent (E = 1 - accuracy).
            s["err_amp"] = (1 - s["accuracy"]) / (1 - base["accuracy"]) if base["accuracy"] < 1 else float("nan")
            # Coordination efficiency: success divided by relative turn count.
            s["effic"] = s["accuracy"] / (s["turns"] / base["turns"])

    cols = [("accuracy", "{:.0%}"), ("side_eff", "{:.0%}"), ("turns", "{:.1f}"), ("tokens", "{:,.0f}"),
            ("turn_ovh", "{:+.0%}"), ("token_ovh", "{:+.0%}"), ("messages", "{:.1f}"), ("density", "{:.2f}"),
            ("err_amp", "{:.2f}"), ("effic", "{:.2f}"), ("seconds", "{:.0f}")]
    cols = [(k, f) for k, f in cols if any(k in s for s in stats.values())]
    width = max(14, max(len(t) for t in stats) + 2)
    print(f"{'topology':<{width}}" + "".join(f"{k:>11}" for k, _ in cols))
    for topology, s in stats.items():
        print(f"{topology:<{width}}" + "".join(f"{f.format(s[k]) if k in s else '-':>11}" for k, f in cols))
    print("\nper task (+ = correct):")
    print(f"{'task':<36}" + "".join(f"{t[:12]:>13}" for t, _ in runs))
    for i in sorted(common):
        print(f"{i:<36}" + "".join(f"{'+' if rows[i]['correct'] else '.':>13}" for _, rows in runs))


if __name__ == "__main__":
    main()
