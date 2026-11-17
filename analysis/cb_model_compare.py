"""CooperBench core subset, single agent: gpt-oss-20b (core01, core02) against Qwen3-Coder-30B (qcore01, qcore02).

    python analysis/cb_model_compare.py          # prints the tables, writes analysis/cb_model_compare.json

All four runs used the original tools (bash can edit the repository) and --repair-calls. A pair counts as solved when both
features' hidden tests pass. Only ids and numbers are read (CooperBench's dataset license is unknown: task contents stay
off pages and out of git).
"""
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"gpt-oss-20b": ["core01", "core02"], "qwen3-coder-30b": ["qcore01", "qcore02"]}


def load(model, tag):
    dirs = sorted(glob.glob(str(ROOT / "runs" / f"*-sas-rc-cooperbench-core-{model}-{tag}")))
    return [json.loads(line) for line in open(Path(dirs[-1], "results.jsonl"))] if dirs else []


runs = {(m, tag): load(m, tag) for m, tags in RUNS.items() for tag in tags}
ids = sorted({r["id"] for rows in runs.values() for r in rows})

summary = {}
for m, tags in RUNS.items():
    rows = [r for tag in tags for r in runs[(m, tag)]]
    if not rows:
        continue
    summary[m] = {
        "runs": ", ".join(f"{sum(r['correct'] for r in runs[(m, tag)])}/{len(runs[(m, tag)])}" for tag in tags),
        "pairs_solved": round(st.mean(r["correct"] for r in rows), 3),
        "features_passed": round(st.mean((r["feature1_passed"] + r["feature2_passed"]) / 2 for r in rows), 3),
        "at_least_one_feature": round(st.mean(r["feature1_passed"] or r["feature2_passed"] for r in rows), 3),
        "empty_patch": sum(r["patch_lines"] == 0 for r in rows),
        "out_of_turns": sum(r["stop"] == "max_turns" for r in rows),
        "errors": sum(str(r["stop"]).startswith("error") for r in rows),
        "turns": round(st.mean(r["turns"] for r in rows), 1),
        "tool_errors": round(st.mean(r["tool_errors"] for r in rows), 1),
        "tokens_k": round(st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / 1000),
        "seconds": round(st.mean(r["seconds"] for r in rows)),
        "tasks": len(rows),
    }

per_task = []
for t in ids:
    row = {"id": t}
    for (m, tag), rows in runs.items():
        r = next((r for r in rows if r["id"] == t), None)
        row[f"{m} {tag}"] = None if r is None else f"{int(r['feature1_passed'])}{int(r['feature2_passed'])}"
    per_task.append(row)

(ROOT / "analysis" / "cb_model_compare.json").write_text(json.dumps({"summary": summary, "per_task": per_task}, indent=1))

keys = ["runs", "pairs_solved", "features_passed", "at_least_one_feature", "empty_patch", "out_of_turns", "errors",
        "turns", "tool_errors", "tokens_k", "seconds", "tasks"]
print(f"{'':22s}" + "".join(f"{m:>26s}" for m in summary))
for k in keys:
    print(f"{k:22s}" + "".join(f"{str(summary[m][k]):>26s}" for m in summary))
print("\nper pair (features passed, f1 f2; - = not run yet):")
cols = [f"{m} {tag}" for m, tags in RUNS.items() for tag in tags]
print(f"{'':42s}" + "".join(f"{c.replace('gpt-oss-20b', 'oss').replace('qwen3-coder-30b', 'qwen'):>14s}" for c in cols))
for row in per_task:
    print(f"{row['id']:42s}" + "".join(f"{row[c] or '-':>14s}" for c in cols))
