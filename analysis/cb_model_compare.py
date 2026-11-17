"""CooperBench core subset, single agent: the open models, two runs of the 10 pairs each, all with --repair-calls.

    python analysis/cb_model_compare.py          # prints the tables, writes analysis/cb_model_compare.json

Two tool settings: "bash" (the original tools: bash can edit the repository; gpt-oss-20b and Qwen3-Coder-30B, 2026-11-17
morning) and "edit" (--edit-tools: bash is read-only, files change only through the edit tools, as in the WAL teams;
gpt-oss-120b, Qwen3.5-122B-A10B, Qwen3.8-27B). A pair counts as solved when both features' hidden tests pass. Only ids
and numbers are read (CooperBench's dataset license is unknown: task contents stay off pages and out of git).
"""
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# label -> (run-directory kind, model, tags)
RUNS = {"gpt-oss-20b bash": ("sas-rc", "gpt-oss-20b", ["core01", "core02"]),
        "qwen3-coder-30b bash": ("sas-rc", "qwen3-coder-30b", ["qcore01", "qcore02"]),
        "gpt-oss-120b edit": ("sas-et-rc", "gpt-oss-120b", ["ecore01", "ecore02"]),
        "qwen3.5-122b edit": ("sas-et-rc", "qwen3.5-122b", ["ecore01", "ecore02"]),
        "qwen3.8-27b edit": ("sas-et-rc", "qwen3.8-27b", ["ecore01", "ecore02"])}


def load(kind, model, tag):
    dirs = sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-cooperbench-core-{model}-{tag}")))
    return [json.loads(line) for line in open(Path(dirs[-1], "results.jsonl"))] if dirs else []


runs = {(m, tag): load(kind, model, tag) for m, (kind, model, tags) in RUNS.items() for tag in tags}
RUNS = {m: tags for m, (_, _, tags) in RUNS.items()}
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
print(f"{'':22s}" + "".join(f"{m:>22s}" for m in summary))
for k in keys:
    print(f"{k:22s}" + "".join(f"{str(summary[m][k]):>22s}" for m in summary))
print("\nper pair (features passed, f1 f2; - = not run yet):")
cols = [f"{m} {tag}" for m, tags in RUNS.items() for tag in tags]
short = lambda c: c.replace("gpt-oss-", "oss").replace("qwen3-coder-30b", "coder").replace("qwen3.", "q").split(" ")[0] \
    + c.split(" ")[-1][-2:]  # noqa: E731
print(f"{'':40s}" + "".join(f"{short(c):>10s}" for c in cols))
for row in per_task:
    print(f"{row['id']:40s}" + "".join(f"{row[c] or '-':>10s}" for c in cols))
