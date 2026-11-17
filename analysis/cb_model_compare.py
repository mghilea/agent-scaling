"""CooperBench core subset, single agent: one run of the 10 pairs per model with the edit tools (the user wants one run
per model; the earlier bash-tools rows have two). Open models with --repair-calls.

    python analysis/cb_model_compare.py          # prints the tables, writes analysis/cb_model_compare.json

Two tool settings: "bash" (the original tools: bash can edit the repository; gpt-oss-20b and Qwen3-Coder-30B, 2026-11-17
morning) and "edit" (--edit-tools: bash is read-only, files change only through the edit tools, as in the WAL teams;
all five models). A pair counts as solved when both features' hidden tests pass. Only ids
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
        "gpt-oss-20b edit": ("sas-et-rc", "gpt-oss-20b", ["ecore01"]),
        "qwen3-coder-30b edit": ("sas-et-rc", "qwen3-coder-30b", ["ecore01"]),
        "gpt-oss-120b edit": ("sas-et-rc", "gpt-oss-120b", ["ecore01"]),
        "qwen3.5-122b edit": ("sas-et-rc", "qwen3.5-122b", ["ecore01"]),
        "qwen3.8-27b edit": ("sas-et-rc", "qwen3.8-27b", ["ecore01"]),
        # Claude through the Anthropic API (claude.sbatch; no --repair-calls, which is for the vLLM agent)
        "claude-sonnet-5-5 edit": ("sas-et", "claude-sonnet-5-5", ["ecore01"]),
        "claude-opus-5-5 edit": ("sas-et", "claude-opus-5-5", ["ecore01"])}


def load(kind, model, tag):
    """Every run directory with this tag, oldest first, a later result for the same pair replacing an earlier one (a run
    that stopped part-way is completed by a second job on the missing pairs, with the same tag)."""
    rows = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-cooperbench-core-{model}-{tag}"))):
        res = Path(d, "results.jsonl")
        for line in (open(res) if res.exists() else []):  # none yet: no pair finished
            r = json.loads(line)
            rows[r["id"]] = r
    return list(rows.values())


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
        "dollars": round(sum(r.get("dollars", 0) or 0 for r in rows), 2),
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
        "turns", "tool_errors", "tokens_k", "seconds", "dollars", "tasks"]
print(f"{'':22s}" + "".join(f"{m:>24s}" for m in summary))
for k in keys:
    print(f"{k:22s}" + "".join(f"{str(summary[m][k]):>24s}" for m in summary))
print("\nper pair (features passed, f1 f2; - = not run yet):")
cols = [f"{m} {tag}" for m, tags in RUNS.items() for tag in tags]
short = lambda c: c.replace("gpt-oss-", "oss").replace("qwen3-coder-30b", "coder").replace("qwen3.", "q").replace("claude-sonnet-5-5", "sonnet").replace("claude-opus-5-5", "opus").split(" ")[0] \
    + c.split(" ")[-1][-2:]  # noqa: E731
print(f"{'':40s}" + "".join(f"{short(c):>10s}" for c in cols))
for row in per_task:
    print(f"{row['id']:40s}" + "".join(f"{row[c] or '-':>10s}" for c in cols))
