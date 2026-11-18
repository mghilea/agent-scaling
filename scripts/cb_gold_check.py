"""Do CooperBench's own reference solutions pass their tests in our setup (Apptainer, cooperbench_bench.score_features)?

    $COOPERBENCH_PYTHON scripts/cb_gold_check.py [--subset core]

For each feature pair of the subset: each feature's reference patch (feature<f>/feature.patch) scored on that feature's
tests alone, then the task's combined.patch scored on both features' tests. A reference that fails points to our
setup (or the dataset), not the models. Writes analysis/cb_gold_check.json (ids and pass/fail only: the dataset's
license is unknown, so no task contents) and each failing test output to runs/cb_gold/ (not in git).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cooperbench_bench as cb  # noqa: E402

subset = sys.argv[sys.argv.index("--subset") + 1] if "--subset" in sys.argv else "core"
out_dir = ROOT / "runs" / "cb_gold"
out_dir.mkdir(parents=True, exist_ok=True)
results = []
for t in cb.load_tasks(subset):
    td = cb.DATASET / t["repo"] / f"task{t['task_id']}"
    row = {"id": t["id"]}
    for f in t["features"]:
        try:
            r = cb.score_features({**t, "features": [f]}, (td / f"feature{f}" / "feature.patch").read_text())
            row[f"f{f}_own"] = bool(r["features_passed"])
            if not r["features_passed"]:
                (out_dir / f"{t['id']}-f{f}-own.txt").write_text(r["_test_output"].get(f"feature{f}", ""))
        except Exception as e:
            row[f"f{f}_own"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    if (td / "combined.patch").exists():
        try:
            r = cb.score_features(t, (td / "combined.patch").read_text())
            row["combined"] = [bool(r[f"feature{i + 1}_passed"]) for i in range(len(t["features"]))]
            for i, f in enumerate(t["features"]):
                if not r[f"feature{i + 1}_passed"]:
                    (out_dir / f"{t['id']}-f{f}-combined.txt").write_text(r["_test_output"].get(f"feature{f}", ""))
        except Exception as e:
            row["combined"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    results.append(row)
    print(json.dumps(row), flush=True)
(ROOT / "analysis" / f"cb_gold_check{'' if subset == 'core' else '_' + subset}.json").write_text(json.dumps(results, indent=1))
print("done")
