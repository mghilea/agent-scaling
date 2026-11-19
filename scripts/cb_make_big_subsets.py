"""Nested feature groups for the 1/2/3/4/6/8-agent CooperBench scaling runs (design A, user 2026-11-18).

    python scripts/cb_make_big_subsets.py      # reads analysis/cb_gold_check_big_all_<i>.json, writes cooperbench/subsets/big{1,2,3,4,6,8}.json

The tasks: the 10 with at least 8 features (cooperbench/subsets/big_all.json). A feature is usable if its own
reference patch passes its tests and it also passes under the task's combined reference (all features merged), so the
features of a group can all be satisfied together. Per task, the usable features in ascending order; the first n form the
n-agent group (n = 1, 2, 3, 4, 6, 8): each larger team gets the same task with features added (big1 is for the single
agent only). A task with fewer than 8 usable features is left out of all three, so the three sizes share their
tasks. Feature ids only.
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(ROOT / "analysis" / "cb_gold_check_big_all_*.json"))) + \
        ([str(ROOT / "analysis" / "cb_gold_check_big_all.json")] if (ROOT / "analysis" / "cb_gold_check_big_all.json").exists() else []):
    rows += json.load(open(f))
spec = json.load(open(ROOT / "cooperbench" / "subsets" / "big_all.json"))["tasks"]
by_id = {r["id"]: r for r in rows}
SIZES = (1, 2, 3, 4, 6, 8)
groups, report = {n: [] for n in SIZES}, []
for t in spec:
    feats = t["groups"][0]
    tid = f"{t['repo']}-{t['task_id']}-" + "-".join(f"f{f}" for f in feats)
    r = by_id.get(tid)
    if r is None:
        report.append((tid, "not checked yet"))
        continue
    combined = r.get("combined")
    if not isinstance(combined, list):
        report.append((tid, f"combined check failed to run: {combined}"))
        continue
    usable = [f for i, f in enumerate(feats) if r.get(f"f{f}_own") is True and combined[i]]
    report.append((tid, f"{len(usable)} of {len(feats)} usable: {usable}"))
    if len(usable) >= 8:
        for n in SIZES:
            groups[n].append({"repo": t["repo"], "task_id": t["task_id"], "groups": [usable[:n]]})
for tid, msg in report:
    print(f"{tid[:60]:60s} {msg}")
for n, tasks in groups.items():
    (ROOT / "cooperbench" / "subsets" / f"big{n}.json").write_text(json.dumps({
        "description": f"{n}-agent CooperBench scaling: the tasks with >= 8 jointly usable features, each with its first {n} "
                       "(nested: big1 within big2 ... within big8; scripts/cb_make_big_subsets.py). Feature ids only.",
        "tasks": tasks}, indent=1))
    print(f"big{n}: {len(tasks)} tasks")
