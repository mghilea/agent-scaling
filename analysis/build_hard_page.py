"""Build the hard-task evaluation page from analysis/hard_eval.json, hard_cap.json and the runs.

    python analysis/build_hard_page.py   # writes analysis/hard-tasks-final.html
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAG = {"Single agent": "sas", "Decentralized (paper rounds)": "decentralized",
        "Agent-driven, text only + coordinate": "agent-driven-text-coord",
        "Agent-driven, all channels + coordinate": "agent-driven-all-coord"}
PREFIX = {"3plus": "20260930-2000", "5plus": "20260930-20165"}

wbdir = Path(os.environ.get("WORKBENCH_DIR", f"/scratch/{os.environ.get('USER')}/WorkBench"))
pub = pd.read_csv(wbdir / "data/results/item_level_results.csv.gz")
pub = pub[pub.run_group == "revisited_2026"]
solve = pub.groupby("task_id").correct.mean().to_dict()
question = pub.groupby("task_id").task.first().to_dict()

grids = {}
for s, prefix in PREFIX.items():
    rows = {}
    for label, frag in FRAG.items():
        d = glob.glob(str(ROOT / f"runs/{prefix}*-{frag}-workbench-*"))[0]
        for r in map(json.loads, open(d + "/results.jsonl")):
            dom, i = r["id"].rsplit("-", 1)
            pid = f"{dom}-{int(i):03d}"
            e = rows.setdefault(r["id"], {"id": r["id"], "changes": len(r["expected"]), "solve": solve.get(pid),
                                          "question": question.get(pid, ""), "results": {}})
            e["results"][label] = bool(r["correct"])
    grids[s] = sorted(rows.values(), key=lambda e: (-e["changes"], e["id"]))

data = {"eval": json.load(open(HERE / "hard_eval.json"))["sets"], "cap": json.load(open(HERE / "hard_cap.json")),
        "grids": grids, "systems": list(FRAG)}
html = (HERE / "hard-tasks.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "hard-tasks-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
