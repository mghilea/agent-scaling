"""Build the AppWorld enforcement-ladder page from analysis/aw_ladder.json (written by analysis/aw_ladder.py).

    python analysis/build_aw_ladder_page.py   # writes analysis/aw-ladder-final.html
"""
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "aw_ladder.json"))
# for the "partial results" banner: tasks in the split, and runs planned (lad02-lad06)
data["total_tasks"] = {"dev": 57, "test_normal": 168, "test_challenge": 417}.get(data["dataset"])
data["expected_runs"] = 5
data["built"] = datetime.now().strftime("%Y-%m-%d %H:%M")
html = (HERE / "aw_ladder.html").read_text().replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
(HERE / "aw-ladder-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
