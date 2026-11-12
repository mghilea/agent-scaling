"""Build the AppWorld enforcement-ladder page from analysis/aw_ladder.json (written by analysis/aw_ladder.py).

    python analysis/build_aw_ladder_page.py   # writes analysis/aw-ladder-final.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "aw_ladder.json"))
html = (HERE / "aw_ladder.html").read_text().replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
(HERE / "aw-ladder-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
