"""Build the hard-task page from analysis/hard64.json and analysis/frontier5.json.

    python analysis/build_hard64_page.py   # writes analysis/hard64-final.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "hard64.json"))
data["frontier"] = json.load(open(HERE / "frontier5.json"))
data.pop("examples", None)  # used to pick the quotes on the page, not needed in it
html = (HERE / "hard64.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "hard64-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
