"""Build the WorkBench concurrency page from analysis/concurrency.json (written by analysis/concurrency.py).

    python analysis/build_concurrency_page.py   # writes analysis/concurrency-final.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "concurrency.json"))
html = (HERE / "concurrency.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "concurrency-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
