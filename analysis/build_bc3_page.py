"""Build the BrowseComp-Plus team comparison page from browsecomp_bc3.py's output.

    python analysis/browsecomp_bc3.py --tag bc3-r1 --out browsecomp_bc3_r1.json
    python analysis/build_bc3_page.py browsecomp_bc3_r1.json     # writes analysis/bc3-final.html

The page gets question ids and numbers only: BrowseComp's questions and answers must not leave /scratch.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / (sys.argv[1] if len(sys.argv) > 1 else "browsecomp_bc3.json")))
first = next(s for s in data["setups"] if s["setup"] == "Single agent")
data["questions"] = sorted(first["questions"], key=lambda q: int(q.split("-")[1]))
html = (HERE / "bc3.html").read_text().replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
(HERE / "bc3-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
