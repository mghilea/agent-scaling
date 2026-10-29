"""Build the WorkBench concurrency page from analysis/concurrency.json (written by analysis/concurrency.py).

    python analysis/build_concurrency_page.py   # writes analysis/concurrency-final.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "concurrency.json"))
if (HERE / "overlaps.json").exists():  # from overlaps.py: would-be collisions between private copies
    data["overlaps"] = json.load(open(HERE / "overlaps.json"))
if (HERE / "piggyback.json").exists():  # from piggyback.py: the log with and without piggybacking, paired
    data["piggyback"] = json.load(open(HERE / "piggyback.json"))
for key, name in (("drafts", "wal_variants_dr.json"), ("ablations", "wal_variants_ab.json")):
    if (HERE / name).exists():  # from wal_variants.py --tag dr / ab: variants of the log run side by side
        data[key] = json.load(open(HERE / name))
if (HERE / "wal_timing.json").exists():  # from wal_timing.py: how long the log's locked operations take
    data["timing"] = json.load(open(HERE / "wal_timing.json"))
html = (HERE / "concurrency.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "concurrency-final.html").write_text(html)
print("written", len(html) // 1024, "KB")
