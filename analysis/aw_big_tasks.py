"""The AppWorld dev tasks where the team without the log (ladder L0, runs lad02-lad06) made over 20 writes on average:
the "big" tasks where enforcement costs accuracy. Prints their ids, comma-separated, for agent.py --only.

    python analysis/aw_big_tasks.py            # e.g. agent.py --benchmark appworld --dataset dev --only "$(python analysis/aw_big_tasks.py)"
Only ids are read and printed (AppWorld task contents must not be redistributed).
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
writes = defaultdict(list)
for d in glob.glob(str(ROOT / "runs" / "*-agent-driven-none-rc-appworld-dev-*-lad0[2-6]")):
    for line in open(Path(d, "results.jsonl")):
        r = json.loads(line)
        writes[r["id"]].append(r["comm"]["writes"])
print(",".join(sorted(t for t, w in writes.items() if st.mean(w) > 20)))
