"""A fixed random sample of AppWorld's hardest tasks (difficulty 3), for agent.py --only.

    python analysis/aw_hard_tasks.py [--split test_challenge] [--n 50] [--seed 0]   # prints ids, comma-separated

Difficulty comes from the single-agent baseline run on that split (tag base02), which recorded AppWorld's own label
for every task. The same seed always gives the same sample. Only ids are read and printed (AppWorld task contents
must not be redistributed).
"""
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
arg = lambda name, default: type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default  # noqa: E731
split, n, seed = arg("--split", "test_challenge"), arg("--n", 50), arg("--seed", 0)
ids = sorted({json.loads(line)["id"] for d in glob.glob(str(ROOT / "runs" / f"*-sas-rc-appworld-{split}-*-base02"))
              for line in open(Path(d, "results.jsonl")) if json.loads(line).get("difficulty") == 3})
if len(ids) < n:
    sys.exit(f"only {len(ids)} difficulty-3 tasks found for {split}")
print(",".join(sorted(random.Random(seed).sample(ids, n))))
