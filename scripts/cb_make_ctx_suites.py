"""Suite files for the equal-context CooperBench ladder (user, 2026-11-24): flash (2-agent teams) and flash3 (3-agent
teams), single agent plus L0-L6, runs cx01 and cx02, as the fl01/fl02 ladder but with one context window per setup:
--context-total 131072 (gpt-oss-120b's full window) goes to the single agent whole and is split evenly among a team's
agents (65536 each for 2, 43690 for 3; agent.py drops the oldest tool outputs to stay within it). Call budgets are equal
per setup too: the single agent gets 100 model calls per feature (--max-turns 200 on pairs, 300 on triples), each team
agent 100 for its one.

    python scripts/cb_make_ctx_suites.py       # writes suites/cb-ctx/*.txt; one suite file = one job (5 chunks of 10 tasks)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "suites" / "cb-ctx"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.txt"):
    f.unlink()
for sub, size in (("flash", 2), ("flash3", 3)):
    for setup in ("single", "L0", "L1", "L2", "L3", "L4", "L5", "L6"):
        base = (f"--benchmark cooperbench --subset {sub} --edit-tools --max-turns {100 * size}" if setup == "single" else
                f"--benchmark cooperbench --subset {sub} --topology agent-driven --comm none"
                + ("" if setup == "L0" else f" --wal-level {setup}"))
        for run in ("cx01", "cx02"):
            (out / f"{sub}-{setup}-{run}.txt").write_text("".join(
                f"{base} --context-total 131072 --offset {off} --limit 10 --repair-calls --tag {run}\n" for off in range(0, 50, 10)))
print(len(list(out.glob("*.txt"))), "suite files in", out)
