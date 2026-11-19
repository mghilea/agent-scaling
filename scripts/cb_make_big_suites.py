"""Suite files for the CooperBench scaling ladder (design A): sizes 1, 2, 3, 4, 6, 8 on the big<n> subsets, single agent
plus L0-L6 (size 1: single only), runs bg01 and bg02, each run's 10 tasks as two chunks of 5 (--offset/--limit). The
single agent's budget grows with the features it implements: --max-turns 100 x size (each team agent has 100 for one).

    python scripts/cb_make_big_suites.py       # writes suites/cb-big/*.txt; one suite file = one job

Packing (agents per model server): sizes up to 4 and every single-agent setup put a setup's 4 chunks in one job; teams
of 6 and 8 put each run's 2 chunks in its own job, so a server carries about 12-16 agents.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "suites" / "cb-big"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.txt"):
    f.unlink()
n_jobs = 0
for size in (1, 2, 3, 4, 6, 8):
    sub = f"big{size}"
    for setup in (["single"] if size == 1 else ["single", "L0", "L1", "L2", "L3", "L4", "L5", "L6"]):
        # the single agent gets 100 model calls per feature (user, 2026-11-18), as each team agent gets 100 for its one
        base = (f"--benchmark cooperbench --subset {sub} --edit-tools --max-turns {100 * size}" if setup == "single" else
                f"--benchmark cooperbench --subset {sub} --topology agent-driven --comm none"
                + ("" if setup == "L0" else f" --wal-level {setup}"))
        lines = {run: [f"{base} --offset {off} --limit 5 --repair-calls --tag {run}" for off in (0, 5)] for run in ("bg01", "bg02")}
        if size >= 6 and setup != "single":
            for run, ls in lines.items():
                (out / f"{sub}-{setup}-{run}.txt").write_text("\n".join(ls) + "\n"); n_jobs += 1
        else:
            (out / f"{sub}-{setup}.txt").write_text("\n".join(lines["bg01"] + lines["bg02"]) + "\n"); n_jobs += 1
print(n_jobs, "suite files in", out)
