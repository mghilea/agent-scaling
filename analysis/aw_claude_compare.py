"""Claude single agents (and Qwen3-Coder-30B) against gpt-oss-20b on a set of AppWorld tasks.

    python analysis/aw_claude_compare.py                     # dev "big" tasks (analysis/aw_big_tasks.py), tag big01
    python analysis/aw_claude_compare.py --hard              # 50 test_challenge difficulty-3 tasks (aw_hard_tasks.py), tag hard01
Writes analysis/aw_claude_compare[_hard].json and prints a table. While runs are going, each model is also compared
with gpt-oss on just the tasks it has finished.

For each of the 18 tasks: whether each Claude model solved it (one run each, tag big01), the gpt-oss-20b single agent
(base02, one run), and the gpt-oss-20b teams of the enforcement ladder (L0-L3, share of 5 runs solved). Also the
Claude runs' calls, tokens, dollars and fallbacks. Only ids and numbers are read (AppWorld task contents must not be
redistributed).
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARD = "--hard" in sys.argv
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ("hard01" if HARD else "big01")
SPLIT = "test_challenge" if HARD else "dev"
MODELS = ["claude-sonnet-5-5", "claude-opus-5-5",
          # open models, run with --repair-calls as the gpt-oss-20b baseline
          "qwen3-coder-30b", "gpt-oss-120b", "qwen3.8-27b", "qwen3.5-122b"]


def rows(pattern):
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / pattern))):
        for line in open(Path(d, "results.jsonl")):
            r = json.loads(line)
            out[r["id"]] = r
    return out


writes = defaultdict(list)
for d in glob.glob(str(ROOT / "runs" / "*-agent-driven-none-rc-appworld-dev-*-lad0[2-6]")):
    for line in open(Path(d, "results.jsonl")):
        r = json.loads(line)
        writes[r["id"]].append(r["comm"]["writes"])
if HARD:
    import subprocess
    big = subprocess.check_output([sys.executable, str(ROOT / "analysis" / "aw_hard_tasks.py")], text=True).strip().split(",")
else:
    big = sorted(t for t, w in writes.items() if st.mean(w) > 20)

claude = {m: rows(f"*-sas-*appworld-{SPLIT}-{m}-{TAG}") for m in MODELS}  # -sas-appworld- or -sas-rc-appworld-
single = rows(f"*-sas-rc-appworld-{SPLIT}-gpt-oss-20b-base02")
ladder = {}
for lvl in ("L0", "L1", "L2", "L3"):
    part = "" if lvl == "L0" else f"-{lvl}"
    solved = defaultdict(list)
    for d in glob.glob(str(ROOT / "runs" / f"*-agent-driven-none{part}-rc-appworld-dev-*-lad0[2-6]")):
        for line in open(Path(d, "results.jsonl")):
            r = json.loads(line)
            solved[r["id"]].append(bool(r["correct"]))
    ladder[lvl] = {} if HARD else {t: st.mean(v) for t, v in solved.items()}  # the ladder ran on dev only

tasks = []
for t in big:
    row = {"id": t, "team_writes_L0": round(st.mean(writes[t]), 1) if writes.get(t) else None,
           "gpt-oss single": bool(single[t]["correct"]) if t in single else None,
           **{f"gpt-oss team {lvl}": ladder[lvl].get(t) for lvl in ladder}}
    for m in MODELS:
        r = claude[m].get(t)
        row[m] = None if r is None else bool(r["correct"])
        if r is not None:
            row[f"{m} calls"], row[f"{m} $"] = r["turns"], r.get("dollars")
    tasks.append(row)


def share(key):
    v = [r[key] for r in tasks if r.get(key) is not None]
    return (st.mean(float(x) for x in v), len(v)) if v else (None, 0)


summary = {key: share(key) for key in [*MODELS, "gpt-oss single", *(f"gpt-oss team {l}" for l in ladder if ladder[l])]}
# gpt-oss single agent on just the tasks each Claude model has finished (fair while the runs are going)
same = {m: (st.mean(float(r["gpt-oss single"]) for r in tasks if r.get(m) is not None and r["gpt-oss single"] is not None)
            if any(r.get(m) is not None for r in tasks) else None) for m in MODELS}
runs = {m: {"tasks": len(claude[m]), "dollars": round(sum(r.get("dollars", 0) for r in claude[m].values()), 4),
            "calls": st.mean(r["turns"] for r in claude[m].values()) if claude[m] else None,
            "tokens": st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in claude[m].values()) if claude[m] else None,
            "seconds": st.mean(r["seconds"] for r in claude[m].values()) if claude[m] else None,
            "fallbacks": sum(r.get("fallbacks", 0) > 0 for r in claude[m].values()),
            "errors": sum(str(r["stop"]).startswith("error") for r in claude[m].values())} for m in MODELS}
out = {"tag": TAG, "split": SPLIT, "tasks": tasks, "summary": summary, "gpt_oss_on_same_tasks": same, "runs": runs}
(ROOT / "analysis" / f"aw_claude_compare{'_hard' if HARD else ''}.json").write_text(json.dumps(out, indent=1))

print(f"{len(big)} {'test_challenge difficulty-3 tasks (seed 0)' if HARD else 'dev big tasks (team without the log made over 20 writes)'}")
for key, (acc, n) in summary.items():
    print(f"  {key:22s} {'' if acc is None else f'{acc:6.1%}'}  ({n} tasks)")
for m in MODELS:
    if same[m] is not None:
        print(f"  gpt-oss single on the {summary[m][1]} tasks {m} has finished: {same[m]:.1%}")
for m, r in runs.items():
    print(f"  {m}: {r['tasks']} tasks, ${r['dollars']:.2f}, {r['calls'] or 0:.1f} calls, "
          f"{(r['tokens'] or 0) / 1000:.0f}k tokens, {r['seconds'] or 0:.0f} s per task, fallbacks {r['fallbacks']}, errors {r['errors']}")
