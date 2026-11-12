"""Health check and summary of AppWorld runs: errors, limits hit, and scores.

    python analysis/appworld_runs.py runs/<run dir> [...]     # default: every AppWorld run in runs/

Per run: task and scenario goal completion (overall and by AppWorld's difficulty 1-3), share of tests passed,
how many tasks called complete_task, stop reasons, agents that ran out of calls, and signs of harness trouble:
tasks that crashed, tool results that are harness errors ("[error] ..." from call_tool, as opposed to the code's
own exceptions, which AppWorld reports as "Execution failed"), truncated results, calls repaired by
--repair-calls, final replies that are a tool call written as text or empty, and the longest conversation.
Prints no task contents (AppWorld's data must not be redistributed unencrypted).
"""
import glob
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = re.compile(r"<\|call\|>|to=functions\.|<\|channel\|>|<\|start\|>")
REPAIR = ("Your last tool call was not in the right format", "Your last reply was empty")


def scenario(task_id):
    return task_id.rsplit("_", 1)[0]


def health(run_dir):
    rows = [json.loads(line) for line in open(Path(run_dir, "results.jsonl"))]
    trouble = Counter()
    longest = 0
    for r in rows:
        f = Path(run_dir, "traces", f"{r['id']}.json")
        conv = json.loads(f.read_text()) if f.exists() else []
        chars = 0
        for m in conv:
            c = str(m.get("content") or "")
            chars += len(c) + len(str(m.get("reasoning") or "")) + len(json.dumps(m.get("tool_calls") or []))
            if m.get("role") == "tool":
                trouble["tool results"] += 1
                trouble["harness errors ([error] ...)"] += c.startswith("[error]")
                trouble["code raised an exception"] += c.startswith("Execution failed")
                trouble["truncated results"] += "[truncated" in c
            if m.get("role") == "user" and c.startswith(REPAIR):
                trouble["repaired calls (nudged to call again)"] += 1
        longest = max(longest, chars)
        asst = [m for m in conv if m.get("role") == "assistant"]
        if asst and not asst[-1].get("tool_calls"):
            text = str(asst[-1].get("content") or "")
            trouble["final reply garbled or empty"] += (not text.strip() or text == "None" or bool(RAW.search(text)))
    return rows, trouble, longest


def summary(run_dir):
    rows, trouble, longest = health(run_dir)
    n = len(rows)
    by_s = {}
    for r in rows:
        by_s.setdefault(scenario(r["id"]), []).append(bool(r["correct"]))
    full = [v for v in by_s.values() if len(v) == 3]
    print(f"\n== {Path(run_dir).name}: {n} tasks")
    print(f"   task goal completion {sum(bool(r['correct']) for r in rows) / n:.1%}   "
          f"scenario goal completion {sum(all(v) for v in full) / len(full):.1%} ({len(full)} complete scenarios)"
          if full else f"   task goal completion {sum(bool(r['correct']) for r in rows) / n:.1%}")
    for d in sorted({r.get("difficulty") for r in rows}, key=lambda x: (x is None, x)):
        g = [r for r in rows if r.get("difficulty") == d]
        print(f"   difficulty {d}: {sum(bool(r['correct']) for r in g)}/{len(g)} solved")
    tests = [r["tests_passed"] / r["tests"] for r in rows if r.get("tests")]
    print(f"   tests passed per task: mean {st.mean(tests):.0%}; tasks with 0 tests passed {sum(t == 0 for t in tests)}, "
          f"all but one {sum(r['tests'] - r['tests_passed'] == 1 for r in rows if r.get('tests'))}")
    print(f"   called complete_task: {sum(bool(r.get('completed')) for r in rows)}/{n}; reminded to: "
          f"{sum(bool(r.get('reminded')) for r in rows)} (then completed: "
          f"{sum(bool(r.get('reminded')) and bool(r.get('completed')) for r in rows)})")
    print(f"   stop reasons: {dict(Counter(r['stop'].split(':')[0] for r in rows))}   "
          f"ran out of calls: {sum(r.get('out_of_calls', 0) > 0 for r in rows)}")
    print(f"   turns per task: median {st.median(r['turns'] for r in rows):.0f}, max {max(r['turns'] for r in rows)}; "
          f"seconds: median {st.median(r['seconds'] for r in rows):.0f}, max {max(r['seconds'] for r in rows):.0f}; "
          f"tokens per task: median {st.median(r['prompt_tokens'] + r['completion_tokens'] for r in rows) / 1000:.0f}k")
    print(f"   harness: {dict(trouble)}; longest conversation {longest / 1000:.0f}k characters")
    crashed = [r["id"] for r in rows if str(r["stop"]).startswith("error")]
    if crashed:
        print(f"   CRASHED: {crashed[:10]}" + (" ..." if len(crashed) > 10 else ""))
        print(f"   first error: {next(r['stop'] for r in rows if str(r['stop']).startswith('error'))[:300]}")


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(str(ROOT / "runs" / "*-appworld-*")))
    for d in dirs:
        if Path(d, "results.jsonl").exists():
            summary(d)


if __name__ == "__main__":
    main()
