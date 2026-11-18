"""The enforcement ladder on AppWorld's 50 hard tasks: gpt-oss-120b, single agent and teams of 3 at L0-L6 (2026-11-18 ladder,
completion rule "all"), two runs each (tags hlad01, hlad02).

    python analysis/aw_hard_ladder.py            # prints the table, writes analysis/aw_hard_ladder.json
    python analysis/aw_hard_ladder.py --tags t3check   # other runs (here the 10-task check of 2026-11-17)

Per setup, averaged over the runs that exist so far (each run over the tasks it has finished):
- solved: share of tasks AppWorld counts as solved; tests: share of AppWorld's checks passed
- writes, duplicate writes (a record or item a teammate had already written, done again) and collisions (the same, but the
  write failed) per task
- per agent: share that made no change at all, share reminded to sign off, share whose last log event was a refusal
- refusals per task by rule; team time, model calls and tokens per task
Only ids and numbers are read (AppWorld task contents must not be redistributed).
"""
import glob
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAGS = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else ["hlad01", "hlad02"]
SETUPS = {"single": "sas-rc", "L0": "agent-driven-none-rc", **{f"L{i}": f"agent-driven-none-L{i}-rc" for i in range(1, 7)}}
RULES = ["intent", "relevant_catchup", "read_before_post", "read_before_write", "no_conflict"]


def run_dir(kind, tag):
    dirs = sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-appworld-test_challenge-gpt-oss-120b-{tag}")))
    return Path(dirs[-1]) if dirs else None


def task_stats(d, r):
    """Per-task numbers from the result row and (for teams) the trace."""
    c = r.get("comm", {})
    out = {"solved": float(bool(r["correct"])), "tests": r["tests_passed"] / r["tests"] if r.get("tests") else 0.0,
           "calls": r["turns"], "tokens_k": (r["prompt_tokens"] + r["completion_tokens"]) / 1000, "seconds": r["seconds"]}
    if not c:
        return out, None
    out.update({"writes": c.get("writes", 0), "dup": c.get("duplicate_writes", 0), "coll": c.get("collisions", 0),
                "team_s": c.get("team_seconds", 0), **{f"ref_{k}": c.get(f"wal_refused_{k}", 0) for k in RULES}})
    tr = json.load(open(d / "traces" / f"{r['id']}.json"))
    ev = tr["events"]
    agents = sorted({e["agent"] for e in ev if e["kind"] == "active"})
    writers = {e["agent"] for e in ev if e["kind"] == "api_write" and e["ok"] and e.get("api") != "complete_task"}
    last = {}
    for e in ev:
        if e["kind"] in ("wal_refused", "wal_intend") or (e["kind"] == "api_write" and e["ok"]):
            last[e["agent"]] = e["kind"]
    reminded = sum(any("You haven't called apis.supervisor.complete_task" in str(m.get("content")) for m in msgs
                       if m["role"] == "user") for msgs in tr["conversations"].values())
    out["writers"] = len(writers)
    per_agent = {"no_change": sum(a not in writers for a in agents), "reminded": reminded,
                 "ended_on_refusal": sum(v == "wal_refused" for v in last.values()), "agents": len(agents)}
    return out, per_agent


rows, per_run = {}, {}
for name, kind in SETUPS.items():
    stats, agents = [], Counter()
    for tag in TAGS:
        d = run_dir(kind, tag)
        res = d / "results.jsonl" if d else None
        if not res or not res.exists():
            continue
        results = [json.loads(line) for line in open(res)]
        per_run[f"{name} {tag}"] = {"tasks": len(results), "solved": sum(bool(r["correct"]) for r in results)}
        for r in results:
            s, a = task_stats(d, r)
            stats.append(s)
            agents.update(a or {})
    if not stats:
        continue
    mean = lambda k: round(st.mean(s[k] for s in stats if k in s), 3) if any(k in s for s in stats) else None  # noqa: E731
    row = {"tasks": len(stats), "solved": mean("solved"), "tests": mean("tests"), "calls": mean("calls"),
           "tokens_k": mean("tokens_k"), "seconds": round(mean("seconds"))}
    if agents:
        row.update({"writes": mean("writes"), "dup": mean("dup"), "coll": mean("coll"), "writers": mean("writers"),
                    "team_s": mean("team_s"),
                    **{f"ref_{k}": mean(f"ref_{k}") for k in RULES},
                    "no_change_agents": round(agents["no_change"] / agents["agents"], 3),
                    "reminded_agents": round(agents["reminded"] / agents["agents"], 3),
                    "ended_on_refusal": round(agents["ended_on_refusal"] / agents["agents"], 3)})
    row["tokens_k"] = round(row["tokens_k"])
    rows[name] = row

if "--tags" not in sys.argv:
    (ROOT / "analysis" / "aw_hard_ladder.json").write_text(json.dumps({"setups": rows, "runs": per_run}, indent=1))
cols = ["tasks", "solved", "tests", "writes", "dup", "coll", "writers", "no_change_agents", "reminded_agents",
        "ended_on_refusal", *(f"ref_{k}" for k in RULES), "seconds", "calls", "tokens_k"]
short = {"no_change_agents": "no_chg", "reminded_agents": "remind", "ended_on_refusal": "end_ref",
         "ref_intent": "r_int", "ref_relevant_catchup": "r_rel", "ref_read_before_post": "r_post",
         "ref_read_before_write": "r_write", "ref_no_conflict": "r_confl", "tokens_k": "tok_k"}
print("gpt-oss-120b, AppWorld 50 hard tasks; per task unless noted; agents' shares over all team agents")
print(f"{'':7s}" + "".join(f"{short.get(c, c):>8s}" for c in cols))
for name, row in rows.items():
    print(f"{name:7s}" + "".join(f"{'' if row.get(c) is None else row[c]:>8}" for c in cols))
print("runs:", ", ".join(f"{k}: {v['solved']}/{v['tasks']}" for k, v in per_run.items()))
