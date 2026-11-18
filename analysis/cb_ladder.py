"""The enforcement ladder on CooperBench: gpt-oss-120b, single agent and teams at L0-L6 (2026-11-18 ladder), on core
(10 feature pairs, 2-agent teams) and core3 (the same pairs plus a third feature, 3-agent teams), runs tagged cl01-cl10.

    python analysis/cb_ladder.py                    # prints the tables, writes analysis/cb_ladder.json
    python analysis/cb_ladder.py --tags t2check     # other runs (here the 2026-11-17 2-agent check)

Per subset and setup, over the runs found so far (each over the pairs or triples it has finished):
- solved: share of tasks with every feature's tests passing (mean over runs, and the spread across runs: sd);
  features: share of features passing
- per task: edits, edits interleaved with a teammate's on the same file (crossed), whole-file rewrites over a teammate's
  edit (overwrites), refusals by rule, wal_wait calls and seconds waited, refused edits never made (unmade at the end),
  team time, model calls, tokens
- agents that made no edit (share of team agents)
Only ids and numbers are read (CooperBench's dataset license is unknown).
"""
import glob
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAGS = (sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv
        else [f"cl{i:02d}" for i in range(1, 11)])
SETUPS = {"single": "sas-et-rc", "L0": "agent-driven-none-rc", **{f"L{i}": f"agent-driven-none-L{i}-rc" for i in range(1, 7)}}
RULES = ["intent", "relevant_catchup", "read_before_post", "read_before_write", "no_conflict"]


def results(kind, subset, tag):
    rows = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-cooperbench-{subset}-gpt-oss-120b-{tag}"))):
        res = Path(d, "results.jsonl")
        for line in (open(res) if res.exists() else []):
            r = json.loads(line)
            rows[r["id"]] = (Path(d), r)
    return list(rows.values())


def per_task(d, r):
    n = r.get("features") or 2
    s = {"solved": float(bool(r["correct"])), "features": (r.get("features_passed") or 0) / n,
         "calls": r["turns"], "tokens_k": (r["prompt_tokens"] + r["completion_tokens"]) / 1000, "seconds": r["seconds"]}
    c = r.get("comm")
    if not c:
        return s, None
    s.update({"edits": c.get("edits", 0), "crossed": c.get("crossed_edits", 0), "overwrites": c.get("overwrites", 0),
              "unmade_end": c.get("unmade_at_end", 0), "team_s": c.get("team_seconds", 0),
              **{f"ref_{k}": c.get(f"wal_refused_{k}", 0) for k in RULES}})
    tr = json.load(open(d / "traces" / f"{r['id']}.json"))
    ev = tr["events"]
    waits = [e for e in ev if e["kind"] == "wal_wait"]
    s.update({"waits": len(waits), "wait_s": sum(e["waited"] for e in waits)})
    writers = {e["agent"] for e in ev if e["kind"] == "edit" and e["ok"]}
    return s, {"agents": len(tr["features"]), "no_edit": len(tr["features"]) - len(writers)}


out = {}
for subset in ("core", "core3"):
    for name, kind in SETUPS.items():
        stats, agents, by_run = [], Counter(), []
        for tag in TAGS:
            rows = results(kind, subset, tag)
            if not rows:
                continue
            run = [per_task(d, r) for d, r in rows]
            by_run.append(st.mean(s["solved"] for s, _ in run))
            stats += [s for s, _ in run]
            for _, a in run:
                agents.update(a or {})
        if not stats:
            continue
        mean = lambda k: round(st.mean(s[k] for s in stats if k in s), 3) if any(k in s for s in stats) else None  # noqa: E731
        row = {"runs": len(by_run), "tasks": len(stats), "solved": mean("solved"),
               "solved_sd": round(st.stdev(by_run), 3) if len(by_run) > 1 else None, "features": mean("features"),
               "calls": mean("calls"), "tokens_k": round(mean("tokens_k")), "seconds": round(mean("seconds"))}
        if agents:
            row.update({k: mean(k) for k in ("edits", "crossed", "overwrites", "waits", "wait_s", "unmade_end", "team_s")},
                       **{f"ref_{k}": mean(f"ref_{k}") for k in RULES},
                       no_edit_agents=round(agents["no_edit"] / agents["agents"], 3))
        out[f"{subset} {name}"] = row

if "--tags" not in sys.argv:
    (ROOT / "analysis" / "cb_ladder.json").write_text(json.dumps(out, indent=1))
cols = ["runs", "tasks", "solved", "solved_sd", "features", "edits", "crossed", "overwrites", "no_edit_agents",
        *(f"ref_{k}" for k in RULES), "waits", "wait_s", "unmade_end", "team_s", "calls", "tokens_k"]
short = {"solved_sd": "sd", "features": "feats", "overwrites": "overwr", "no_edit_agents": "no_edit",
         "ref_intent": "r_int", "ref_relevant_catchup": "r_rel", "ref_read_before_post": "r_post",
         "ref_read_before_write": "r_write", "ref_no_conflict": "r_confl", "unmade_end": "unmade", "tokens_k": "tok_k"}
print("gpt-oss-120b, CooperBench core (2-agent teams) and core3 (3-agent teams); per task unless noted")
print(f"{'':13s}" + "".join(f"{short.get(c, c):>8s}" for c in cols))
for name, row in out.items():
    print(f"{name:13s}" + "".join(f"{'' if row.get(c) is None else row[c]:>8}" for c in cols))
