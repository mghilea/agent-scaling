"""The enforcement ladder on CooperBench: gpt-oss-120b, single agent and teams at L0-L6 (2026-11-18 ladder), on flash
(50 feature pairs over 20 tasks, 2-agent teams) and flash3 (the same pairs plus a third feature, 3-agent teams), two
runs tagged fl01 and fl02 (each run's 50 tasks in 5 chunks of 10, merged by tag).

    python analysis/cb_ladder.py                    # prints the tables, writes analysis/cb_ladder.json
    python analysis/cb_ladder.py --subsets core --tags t2check     # other runs (here the 2026-11-17 2-agent check)
    python analysis/cb_ladder.py --ctx              # the equal-context ladder (runs cx01-cx10; writes cb_ladder_ctx.json)

--ctx reads the runs made with --context-total 131072 (2026-11-24): one 131k-token window per setup, the single agent's
whole and split evenly among a team's agents, which drop their oldest tool outputs to stay within it; the single agent
also gets 100 calls per feature. Its extra columns: tool outputs dropped per task (drops), the share of agents that had
to drop any (trimmed), and the largest prompt in thousands of tokens (max_pk, the task's largest over its agents).

Per subset and setup, over the runs found so far (each over the pairs or triples it has finished):
- task runs that errored before the agents started (the task image couldn't be unpacked) are left out and counted
  (err); a later run of the same task and tag replaces them. Other errors (e.g. over the context window) count as failures
- solved: share of tasks with every feature's tests passing (mean over runs, and the spread across runs: sd);
  features: share of features passing
- per task: edits, edits interleaved with a teammate's on the same file (crossed), whole-file rewrites over a teammate's
  edit (overwrites), refusals by rule, wal_wait calls and seconds waited, refused edits never made (unmade at the end),
  team time, model calls, tokens
- agents that made no edit (share of team agents)
- solved with a 95% interval, and paired comparisons: setup A minus setup B on the tasks both ran, each task's solved
  rate averaged over its runs first, so task difficulty cancels; intervals bootstrap over tasks (10,000 resamples)
Only ids and numbers are read (CooperBench's dataset license is unknown).
"""
import glob
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTX = "--ctx" in sys.argv
TAGS = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else [f"cx{i:02d}" for i in range(1, 11)] if CTX else ["fl01", "fl02"]
# the 4/6/8-agent scaling runs: python analysis/cb_ladder.py --subsets big4,big6,big8 --tags bg01,bg02
SUBSETS = sys.argv[sys.argv.index("--subsets") + 1].split(",") if "--subsets" in sys.argv else ["flash", "flash3"]
SETUPS = {"single": "sas-et-rc", "L0": "agent-driven-none-rc", **{f"L{i}": f"agent-driven-none-L{i}-rc" for i in range(1, 7)}}
if CTX:
    SETUPS = {k: v + "-ctx128k" for k, v in SETUPS.items()}
RULES = ["intent", "relevant_catchup", "read_before_post", "read_before_write", "no_conflict"]


def results(kind, subset, tag):
    rows = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / f"*-{kind}-cooperbench-{subset}-gpt-oss-120b-{tag}"))):
        res = Path(d, "results.jsonl")
        for line in (open(res) if res.exists() else []):
            r = json.loads(line)
            # a later run of the same task replaces an earlier one, unless the later one errored and the earlier didn't
            if r["id"] not in rows or not infra_error(r):
                rows[r["id"]] = (Path(d), r)
    return list(rows.values())


INFRA = ("error: CalledProcessError: Command '['apptainer'", "error: RuntimeError: building ")  # the task image wasn't built


def infra_error(r):
    return str(r["stop"]).startswith(INFRA)


def ran(rows):
    """Task runs that ran: one that errored before the agents started (its task image couldn't be unpacked) is no
    result. Any other error (e.g. a model call over the context window) is the run's outcome and counts as a failure."""
    return [(d, r) for d, r in rows if not infra_error(r)]


def per_task(d, r):
    n = r.get("features") or 2
    s = {"solved": float(bool(r["correct"])), "features": (r.get("features_passed") or 0) / n,
         "calls": r["turns"], "tokens_k": (r["prompt_tokens"] + r["completion_tokens"]) / 1000, "seconds": r["seconds"]}
    c = r.get("comm")
    if CTX:  # single agent: its own row; team: the comm's totals over agents
        src = c or r
        s.update({"drops": src.get("context_drops", 0), "max_pk": src.get("max_prompt_tokens", 0) / 1000})
    if not c:
        return s, ({"agents": 1, "trimmed": int(r.get("context_drops", 0) > 0)} if CTX else None)
    s.update({"edits": c.get("edits", 0), "crossed": c.get("crossed_edits", 0), "overwrites": c.get("overwrites", 0),
              "unmade_end": c.get("unmade_at_end", 0), "team_s": c.get("team_seconds", 0),
              **{f"ref_{k}": c.get(f"wal_refused_{k}", 0) for k in RULES}})
    tr = json.load(open(d / "traces" / f"{r['id']}.json"))
    ev = tr["events"]
    waits = [e for e in ev if e["kind"] == "wal_wait"]
    s.update({"waits": len(waits), "wait_s": sum(e["waited"] for e in waits)})
    writers = {e["agent"] for e in ev if e["kind"] == "edit" and e["ok"]}
    return s, {"agents": len(tr["features"]), "no_edit": len(tr["features"]) - len(writers),
               **({"trimmed": c.get("agents_trimmed", 0)} if CTX else {})}


out, by_task = {}, {}
for subset in SUBSETS:
    for name, kind in SETUPS.items():
        stats, agents, by_run, errored = [], Counter(), [], 0
        for tag in TAGS:
            rows = results(kind, subset, tag)
            errored += len(rows) - len(ran(rows))
            rows = ran(rows)
            if not rows:
                continue
            run = [per_task(d, r) for d, r in rows]
            for (d, r), (s, _) in zip(rows, run):
                by_task.setdefault((subset, name), {}).setdefault(r["id"], []).append(s["solved"])
            by_run.append(st.mean(s["solved"] for s, _ in run))
            stats += [s for s, _ in run]
            for _, a in run:
                agents.update(a or {})
        if not stats:
            continue
        mean = lambda k: round(st.mean(s[k] for s in stats if k in s), 3) if any(k in s for s in stats) else None  # noqa: E731
        row = {"runs": len(by_run), "tasks": len(stats), "errored": errored, "solved": mean("solved"),
               "solved_sd": round(st.stdev(by_run), 3) if len(by_run) > 1 else None, "features": mean("features"),
               "calls": mean("calls"), "tokens_k": round(mean("tokens_k")), "seconds": round(mean("seconds"))}
        if CTX:
            row.update(drops=mean("drops"), max_pk=round(mean("max_pk"), 1),
                       trimmed=round(agents["trimmed"] / agents["agents"], 3) if agents["agents"] else None)
        if "no_edit" in agents:  # teams only
            row.update({k: mean(k) for k in ("edits", "crossed", "overwrites", "waits", "wait_s", "unmade_end", "team_s")},
                       **{f"ref_{k}": mean(f"ref_{k}") for k in RULES},
                       no_edit_agents=round(agents["no_edit"] / agents["agents"], 3))
        out[f"{subset} {name}"] = row



def boot(per_task_values, n=10000, seed=0):
    """Mean over tasks and its 95% interval, resampling tasks."""
    xs, rng = list(per_task_values), random.Random(seed)
    means = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return st.mean(xs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


PAIRS = [("L0", "single"), ("L6", "single"), ("L5", "single"), ("L1", "L0"), ("L2", "L0"), ("L5", "L0"), ("L6", "L0"),
         ("L6", "L5"), ("L3", "L0"), ("L4", "L0"), ("L5", "L3"), ("L6", "L4")]
paired = {}
for (subset, name), tasks in by_task.items():
    out[f"{subset} {name}"]["solved_ci"] = [round(v, 3) for v in boot(st.mean(v) for v in tasks.values())[1:]]
for subset in SUBSETS:
    for a, b in PAIRS:
        A, B = by_task.get((subset, a)), by_task.get((subset, b))
        if not A or not B:
            continue
        ids = sorted(set(A) & set(B))
        m, lo, hi = boot(st.mean(A[i]) - st.mean(B[i]) for i in ids)
        paired[f"{subset} {a}-{b}"] = {"tasks": len(ids), "diff": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3)}

if "--tags" not in sys.argv and "--subsets" not in sys.argv:
    (ROOT / "analysis" / ("cb_ladder_ctx.json" if CTX else "cb_ladder.json")).write_text(json.dumps(out, indent=1))
    (ROOT / "analysis" / ("cb_ladder_paired_ctx.json" if CTX else "cb_ladder_paired.json")).write_text(json.dumps(paired, indent=1))
cols = ["runs", "tasks", "errored", "solved", "solved_sd", "features", "edits", "crossed", "overwrites", "no_edit_agents",
        *(f"ref_{k}" for k in RULES), "waits", "wait_s", "unmade_end", "team_s", "calls", "tokens_k",
        *(["drops", "trimmed", "max_pk"] if CTX else [])]
short = {"errored": "err", "solved_sd": "sd", "features": "feats", "overwrites": "overwr", "no_edit_agents": "no_edit",
         "ref_intent": "r_int", "ref_relevant_catchup": "r_rel", "ref_read_before_post": "r_post",
         "ref_read_before_write": "r_write", "ref_no_conflict": "r_confl", "unmade_end": "unmade", "tokens_k": "tok_k"}
print(f"gpt-oss-120b, CooperBench {' and '.join(SUBSETS)} (pairs: 2-agent teams; triples: 3-agent teams); per task unless noted")
print(f"{'':13s}" + "".join(f"{short.get(c, c):>8s}" for c in cols))
for name, row in out.items():
    print(f"{name:13s}" + "".join(f"{'' if row.get(c) is None else row[c]:>8}" for c in cols))
print("\nsolved, 95% interval over tasks:  " + "  ".join(f"{k} {v['solved']:.0%} [{v['solved_ci'][0]:.0%}, {v['solved_ci'][1]:.0%}]"
                                                    for k, v in out.items()))
print("paired (A - B on the same tasks), solved, points with 95% interval:")
for k, v in paired.items():
    print(f"  {k:20s} {v['diff'] * 100:+6.1f}  [{v['lo'] * 100:+6.1f}, {v['hi'] * 100:+6.1f}]  over {v['tasks']} tasks"
          + ("  *" if v["lo"] > 0 or v["hi"] < 0 else ""))
