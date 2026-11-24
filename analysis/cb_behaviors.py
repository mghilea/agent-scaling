"""Behaviors of the CooperBench ladder teams (flash: 2 agents, flash3: 3 agents; runs fl01/fl02), per level.

    python analysis/cb_behaviors.py

- waiting for files (L5, L6): wal_wait calls and seconds per task, and as a share of all agents' time; blocked episodes (an
  agent's first conflict refusal on a file): did it edit that file later (got_file) or never (gave_up), and how long it took
- refusals per task; failed edits, and those on a file a teammate had changed since (stale_fail); whole-file rewrites
  over a teammate's edit (overwr); bash writes discarded; refused-change reminders and refused changes still unmade at the
  end; agents that ran out of turns
Task runs that errored before the agents started are left out. Ids and numbers only.
--ctx: the equal-context ladder instead (--context-total 131072, runs cx01/cx02).
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, glob, re, statistics as st, sys
CTX = "--ctx" in sys.argv
from collections import Counter, defaultdict
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
def kind(L): return ("agent-driven-none-rc" if L == "L0" else f"agent-driven-none-{L}-rc") + ("-ctx128k" if CTX else "")
def load(sub, L):
    rows = {}
    for d in sorted(glob.glob(f"runs/*-{kind(L)}-cooperbench-{sub}-gpt-oss-120b-{'cx' if CTX else 'fl'}0[12]")):
        f = d + "/results.jsonl"
        for l in (open(f) if glob.glob(f) else []):
            r = json.loads(l)
            if str(r["stop"]).startswith(("error: CalledProcessError: Command '['apptainer'", "error: RuntimeError: building ")): continue
            rows[(d.rsplit("-", 1)[1], r["id"])] = (d, r)
    return rows.values()
pct = lambda xs, q: sorted(xs)[int(q * (len(xs) - 1))] if xs else 0
for sub in ("flash", "flash3"):
    print(f"\n===== {sub} ({'2' if sub == 'flash' else '3'}-agent teams) =====")
    hdr = ["tasks", "solved", "team_s", "wait_calls", "wait_s/task", "wait%time", "blocked", "got_file", "gave_up", "acq_med_s", "acq_max_s",
           "refusals", "failed_ed", "stale_fail", "overwr", "bash_disc", "unmade_rem", "unmade_end", "maxturns"]
    print(f"{'':4s}" + "".join(f"{h:>11s}" for h in hdr))
    for L in LEVELS:
        rows = list(load(sub, L))
        if not rows: continue
        agg = Counter(); team_s = []; acq = []; waits = []
        for d, r in rows:
            tr = json.load(open(f"{d}/traces/{r['id']}.json")); ev = tr["events"]; c = r["comm"]
            team_s.append(c.get("team_seconds", 0))
            w = [e for e in ev if e["kind"] == "wal_wait"]
            agg["wait_calls"] += len(w); ws = sum(e["waited"] for e in w); waits.append(ws)
            agg["wait_share"] += ws / max(1.0, c.get("team_seconds", 1) * len(tr["features"]))
            # blocked episodes: first conflict refusal per (agent, file); did the agent edit that file later, and when?
            seen = set()
            for i, e in enumerate(ev):
                if e["kind"] == "wal_refused" and e["rule"] == "no_conflict":
                    m = re.search(r"repo\.edit (\S+)", e.get("reason", ""))
                    if not m: continue
                    key = (e["agent"], m.group(1))
                    if key in seen: continue
                    seen.add(key); agg["blocked"] += 1
                    later = [x["t"] for x in ev[i:] if x["kind"] == "edit" and x["ok"] and x["agent"] == e["agent"] and x["file"] == m.group(1)]
                    if later: agg["got_file"] += 1; acq.append(later[0] - e["t"])
                    else: agg["gave_up"] += 1
            # failed edits, and those on a file a teammate had changed since this agent's own last edit of it
            last = {}
            for e in ev:
                if e["kind"] != "edit": continue
                if e["ok"]:
                    last[e["file"]] = e["agent"]
                else:
                    agg["failed_ed"] += 1
                    agg["stale_fail"] += last.get(e["file"]) not in (None, e["agent"])
            agg["refusals"] += sum(v for k, v in c.items() if k.startswith("wal_refused_"))
            agg["overwr"] += c.get("overwrites", 0); agg["bash_disc"] += c.get("bash_writes_discarded", 0)
            agg["unmade_rem"] += c.get("unmade_reminded", 0); agg["unmade_end"] += c.get("unmade_at_end", 0)
            agg["maxturns"] += sum(s == "max_turns" for s in c.get("agent_stops", {}).values())
            agg["solved"] += bool(r["correct"])
        n = len(rows)
        vals = [n, f"{agg['solved']/n:.0%}", f"{st.median(team_s):.0f}", f"{agg['wait_calls']/n:.1f}", f"{st.mean(waits):.0f}",
                f"{agg['wait_share']/n:.1%}", agg["blocked"], agg["got_file"], agg["gave_up"], f"{st.median(acq):.0f}" if acq else "-",
                f"{max(acq):.0f}" if acq else "-", f"{agg['refusals']/n:.1f}", f"{agg['failed_ed']/n:.1f}", f"{agg['stale_fail']/n:.1f}",
                f"{agg['overwr']/n:.2f}", f"{agg['bash_disc']/n:.1f}", agg["unmade_rem"], agg["unmade_end"], agg["maxturns"]]
        print(f"{L:4s}" + "".join(f"{str(v):>11s}" for v in vals))
