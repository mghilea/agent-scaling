"""Per-agent participation in CooperBench teams (flash: 2 agents, flash3: 3, big4/6/8): agents with no edits, own feature
passing, active share of team time, the busiest agent's share of calls, edits on the agent's own feature's reference files vs
files only teammates' features touch, out of calls. Usage: python analysis/cb_participation.py [--ctx] [L0 L4 L5 L6] (--ctx: the equal-context ladder, runs cx01/cx02). Ids and numbers only."""
import json, glob, statistics as st, re, sys
from collections import defaultdict, Counter
D = "/scratch/mg3117/cooperbench-src/dataset"
INFRA = ("error: CalledProcessError: Command '['apptainer'", "error: RuntimeError: building ")
CTX = "--ctx" in sys.argv
def kind(L): return ("agent-driven-none-rc" if L == "L0" else f"agent-driven-none-{L}-rc") + ("-ctx128k" if CTX else "")
_files = {}
def ref_files(task_id, f):
    m = re.match(r"(.+_task)-(\d+)-", task_id)
    key = (m.group(1), m.group(2), f)
    if key not in _files:
        p = f"{D}/{m.group(1)}/task{m.group(2)}/feature{f}/feature.patch"
        _files[key] = {l.split(" b/", 1)[1].strip() for l in open(p) if l.startswith("diff --git")}
    return _files[key]
def analyze(sub, tags, L):
    a = Counter(); share_max = []; active_frac = []; calls_cv = []
    for d in glob.glob(f"runs/*-{kind(L)}-cooperbench-{sub}-gpt-oss-120b-*"):
        if d.rsplit("-", 1)[1] not in tags: continue
        f = d + "/results.jsonl"
        for l in (open(f) if glob.glob(f) else []):
            r = json.loads(l)
            if str(r["stop"]).startswith(INFRA): continue
            tr = json.load(open(f"{d}/traces/{r['id']}.json")); ev = tr["events"]; feats = tr["features"]
            n = len(feats)
            calls = {ag: sum(1 for m in tr["conversations"][ag] if m.get("role") == "assistant") for ag in feats}
            edits = Counter(e["agent"] for e in ev if e["kind"] == "edit" and e["ok"])
            span = {}
            for e in ev:
                if e["kind"] == "active": span.setdefault(e["agent"], [e["t"], e["t"]])
                elif e["kind"] == "idle" and e["agent"] in span: span[e["agent"]][1] = e["t"]
            team_end = max((b for _, b in span.values()), default=1) or 1
            tot_calls = sum(calls.values()) or 1
            share_max.append(max(calls.values()) / tot_calls)
            if n > 1 and st.mean(calls.values()):
                calls_cv.append(st.pstdev(calls.values()) / st.mean(calls.values()))
            own_passed = {ag: bool(r.get(f"feature{i + 1}_passed")) for i, ag in enumerate(sorted(feats, key=lambda x: int(x.split('_')[1])))}
            for ag, f in feats.items():
                a["agents"] += 1
                if edits[ag] == 0: a["no_edit"] += 1
                if ag in span: active_frac.append((span[ag][1] - span[ag][0]) / team_end)
                mine = ref_files(r["id"], f)
                others = set().union(*(ref_files(r["id"], g) for x, g in feats.items() if x != ag)) - mine
                ed_files = [e["file"] for e in ev if e["kind"] == "edit" and e["ok"] and e["agent"] == ag]
                if ed_files:
                    a["edited_any"] += 1
                    a["edited_own_ref"] += any(x in mine for x in ed_files)
                    a["edited_only_teammates_ref"] += (any(x in others for x in ed_files) and not any(x in mine for x in ed_files))
                    a["edits_total"] += len(ed_files); a["edits_on_own_ref"] += sum(x in mine for x in ed_files)
                    a["edits_on_teammates_ref"] += sum(x in others for x in ed_files)
                a["own_feature_passed"] += own_passed.get(ag, False)
                stop = r["comm"].get("agent_stops", {}).get(ag)
                a["out_of_calls"] += stop == "max_turns"
    if not a["agents"]: return None
    n = a["agents"]; e = max(1, a["edits_total"])
    return {"agents": n, "no edits": f"{a['no_edit']/n:.0%}", "own feature passes": f"{a['own_feature_passed']/n:.0%}",
            "active (of team time)": f"{st.median(active_frac):.0%}", "busiest agent's share of calls": f"{st.mean(share_max):.0%}",
            "edits on own feature's files": f"{a['edits_on_own_ref']/e:.0%}", "edits on teammates' files": f"{a['edits_on_teammates_ref']/e:.0%}",
            "agents editing only teammates' files": a["edited_only_teammates_ref"], "out of calls": f"{a['out_of_calls']/n:.0%}"}
sets = ([("flash", ["cx01", "cx02"], "2"), ("flash3", ["cx01", "cx02"], "3")] if CTX else
        [("flash", ["fl01", "fl02"], "2"), ("flash3", ["fl01", "fl02"], "3")] + [(f"big{k}", ["bg01", "bg02"], f"{k}*") for k in (4, 6, 8)])
which = [a for a in sys.argv[1:] if a != "--ctx"] or ["L0", "L5", "L6"]
for sub, tags, label in sets:
    for L in which:
        res = analyze(sub, tags, L)
        if res: print(f"{label:3s} agents {L}: " + " | ".join(f"{k} {v}" for k, v in res.items()))
