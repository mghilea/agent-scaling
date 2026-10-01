"""Hard-task evaluation: accuracy by task size, how agent-driven teams split big tasks, what they said.

    python analysis/hard_eval.py            # writes analysis/hard_eval.json (labels messages with the local vLLM)

Run from ~/agent-scaling with the vLLM venv active and WORKBENCH_DIR set (for published per-task results).
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SYSTEMS = {  # label -> run-name fragment
    "Single agent": "sas", "Decentralized (paper rounds)": "decentralized",
    "Agent-driven, text only + coordinate": "agent-driven-text-coord",
    "Agent-driven, all channels + coordinate": "agent-driven-all-coord",
}
SETS = {  # task set -> run timestamp prefix (each set ran all four systems)
    "small": {"Single agent": "20260923-203411", "Decentralized (paper rounds)": "20260923-205237",
              "Agent-driven, text only + coordinate": "20260930-193802",
              "Agent-driven, all channels + coordinate": "20260930-193800"},
    "3plus": "20260930-2000", "5plus": "20260930-20165",
}
COMM = ("message", "a2a_send", "a2a_update", "state_write")
KEYS = ["email_id", "event_id", "task_id", "customer_id", "value_to_plot", "event_name", "task_name", "customer_name", "recipient"]
ACTS = ["plan", "claim or assignment", "question or request", "status or result report", "information or data",
        "acknowledgment or agreement", "other"]
LABEL_PROMPT = """An AI agent working with two teammates on an office task (email, calendar, CRM) produced this communication act. Classify it.

Channel: {channel}
Content:
{content}

Categories:
- claim or assignment: says who will do what ("I'll forward the emails", "agent_2 handles the calendar", owner/lock keys)
- status or result report: reports what was done or its outcome
- question or request: asks a teammate for something or to confirm something
- information or data: shares facts found, such as IDs, addresses, search results
- plan: lays out several steps or a division of the whole task
- acknowledgment or agreement: thanks, agrees, confirms receipt, no new content
- other

Reply with only JSON: {{"act": "<one category>"}}"""
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


def find_run(prefix, frag):
    hits = [d for d in sorted(glob.glob(str(ROOT / f"runs/{prefix}*"))) if f"-{frag}-workbench" in d]
    return hits[-1] if hits else None


def target(change):
    args = dict(re.findall(r'(\w+)="([^"]*)"', change))
    for k in KEYS:
        if k in args:
            return (change.split("(")[0], k, args[k].lower())
    return (change,)


def label(content):
    try:
        r = client.chat.completions.create(model="gpt-oss-20b", reasoning_effort="low", messages=[
            {"role": "user", "content": LABEL_PROMPT.format(channel=content[0], content=content[1][:1500])}])
        m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
        act = json.loads(m.group(0)).get("act") if m else None
        return act if act in ACTS else "other"
    except Exception:
        return "other"


def act_content(e):
    if e["kind"] == "message":
        return "message", e["text"]
    if e["kind"] == "a2a_send":
        return "A2A task", json.dumps({"instruction": e["instruction"], "data": e.get("data")})
    if e["kind"] == "a2a_update":
        return "A2A update", json.dumps({"state": e["state"], "text": e.get("text"), "data": e.get("data")})
    return "shared state write", f"key={e['key']}\nvalue={e['value']}"


# Published per-task results: share of the 24 models (2026 re-run) that solved each task.
wbdir = Path(os.environ.get("WORKBENCH_DIR", f"/scratch/{os.environ.get('USER')}/WorkBench"))
pub = pd.read_csv(wbdir / "data/results/item_level_results.csv.gz")
pub = pub[pub.run_group == "revisited_2026"]
solve = pub.groupby("task_id").correct.mean().to_dict()
templates = pub.groupby("task_id").base_template.first().to_dict()


def pub_id(tid):
    dom, i = tid.rsplit("-", 1)
    return f"{dom}-{int(i):03d}"


out = {"sets": {}, "tasks": {}}
for set_name, spec in SETS.items():
    S = {}
    for label_, frag in SYSTEMS.items():
        d = find_run(spec[label_] if isinstance(spec, dict) else spec, frag)
        if not d or not os.path.exists(d + "/results.jsonl"):
            continue
        rows = {r["id"]: r for r in map(json.loads, open(d + "/results.jsonl"))}
        S[label_] = (d, rows)
    if not S:
        continue
    common = sorted(set.intersection(*(set(r) for _, r in S.values())))
    res = {}
    for label_, (d, rows) in S.items():
        rs = [rows[i] for i in common]
        n = len(rs)
        entry = {"run": Path(d).name, "n": n,
                 "accuracy": sum(bool(r["correct"]) for r in rs) / n,
                 "side_effects": sum(bool(r.get("side_effects")) for r in rs) / n,
                 "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
                 "seconds": sum(r["seconds"] for r in rs) / n, "turns": sum(r["turns"] for r in rs) / n}
        if "agent-driven" in d:
            acts, kinds, overlap, changers, top_share, before = [], Counter(), 0, [], [], []
            for i in common:
                t = json.load(open(f"{d}/traces/{i}.json"))
                ev = t["events"]
                comm = [e for e in ev if e["kind"] in COMM]
                acts += [act_content(e) for e in comm]
                kinds.update(e["kind"] for e in comm)
                who = defaultdict(set)
                for c in t["changes"]:
                    who[target(c["change"])].add(c["agent"])
                overlap += any(len(a) > 1 for a in who.values())
                by_agent = Counter(c["agent"] for c in t["changes"])
                changers.append(len(by_agent))
                if by_agent:
                    top_share.append(max(by_agent.values()) / sum(by_agent.values()))
                first_change = min((e["t"] for e in ev if e["kind"] == "tool" and e["changes"]), default=None)
                first_comm = min((e["t"] for e in comm), default=None)
                if first_comm is not None:
                    before.append(first_change is None or first_comm < first_change)
            with ThreadPoolExecutor(16) as pool:
                labels = Counter(pool.map(label, acts))
            entry.update({"comm_acts": len(acts) / n, "kinds": {k: kinds[k] / n for k in COMM},
                          "tasks_with_overlap": overlap, "agents_changing": sum(changers) / n,
                          "top_agent_share": sum(top_share) / max(1, len(top_share)),
                          "before_first_change": sum(before) / max(1, len(before)) if before else None,
                          "speech_acts": dict(labels)})
        res[label_] = entry
        for i in common:
            out["tasks"].setdefault(i, {"changes": len(rows[i]["expected"]), "set": set_name,
                                        "published_solve": solve.get(pub_id(i)), "template": templates.get(pub_id(i)), "results": {}})
            out["tasks"][i]["results"][label_] = bool(rows[i]["correct"])
    out["sets"][set_name] = {"n": len(common), "systems": res}
    print(f"\n## {set_name}: {len(common)} tasks")
    for k, v in res.items():
        extra = (f"  comm {v['comm_acts']:.1f}/task, overlap {v['tasks_with_overlap']}, agents changing {v['agents_changing']:.1f}, "
                 f"top-agent share {v['top_agent_share']:.0%}, acts {v['speech_acts']}" if "comm_acts" in v else "")
        print(f"  {k:42} {v['accuracy']:.0%} correct, {v['side_effects']:.0%} side effects, {v['tokens'] / 1000:.0f}k tokens, "
              f"{v['seconds']:.0f}s{extra}")
json.dump(out, open(HERE / "hard_eval.json", "w"), indent=1)
