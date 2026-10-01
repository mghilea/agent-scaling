"""Analyze the 64-task hard suite (all WorkBench tasks needing 3+ changes) across eight systems.

    python analysis/hard64.py      # writes analysis/hard64.json; labels messages with the local vLLM

Run from ~/agent-scaling with the vLLM venv active and WORKBENCH_DIR set.
"""
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import workbench as wb  # noqa: E402
from hard_common import classify, effective_writes, main_tool, target  # noqa: E402

RUNS = {  # label -> run directory (suites/hard-a.txt and hard-b.txt, 30 Sep 2026)
    "Single agent": "20260930-201438-sas-workbench-gpt-oss-20b",
    "Independent": "20260930-201858-independent-workbench-gpt-oss-20b",
    "Centralized": "20260930-201433-centralized-workbench-gpt-oss-20b",
    "Decentralized": "20260930-201428-decentralized-workbench-gpt-oss-20b",
    "Hybrid": "20260930-201853-hybrid-workbench-gpt-oss-20b",
    "Agent-driven, no channels": "20260930-201443-agent-driven-none-workbench-gpt-oss-20b",
    "Agent-driven, text + coordinate": "20260930-201903-agent-driven-text-coord-workbench-gpt-oss-20b",
    "Agent-driven, all channels + coordinate": "20260930-201908-agent-driven-all-coord-workbench-gpt-oss-20b",
}
COMM = ("message", "a2a_send", "a2a_update", "state_write")
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


def label(item):
    channel, content = item
    try:
        r = client.chat.completions.create(model="gpt-oss-20b", reasoning_effort="low", messages=[
            {"role": "user", "content": LABEL_PROMPT.format(channel=channel, content=content[:1500])}])
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
    return "shared-state write", f"key={e['key']}\nvalue={e['value']}"


wbdir = Path(os.environ.get("WORKBENCH_DIR", f"/scratch/{os.environ.get('USER')}/WorkBench"))
pub = pd.read_csv(wbdir / "data/results/item_level_results.csv.gz")
pub = pub[pub.run_group == "revisited_2026"]


def pub_id(tid):
    dom, i = tid.rsplit("-", 1)
    return f"{dom}-{int(i):03d}"


data = {"systems": {}, "tasks": {}, "published": {}, "examples": {}}
rows_by = {}
for label_, run in RUNS.items():
    d = ROOT / "runs" / run
    rows_by[label_] = (d, {r["id"]: r for r in map(json.loads, open(d / "results.jsonl"))})
ids = sorted(set.intersection(*(set(r) for _, r in rows_by.values())))
size = {i: len(rows_by["Single agent"][1][i]["expected"]) for i in ids}

for label_, (d, rows) in rows_by.items():
    driven = "agent-driven" in d.name
    S = {"n": len(ids), "fail_kinds": Counter(), "cap": [0, 0]}
    acc = {"all": [], "3-4": [], "5+": []}
    side, tok, sec, turns, msgs = [], [], [], [], []
    comm_items, kinds, overlap, changers, top_share, before = [], Counter(), 0, [], [], []
    state_reads = state_writes = conflicts = v0 = 0
    workers = {"per": [], "any": 0, "all": 0, "pairs_both": 0, "pairs_one": 0}
    for i in ids:
        r = rows[i]
        gt = r["expected"]
        bucket = "5+" if size[i] >= 5 else "3-4"
        acc["all"].append(bool(r["correct"]))
        acc[bucket].append(bool(r["correct"]))
        side.append(bool(r.get("side_effects")))
        tok.append(r["prompt_tokens"] + r["completion_tokens"])
        sec.append(r["seconds"])
        turns.append(r["turns"])
        msgs.append(r.get("agent_messages", 0))
        trace = json.load(open(d / "traces" / f"{i}.json"))
        if driven:
            made = [c["change"].replace("(", ".func(", 1) for c in trace["changes"]]
        elif label_ == "Single agent":
            made = effective_writes(trace)
        else:
            made = [a for a in r["answer"] if wb.is_write(a)]
        S["fail_kinds"][classify(made, gt, r["correct"])] += 1
        tool = main_tool(gt)
        if sum(a.startswith(tool) for a in gt) > 5:
            S["cap"][1] += 1
            S["cap"][0] += sum(a.startswith(tool) for a in made) > 5
        if label_ in ("Independent", "Decentralized"):
            ok = [wb.score(effective_writes(c), gt)["correct"] for c in trace["conversations"].values()]
            workers["per"] += ok
            workers["any"] += any(ok)
            workers["all"] += all(ok)
            for a in range(len(ok)):
                for b in range(a + 1, len(ok)):
                    if not ok[a] and not ok[b]:
                        workers["pairs_both"] += 1
                    elif not ok[a] or not ok[b]:
                        workers["pairs_one"] += 1
        if driven:
            ev = trace["events"]
            comm = [e for e in ev if e["kind"] in COMM]
            comm_items += [(i, e) for e in comm]
            kinds.update(e["kind"] for e in comm)
            who = defaultdict(set)
            for c in trace["changes"]:
                who[target(c["change"])].add(c["agent"])
            overlap += any(len(a) > 1 for a in who.values())
            by_agent = Counter(c["agent"] for c in trace["changes"])
            changers.append(len(by_agent))
            if by_agent:
                top_share.append(max(by_agent.values()) / sum(by_agent.values()))
            first_change = min((e["t"] for e in ev if e["kind"] == "tool" and e["changes"]), default=None)
            first_comm = min((e["t"] for e in comm), default=None)
            if first_comm is not None:
                before.append(first_change is None or first_comm < first_change)
            state_reads += sum(e["kind"] == "state_read" for e in ev)
            state_writes += sum(e["kind"] == "state_write" for e in ev)
            cf = [e for e in ev if e["kind"] == "state_conflict"]
            conflicts += len(cf)
            v0 += sum(str(e["expected"]) == "0" for e in cf)
        data["tasks"].setdefault(i, {"changes": size[i], "results": {}})["results"][label_] = bool(r["correct"])
    n = len(ids)
    S.update({k: sum(v) / max(1, len(v)) for k, v in (("accuracy", acc["all"]), ("acc_34", acc["3-4"]), ("acc_5", acc["5+"]))})
    S.update({"n_34": len(acc["3-4"]), "n_5": len(acc["5+"]), "side_effects": sum(side) / n, "tokens": sum(tok) / n,
              "seconds": sum(sec) / n, "turns": sum(turns) / n, "messages": sum(msgs) / n, "fail_kinds": dict(S["fail_kinds"])})
    if workers["per"]:
        p = 1 - sum(workers["per"]) / len(workers["per"])
        pairs = workers["pairs_both"] + workers["pairs_one"]
        S["workers"] = {"worker_right": 1 - p, "any_right": workers["any"] / n, "all_right": workers["all"] / n,
                        "both_wrong": workers["pairs_both"] / max(1, pairs), "both_wrong_if_independent": p / (2 - p)}
    if driven:
        with ThreadPoolExecutor(16) as pool:
            labels = list(pool.map(label, [act_content(e) for _, e in comm_items]))
        S.update({"comm_acts": len(comm_items) / n, "kinds": {k: kinds[k] / n for k in COMM},
                  "tasks_with_overlap": overlap, "agents_changing": sum(changers) / n,
                  "top_agent_share": sum(top_share) / max(1, len(top_share)),
                  "before_first_change": sum(before) / max(1, len(before)) if before else None,
                  "tasks_with_comm": sum(1 for i in ids if any(t == i for t, _ in comm_items)) / n,
                  "speech_acts": dict(Counter(labels)), "state": {"writes": state_writes, "reads": state_reads,
                                                                 "conflicts": conflicts, "create_if_absent_refusals": v0}})
        # Example messages: data hand-offs (3+ record IDs) and catches of leftover records.
        ex = []
        for (tid, e), lab in zip(comm_items, labels):
            text = e.get("text") or e.get("value") or e.get("instruction") or ""
            if e["kind"] in ("message", "state_write") and (len(re.findall(r"\b\d{8}\b", text)) >= 3 or
                                                            re.search(r"remaining|still|left over|leftover|pending", text, re.I)):
                ex.append({"task": tid, "t": e["t"], "agent": e["agent"], "kind": e["kind"], "to": e.get("to"),
                           "key": e.get("key"), "text": text[:420], "act": lab, "correct": bool(rows[tid]["correct"])})
        data["examples"][label_] = ex[:40]
    data["systems"][label_] = S
    print(f"{label_:42} {S['accuracy']:.0%} (3-4 {S['acc_34']:.0%}, 5+ {S['acc_5']:.0%}) side {S['side_effects']:.0%} "
          f"tok {S['tokens'] / 1000:.0f}k cap {S['cap'][0]}/{S['cap'][1]} fails {S['fail_kinds']}"
          + (f" workers {S['workers']}" if "workers" in S else "")
          + (f" comm {S['comm_acts']:.1f} overlap {S['tasks_with_overlap']} agents {S['agents_changing']:.1f} acts {S['speech_acts']} state {S['state']}" if driven else ""))

# Published runs (24 models, 2026 re-run) on these same 64 tasks.
P = pub[pub.task_id.isin([pub_id(i) for i in ids])].copy()
P["bucket"] = P.num_ground_truth_actions.apply(lambda k: "5+" if k >= 5 else "3-4")
fk = Counter()
for _, r in P.iterrows():
    gt = ast.literal_eval(r.ground_truth_actions) if isinstance(r.ground_truth_actions, str) else []
    pred = ast.literal_eval(r.predicted_actions) if isinstance(r.predicted_actions, str) else []
    fk[classify([a for a in pred if wb.is_write(a)], gt, bool(r.correct))] += 1
data["published"] = {"runs": len(P), "models": P.model.nunique(), "accuracy": P.correct.mean(),
                     "by_bucket": P.groupby("bucket").correct.mean().to_dict(),
                     "by_model": P.groupby("model").correct.mean().sort_values().to_dict(),
                     "fail_kinds": dict(fk)}
solve = P.groupby("task_id").correct.mean().to_dict()
question = P.groupby("task_id").task.first().to_dict()
for i in ids:
    data["tasks"][i].update({"solve": solve.get(pub_id(i)), "question": question.get(pub_id(i), "")})
print("\npublished:", {k: v for k, v in data["published"].items() if k != "by_model"})
json.dump(data, open(HERE / "hard64.json", "w"), indent=1, default=float)
