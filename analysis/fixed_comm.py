"""What agents exchange in the paper's fixed-round topologies (saved WorkBench traces)."""
import json, re, glob
from collections import defaultdict
RUNS = ["20260923-2034", "20260923-2052"]
ENT = re.compile(r"\b\d{8}\b|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
WORD = re.compile(r"[a-z0-9@._-]{3,}")
SUMM = "Based on your exploration so far"
def words(s): return set(WORD.findall((s or "").lower()))
def jacc(a, b): return len(a & b) / max(1, len(a | b))
out = {}
for topo in ["centralized", "decentralized", "hybrid"]:
    S = defaultdict(list)
    for prefix in RUNS:
        run = glob.glob(f"runs/{prefix}*-{topo}-workbench-*")[0]
        for f in sorted(glob.glob(run + "/traces/*.json")):
            t = json.load(open(f))
            msgs = t["messages"]
            S["tasks"].append(1); S["msgs"].append(len(msgs))
            for m in msgs:
                kind = "guidance" if m["from"] == "orchestrator" else ("report" if m["to"] == "orchestrator" else "peer")
                m["text"] = m["text"] or ""; S[f"len_{kind}"].append(len(m["text"]))
                S["table"].append("|---" in m["text"] or "|--" in m["text"])
                S["json"].append(m["text"].strip().startswith("{"))
            # per-worker rounds: tool calls, writes; summary redundancy
            for aid, conv in t["conversations"].items():
                users = [i for i, m in enumerate(conv) if m["role"] == "user" and not m["content"].startswith(SUMM)]
                summaries = [conv[i + 1]["content"] for i, m in enumerate(conv[:-1]) if m["role"] == "user" and m["content"].startswith(SUMM) and conv[i + 1]["role"] == "assistant"]
                for r, start in enumerate(users):
                    end = users[r + 1] if r + 1 < len(users) else len(conv)
                    calls = [tc for m in conv[start:end] for tc in (m.get("tool_calls") or [])]
                    if r >= 1:
                        S["later_round_no_tools"].append(len(calls) == 0)
                for a, b in zip(summaries, summaries[1:]):
                    S["self_overlap"].append(jacc(words(a), words(b)))
            # information uptake: entities a worker received in messages that it later used in a tool call,
            # and that it had not already seen in its own earlier tool outputs or arguments
            for aid, conv in t["conversations"].items():
                seen, received = set(), set()
                used_from_msgs = 0; received_new = 0
                for m in conv:
                    if m["role"] == "user":
                        ents = set(ENT.findall(m["content"] or "")) - seen
                        received |= ents; received_new += len(ents); seen |= ents
                    elif m["role"] == "tool":
                        seen |= set(ENT.findall(m["content"] or ""))
                    elif m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            e = set(ENT.findall(tc["function"]["arguments"]))
                            used_from_msgs += len(e & received); received -= e
                            seen |= e
                S["uptake_used"].append(used_from_msgs); S["uptake_recv"].append(received_new)
    n = len(S["tasks"])
    mean = lambda k: sum(S[k]) / max(1, len(S[k]))
    print(f"\n## {topo}: {n} task-runs, {mean('msgs'):.1f} messages/task")
    for k in ("guidance", "report", "peer"):
        if S[f"len_{k}"]: print(f"   {k:9} messages: {len(S[f'len_{k}'])}, mean {mean(f'len_{k}'):,.0f} chars (~{mean(f'len_{k}')/4:,.0f} tokens)")
    print(f"   messages with markdown tables {100*mean('table'):.0f}%, starting as JSON {100*mean('json'):.0f}%")
    print(f"   worker-rounds after round 1 with no tool call: {100*mean('later_round_no_tools'):.0f}% of {len(S['later_round_no_tools'])}")
    print(f"   word overlap between a worker's consecutive summaries: {mean('self_overlap'):.2f}")
    print(f"   IDs/addresses received in messages (new to the worker): {sum(S['uptake_recv'])}; later used in its own tool calls: {sum(S['uptake_used'])}")
