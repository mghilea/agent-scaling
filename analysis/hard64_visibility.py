"""What a user could see of each system's work on the 64-task hard suite, without the models' private reasoning.

    python analysis/hard64_visibility.py   # writes analysis/hard64_visibility.json

Every system shows its actions (tool calls) and a final reply. Beyond that, a user sees only text the agents
write in the open: for the single agent, notes written alongside tool calls; for agent-driven teams, their
messages, A2A tasks and shared-state writes; for the paper's setups, the messages the harness passes between
agents (round summaries, orchestrator instructions). Each distinct text is counted once, even when it went
to several agents. Averaged over the ten repeat runs (tags rep01-rep10).
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYSTEMS = {"sas": "Single agent", "independent": "Independent", "centralized": "Centralized",
           "decentralized": "Decentralized", "hybrid": "Hybrid", "agent-driven-none": "Agent-driven, no channels",
           "agent-driven-text-coord": "Agent-driven, text + coordinate",
           "agent-driven-all-coord": "Agent-driven, all channels + coordinate"}
COMM = ("message", "a2a_send", "a2a_update", "state_write")


def visible(trace):
    """(number of visible texts, their total characters) for one task-run's trace."""
    if isinstance(trace, list):  # single agent: its conversation
        texts = [m.get("content") or "" for m in trace if m.get("role") == "assistant" and m.get("tool_calls")]
    elif "events" in trace:  # agent-driven
        texts = [e.get("text") or e.get("instruction") or e.get("value") or "" for e in trace["events"] if e["kind"] in COMM]
    else:  # the paper's setups
        texts = list(dict.fromkeys((m["round"], m["from"], m["text"]) for m in trace.get("messages", [])))
        texts = [t for _, _, t in texts]
    texts = [t for t in texts if t.strip()]
    return len(texts), sum(len(t) for t in texts)


out = {}
per = defaultdict(lambda: {"items": [], "chars": [], "any": [], "actions": []})
for d in sorted((ROOT / "runs").iterdir()):
    m = re.match(r"\d{8}-\d{6}-(.+)-workbench-gpt-oss-20b-rep\d+$", d.name)
    if not m or m.group(1) not in SYSTEMS:
        continue
    label = SYSTEMS[m.group(1)]
    for line in open(d / "results.jsonl"):
        r = json.loads(line)
        n, c = visible(json.load(open(d / "traces" / f"{r['id']}.json")))
        per[label]["items"].append(n)
        per[label]["chars"].append(c)
        per[label]["any"].append(n > 0)
        per[label]["actions"].append(r["tool_calls"])
for label, v in per.items():
    out[label] = {"task_runs": len(v["items"]), "items": st.mean(v["items"]), "chars": st.mean(v["chars"]),
                  "share_with_any": st.mean(v["any"]), "tool_calls": st.mean(v["actions"])}
(HERE / "hard64_visibility.json").write_text(json.dumps(out, indent=1))
for label, v in out.items():
    print(f"{label:42s} {v['task_runs']:4d} task-runs  {v['items']:5.1f} texts  {v['chars']:7.0f} chars  "
          f"{v['share_with_any']:4.0%} with any  {v['tool_calls']:5.1f} tool calls")
