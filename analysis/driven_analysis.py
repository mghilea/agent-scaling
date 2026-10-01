"""Analyze agent-driven runs: outcomes, channel choice, timing, shared-state use, and message content."""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

# Agent-driven run directories to analyze: pass them as arguments, or default to the 30 Sep runs.
RUNS = [a for a in sys.argv[1:] if not a.endswith(".json")] or sorted(
    glob.glob(str(Path.home() / "agent-scaling/runs/20260930-19375*-agent-driven-*")) +
    glob.glob(str(Path.home() / "agent-scaling/runs/20260930-1938*-agent-driven-*")))
COMM_KINDS = ("message", "a2a_send", "a2a_update", "state_write")
ACTS = ["claim or assignment", "status or result report", "question or request", "information or data",
        "plan", "acknowledgment or agreement", "other"]
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

LABEL_PROMPT = """An AI agent working with two teammates on an office task (email, calendar, CRM) produced this communication act. Classify it.

Channel: {channel}
Content:
{content}

Categories:
- claim or assignment: says who will do what ("I'll forward the emails", "agent_2 handles the calendar", owner/lock keys)
- status or result report: reports what was done or its outcome ("forwarded 2 emails", "completed")
- question or request: asks a teammate for something or to confirm something
- information or data: shares facts found, such as IDs, addresses, search results
- plan: lays out several steps or a division of the whole task
- acknowledgment or agreement: thanks, agrees, confirms receipt, no new content
- other

Reply with only JSON: {{"act": "<one category>", "structured": true or false}} where structured means the content is JSON or key-value data rather than prose."""


def label(item):
    channel, content = item
    try:
        r = client.chat.completions.create(model="gpt-oss-20b", reasoning_effort="low", messages=[
            {"role": "user", "content": LABEL_PROMPT.format(channel=channel, content=content[:1500])}])
        m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
        d = json.loads(m.group(0)) if m else {}
        act = d.get("act") if d.get("act") in ACTS else "other"
        return act, bool(d.get("structured"))
    except Exception:
        return "other", False


def act_content(e):
    if e["kind"] == "message":
        return "message", e["text"]
    if e["kind"] == "a2a_send":
        return "A2A task", json.dumps({"instruction": e["instruction"], "data": e.get("data")})
    if e["kind"] == "a2a_update":
        return "A2A update", json.dumps({"state": e["state"], "text": e.get("text"), "data": e.get("data")})
    return "shared state write", f"key={e['key']}\nvalue={e['value']}"


summary = {}
for run in RUNS:
    cfg = json.load(open(run + "/config.json"))
    name = f"{cfg['comm']}{'+coord' if cfg.get('coordinate') else ''}"
    rows = {r["id"]: r for r in map(json.loads, open(run + "/results.jsonl"))}
    S = defaultdict(list)
    acts, keys, a2a_states, a2a_fields, examples = [], Counter(), Counter(), Counter(), []
    for tid, r in rows.items():
        t = json.load(open(f"{run}/traces/{tid}.json"))
        ev = t["events"]
        comm = [e for e in ev if e["kind"] in COMM_KINDS]
        first_change = min((e["t"] for e in ev if e["kind"] == "tool" and e["changes"]), default=None)
        first_comm = min((e["t"] for e in comm), default=None)
        S["correct"].append(bool(r["correct"])); S["side"].append(bool(r.get("side_effects")))
        S["tokens"].append(r["prompt_tokens"] + r["completion_tokens"]); S["turns"].append(r["turns"])
        S["seconds"].append(r["seconds"]); S["dups"].append(r.get("duplicate_changes", 0))
        S["any_dup"].append(r.get("duplicate_changes", 0) > 0)
        S["any_comm"].append(bool(comm)); S["n_comm"].append(len(comm))
        S["chars"].append(sum(len(act_content(e)[1]) for e in comm))
        if comm and first_change is not None:
            S["comm_before_change"].append(first_comm < first_change)
        elif comm:
            S["comm_before_change"].append(True)
        for k in COMM_KINDS:
            S["n_" + k].append(sum(1 for e in comm if e["kind"] == k))
        S["reads"].append(sum(1 for e in ev if e["kind"] in ("state_read", "a2a_get")))
        S["conflicts"].append(sum(1 for e in ev if e["kind"] == "state_conflict"))
        S["wakes"].append(r.get("rounds", 0))
        changers = {c["agent"] for c in t["changes"]}
        S["agents_changing"].append(len(changers))
        for e in comm:
            acts.append((tid, e["kind"], act_content(e)))
        for e in ev:
            if e["kind"] == "state_write" and e["created"]:
                keys[e["key"]] += 1
            if e["kind"] == "a2a_update":
                a2a_states[e["state"]] += 1
            if e["kind"] in ("a2a_send", "a2a_update") and e.get("data"):
                a2a_fields.update(e["data"].keys())
    with ThreadPoolExecutor(16) as pool:
        labels = list(pool.map(label, [c for _, _, c in acts]))
    by_kind = defaultdict(Counter)
    structured = Counter()
    for (tid, kind, _), (act, st) in zip(acts, labels):
        by_kind[kind][act] += 1
        structured[kind] += st
    n = len(rows)
    mean = lambda k: sum(S[k]) / max(1, len(S[k]))  # noqa: E731
    summary[name] = {
        "run": Path(run).name, "tasks": n, "accuracy": mean("correct"), "side_effects": mean("side"),
        "tokens": mean("tokens"), "turns": mean("turns"), "seconds": mean("seconds"),
        "dup_changes": mean("dups"), "tasks_with_dup": mean("any_dup"), "agents_changing": mean("agents_changing"),
        "tasks_with_comm": mean("any_comm"), "comm_acts": mean("n_comm"), "chars": mean("chars"),
        "comm_before_first_change": mean("comm_before_change") if S["comm_before_change"] else None,
        "per_kind": {k: mean("n_" + k) for k in COMM_KINDS}, "reads": mean("reads"), "conflicts": sum(S["conflicts"]),
        "acts": {k: dict(v) for k, v in by_kind.items()}, "structured": dict(structured),
        "keys": keys.most_common(30), "a2a_states": dict(a2a_states), "a2a_fields": a2a_fields.most_common(15),
    }
    s = summary[name]
    print(f"\n######## {name}  ({s['run']})")
    print(f"accuracy {s['accuracy']:.0%}  side effects {s['side_effects']:.0%}  tokens/task {s['tokens']:,.0f}  turns {s['turns']:.1f}  {s['seconds']:.0f}s")
    print(f"duplicate changes/task {s['dup_changes']:.2f}; tasks with a duplicate {s['tasks_with_dup']:.0%}; agents making changes/task {s['agents_changing']:.1f}")
    print(f"tasks with any communication {s['tasks_with_comm']:.0%}; comm acts/task {s['comm_acts']:.1f}; chars/task {s['chars']:,.0f}; "
          f"first communication before first change: {s['comm_before_first_change']}")
    print("per kind:", {k: round(v, 2) for k, v in s["per_kind"].items()}, "reads/task", round(s["reads"], 2), "conflicts", s["conflicts"])
    print("speech acts:", s["acts"]); print("structured content:", s["structured"])
    if keys: print("shared-state keys created:", s["keys"])
    if a2a_states: print("A2A states:", s["a2a_states"], "data fields:", s["a2a_fields"])
out = next((a for a in sys.argv[1:] if a.endswith(".json")), "driven_summary.json")
json.dump(summary, open(out, "w"), indent=1)
