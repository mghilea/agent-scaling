"""Build an animated replay page of one task across several runs (single and multi-agent).

    python trace_viz.py email-41 runs/<single> runs/<independent> ... -o replay.html [--notes notes.json]

Each run becomes a tab: a diagram of the topology where every real message moves along its edge,
step by step, next to the exact text of each message and every tool call. --notes adds a short
"what happened" paragraph per topology ({"centralized": "...", ...}). Standard library only.
"""
import argparse
import json
from pathlib import Path

from show_trace import WRITE_TOOLS, rounds_of

HERE = Path(__file__).resolve().parent
OUT_CHARS = 1500  # tool outputs longer than this are cut in the page


def call_record(name, args, out):
    try:
        a = json.loads(args)
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items()) if isinstance(a, dict) else args
    except (json.JSONDecodeError, TypeError):
        pass
    write = name in WRITE_TOOLS
    return {"name": name, "args": args, "out": out[:OUT_CHARS] + (" …" if len(out) > OUT_CHARS else ""),
            "write": write, "ok": (not write) or "success" in out.lower()}


def show(action):
    return action.replace(".func(", "(", 1)


def single_steps(conv, task):
    calls = [call_record(*c) for c in rounds_of(conv)[0]["calls"]]
    answer = rounds_of(conv)[0]["answer"] or ""
    return [
        {"kind": "task", "round": 0, "title": "The task goes to the agent", "edges": [{"from": "task", "to": "agent"}], "text": task},
        {"kind": "work", "round": 1, "title": "The agent works with its tools", "agents": {"agent": {"calls": calls, "summary": answer}}},
    ]


def multi_steps(t, task):
    topo = t["topology"]
    orch = topo in ("centralized", "hybrid")
    conv = {aid: rounds_of(msgs) for aid, msgs in t["conversations"].items()}
    agents = list(conv)
    stops = {c["tag"]: c["reply"] for c in t.get("orchestrator_calls", []) if c["tag"].startswith("stopping")}
    by_round = {}
    for m in t["messages"]:
        by_round.setdefault(m["round"], []).append(m)

    steps = [{"kind": "task", "round": 0, "text": task,
              "title": "The task goes to the orchestrator" if orch else "The task goes to every agent",
              "edges": [{"from": "task", "to": "orch"}] if orch else [{"from": "task", "to": a} for a in agents]}]
    if isinstance(t.get("plan"), list):
        steps.append({"kind": "plan", "round": 0, "title": "The orchestrator splits the task into subtasks",
                      "subtasks": [{"agent": f"agent_{i}", "objective": s.get("objective", ""), "focus": s.get("focus", "")}
                                   for i, s in enumerate(t["plan"], 1)]})
    n_rounds = max(len(r) for r in conv.values())
    for r in range(1, n_rounds + 1):
        msgs = by_round.get(r, [])
        down = [m for m in msgs if m["from"] == "orchestrator"]
        peer = [m for m in msgs if m["from"] != "orchestrator" and m["to"] != "orchestrator"]
        up = [m for m in msgs if m["to"] == "orchestrator"]
        if down:
            steps.append({"kind": "send", "round": r, "title": f"The orchestrator sends each worker its guidance",
                          "edges": [{"from": "orch", "to": m["to"], "text": m["text"]} for m in down]})
        if peer:
            what = ("Every agent sends its last summary to both peers" if topo == "decentralized"
                    else "Peer round: each worker gets its peers' summaries directly (first 400 characters)")
            steps.append({"kind": "send", "round": r, "title": what,
                          "edges": [{"from": m["from"], "to": m["to"], "text": m["text"]} for m in peer]})
        work = {}
        for a in agents:
            if r <= len(conv[a]):
                rnd = conv[a][r - 1]
                work[a] = {"calls": [call_record(*c) for c in rnd["calls"]],
                           "summary": None if orch else rnd["summary"]}
        last = r == n_rounds
        steps.append({"kind": "work", "round": r, "agents": work,
                      "title": "Each worker acts in its own copy of the databases" + (", then writes a summary" if not orch else ""),
                      "note": ("Nobody receives these summaries: this is the last round." if topo == "decentralized" and last
                               else "The summaries go nowhere: independent agents never talk." if topo == "independent" else None)})
        if up:
            steps.append({"kind": "send", "round": r, "title": "Each worker reports back with a summary of what it did",
                          "edges": [{"from": m["from"], "to": "orch", "text": m["text"]} for m in up]})
        if f"stopping_{r}" in stops:
            reply = stops[f"stopping_{r}"]
            steps.append({"kind": "decide", "round": r, "title": "The orchestrator decides whether to stop",
                          "text": reply, "stop": reply.strip().lstrip("*\"'").upper().startswith("STOP")})
    return steps


def final_step(t, topo, row):
    scored = [show(a) for a in row["answer"] if a.split(".func(")[0].replace(".", "_") in WRITE_TOOLS]
    step = {"kind": "final", "round": 99, "mode": topo, "scored": scored, "expected": [show(a) for a in row["expected"]],
            "correct": bool(row["correct"]), "side_effects": bool(row.get("side_effects"))}
    if topo == "single":
        step.update(title="Its changes are scored by WorkBench", edges=[{"from": "agent", "to": "score"}])
    elif topo == "independent":
        step.update(title="Every distinct change from every agent is kept, then scored",
                    edges=[{"from": a, "to": "score"} for a in t["conversations"]])
    elif topo == "decentralized":
        v = t["vote"]
        step.update(title="Majority vote over the agents' changes, then scored", edges=[{"from": a, "to": "score"} for a in v["answers"]],
                    answers=v["answers"], winner=v["winner"], votes=v["votes"])
    else:
        cands = [{"change": show(a), "who": who} for a, who in t.get("candidates", {}).items()]
        sel = t.get("selection")
        step.update(title="The orchestrator picks which workers' changes to apply, then they are scored",
                    edges=[{"from": "orch", "to": "score"}], candidates=cands,
                    selection=sel if isinstance(sel, list) else [], fallback=sel if isinstance(sel, str) else None)
    return step


def build(task_id, run_dirs, notes):
    systems, task_text = {}, None
    for d in map(Path, run_dirs):
        config = json.loads((d / "config.json").read_text())
        topo = config.get("topology", "single")
        row = next(r for r in map(json.loads, open(d / "results.jsonl")) if r["id"] == task_id)
        t = json.loads((d / "traces" / f"{task_id}.json").read_text())
        if topo == "single":
            task_text = task_text or next(m["content"] for m in t if m["role"] == "user")
            steps = single_steps(t, task_text)
            nodes = ["agent"]
        else:
            first = next(m["content"] for m in t["conversations"]["agent_1"] if m["role"] == "user")
            task_text = task_text or first.split("Here is the task for the team to work on:\n", 1)[-1].split("\n\nTo start,")[0]
            steps = multi_steps(t, task_text)
            nodes = list(t["conversations"])
        steps.append(final_step(t, topo, row))
        systems[topo] = {
            "run": d.name, "nodes": nodes, "steps": steps, "note": notes.get(topo),
            "stats": {"correct": bool(row["correct"]), "side_effects": bool(row.get("side_effects")), "turns": row["turns"],
                      "tokens": row["prompt_tokens"] + row["completion_tokens"], "messages": row.get("agent_messages", 0),
                      "seconds": row["seconds"], "model": config.get("model")},
        }
    return {"task_id": task_id, "task": task_text, "expected": [show(a) for a in row["expected"]],
            "context": notes.get("_context"), "systems": systems}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", default="replay.html")
    ap.add_argument("--notes", help="JSON file with a 'what happened' note per topology (and '_context')")
    args = ap.parse_args()
    notes = json.loads(Path(args.notes).read_text()) if args.notes else {}
    data = build(args.task, args.runs, notes)
    template = (HERE / "trace_viz_template.html").read_text()
    html = template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB, {len(data['systems'])} systems)")


if __name__ == "__main__":
    main()
