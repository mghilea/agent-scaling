"""Condensed timelines of chosen agent-driven tasks from the 64-task hard suite, for the hard-task page.

    python analysis/hard64_examples.py   # writes analysis/hard64_examples.json

Each timeline keeps every communication act (message, A2A task or update, shared-state write or
refused write), every workspace change, every search and each agent's final reply. Runs of the same
action by one agent are merged into one row. Directory lookups, reads and get-by-id calls are dropped.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"none": "20260930-201443-agent-driven-none-workbench-gpt-oss-20b",
        "text": "20260930-201903-agent-driven-text-coord-workbench-gpt-oss-20b",
        "all": "20260930-201908-agent-driven-all-coord-workbench-gpt-oss-20b"}

EXAMPLES = {
    "none": [
        ("multi_domain-146", "Getting past the cap by accident",
         "agent_2 deletes the first 5 of Lena's 11 leads and declares the job done. agent_3's search runs after those "
         "deletions, so it gets the next 5. agent_1 spends its calls re-deleting IDs that are already gone, then "
         "searches again and finds the last one. All 11 leads deleted, purely from timing."),
        ("customer_relationship_manager-40", "Two readings, no way to compare them",
         "Only 3 of Lena's software customers are leads. Neither agent_2 nor agent_3 filters on status, so agent_2 "
         "reassigns the first 5 of her software customers and agent_3, searching after it, the next 5. Between them "
         "they catch all 3 leads, plus 7 customers who aren't leads. agent_1 guesses an email address, finds "
         "nothing, and reports that there is nothing to do. Nobody can see that the others read the task differently."),
    ],
    "text": [
        ("calendar-86", "A clean hand-off",
         "Two agents propose plans within seconds of each other. They settle on agent_3's: agent_3 searches and sends the IDs, "
         "agent_2 deletes. agent_1 stays out of it. Each change is made exactly once."),
        ("email-15", "Right answer, from two readings of \"the last 3 days\"",
         "agent_3 announces it is going ahead and deletes 3 emails from Nov 27–29 before anyone replies. agent_1 "
         "then claims the job, and agent_2 proposes to do it. agent_1 asks agent_2 to include Nov 30, so agent_2 "
         "deletes 3 more. The union of the two readings is the correct 6."),
        ("multi_domain-147", "A late message triggers an undo",
         "agent_3 deletes 5 of Raj's customers. agent_1's \"please hold off\" message arrives afterwards, so agent_3 "
         "tries to undo its deletions by re-adding 3 customers as new records. agent_2 tells it to stop. Then all "
         "three delete again, each with its own search, including the 3 re-added records. Nobody filters on "
         "\"Lead\": 14 of Raj's customers are deleted, only 2 of them among his 10 leads. 20 changes, and the CRM "
         "ends up wrong."),
    ],
    "all": [
        ("project_management-29", "Shared key, A2A task, version conflicts, and duplicate work anyway",
         "agent_1 writes the 3 overdue task IDs to a shared key and sends agent_2 an A2A task to move them. agent_3 "
         "reads the shared state and moves them itself, and agent_2 moves them again (no effect the second time). "
         "agent_1 sends a second A2A task for the same work and marks the status key completed. agent_2 and agent_3 "
         "then both try to update that key from a stale version and are refused. Right result; agent_2 notices: "
         "\"So there's duplication.\""),
        ("customer_relationship_manager-58", "The protocol starts after the work is done",
         "agent_1 and agent_2 reassign Lena's customers at the same moment without saying anything, splitting them by "
         "accident (agent_2 repeats two of agent_1's updates, which change nothing). 20 seconds later agent_3 delegates by A2A: agent_2 is to store two emails in the "
         "shared state, and agent_1 is to reassign customers who have already been reassigned. agent_1 finds none "
         "left and marks its task completed."),
        ("multi_domain-148", "Structured hand-off of the wrong list",
         "agent_1 sends agent_2 an A2A task with a JSON list of Akira's customer IDs, and agent_2 stores its own list "
         "under a shared key. Nobody filtered on status \"Lead\": the team deletes 9 of Akira's customers, and none "
         "of the 9 is one of his 8 leads. The channels passed the list along faithfully, with the error in it."),
    ],
}

DOMAIN = {"customer_relationship_manager": "CRM", "calendar": "calendar", "email": "email",
          "project_management": "project board", "analytics": "analytics"}
NOUN = {"customer": "customers", "event": "events", "email": "emails", "task": "tasks"}
SKIP_TOOLS = ("company_directory_", "get_", "_information_by_id")


def split_tool(tool):
    for d in DOMAIN:
        if tool.startswith(d + "_"):
            return d, tool[len(d) + 1:]
    return "", tool


def ids_of(args):
    return [str(v) for k, v in args.items() if k.endswith("_id")]


def phrase(tool, args_list, effective):
    """One sentence for a run of calls to the same write tool."""
    d, fn = split_tool(tool)
    verb, _, noun = fn.partition("_")
    ids = [i for a in args_list for i in ids_of(a)]
    n = len(ids)
    dupes = Counter(ids)
    ids = [f"{i} (×{dupes[i]})" if dupes[i] > 1 else i for i in dict.fromkeys(ids)]
    a0 = args_list[0]
    if verb == "delete":
        s = f"deleted {n} {NOUN.get(noun, noun) if n > 1 else noun}: {', '.join(ids)}" if effective else \
            f"tried to delete {', '.join(ids)}: already gone"
    elif verb == "update":
        what = (f"moved {NOUN.get(noun, noun)} to \"{a0.get('new_value')}\"" if a0.get("field") == "list_name"
                else f"set {a0.get('field')} to \"{a0.get('new_value')}\" on {NOUN.get(noun, noun)}")
        s = f"{what}: {', '.join(ids)}" if effective else f"repeated: {what} {', '.join(ids)} (no change)"
    elif verb == "add":
        names = ", ".join(f"{a.get('customer_name')} ({a.get('status')})" for a in args_list)
        s = f"added {len(args_list)} new {NOUN.get(noun, noun)}: {names}"
    else:
        s = f"{fn}({json.dumps(a0)[:120]})" + ("" if effective else " (no change)")
    return s


def search_phrase(tool, args_list):
    d, _ = split_tool(tool)
    shown = []
    for a in args_list:
        f = ", ".join(f"{k}={v}" for k, v in a.items() if v not in (None, "", []))
        if f not in shown:
            shown.append(f)
    times = f" ×{len(args_list)}" if len(args_list) > 1 else ""
    return f"searched the {DOMAIN.get(d, d)}{times}: " + " | ".join(shown or ["no filters"])


def timeline(trace):
    rows, open_burst = [], {}

    def close(agent):
        b = open_burst.pop(agent, None)
        if b:
            kind, tool, effective = b["key"]
            if kind == "write":
                text = phrase(tool, b["args"], effective)
            elif kind == "search":
                text = search_phrase(tool, b["args"])
            else:
                text = "checked the engaged-user counts"
            rows.append({"t": b["t"], "t2": b["t2"], "agent": agent, "kind": "action" if effective else "noop", "text": text})

    for e in trace["events"]:
        a, k = e["agent"], e["kind"]
        if k == "tool":
            tool = e["tool"]
            if any(s in tool for s in SKIP_TOOLS):
                continue
            if "search" in tool:
                key = ("search", tool, True)
            elif tool.startswith("analytics_"):
                key = ("analytics", tool, True)
            else:
                key = ("write", tool, bool(e["changes"]))
            b = open_burst.get(a)
            if b and b["key"] == key:
                b["args"].append(e["args"]); b["t2"] = e["t"]
            else:
                close(a)
                open_burst[a] = {"key": key, "t": e["t"], "t2": e["t"], "args": [e["args"]]}
            continue
        if k in ("active", "state_read", "a2a_get"):
            continue
        close(a)
        to = lambda e: "all" if e.get("broadcast") or len(e.get("to", [])) > 1 else e["to"][0]
        if k == "message":
            rows.append({"t": e["t"], "agent": a, "kind": "message", "to": to(e), "text": e["text"]})
        elif k == "a2a_send":
            rows.append({"t": e["t"], "agent": a, "kind": "a2a", "to": e["to"][0],
                         "text": f"task {e['task_id']} (submitted): {e['instruction']}", "data": e["data"] or None})
        elif k == "a2a_update":
            rows.append({"t": e["t"], "agent": a, "kind": "a2a", "to": e["to"][0],
                         "text": f"task {e['task_id']} → {e['state']}" + (f": {e['text']}" if e["text"] else ""),
                         "data": e["data"] or None})
        elif k == "state_write":
            rows.append({"t": e["t"], "agent": a, "kind": "state", "text": f"{e['key']} (version {e['version']}) = {e['value']}"})
        elif k == "state_conflict":
            rows.append({"t": e["t"], "agent": a, "kind": "conflict",
                         "text": f"write to {e['key']} refused: expected version {e['expected']}, current is {e['actual']}"})
        elif k == "idle" and e["reply"].strip():
            rows.append({"t": e["t"], "agent": a, "kind": "reply", "text": e["reply"]})
    for a in list(open_burst):
        close(a)
    rows.sort(key=lambda r: r["t"])
    for r in rows:
        r["text"] = r["text"] if len(r["text"]) <= 520 else r["text"][:520] + "…"
    return rows


def usage(run_dir):
    """How many of the 64 tasks used each channel."""
    c = Counter()
    for f in sorted((ROOT / "runs" / run_dir / "traces").glob("*.json")):
        kinds = {e["kind"] for e in json.load(open(f))["events"]}
        c["tasks"] += 1
        c["message"] += "message" in kinds
        c["a2a"] += "a2a_send" in kinds
        c["shared"] += "state_write" in kinds
        c["any"] += bool(kinds & {"message", "a2a_send", "state_write"})
    return dict(c)


out = {"usage": {}, "examples": {}}
for team, run_dir in RUNS.items():
    results = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs" / run_dir / "results.jsonl")}
    out["usage"][team] = usage(run_dir)
    out["examples"][team] = []
    for tid, title, caption in EXAMPLES[team]:
        trace = json.load(open(ROOT / "runs" / run_dir / "traces" / f"{tid}.json"))
        out["examples"][team].append({"task_id": tid, "title": title, "caption": caption, "request": trace["task"],
                                      "correct": results[tid]["correct"], "changes_made": len(trace["changes"]),
                                      "rows": timeline(trace)})
json.dump(out, open(ROOT / "analysis" / "hard64_examples.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out["usage"]), {t: [len(x["rows"]) for x in v] for t, v in out["examples"].items()})
