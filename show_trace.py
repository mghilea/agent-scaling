"""Show one task's trace as a readable timeline: who told whom what, and what each agent did.

    python show_trace.py runs/<run> email-41          # one task
    python show_trace.py runs/<run> email-41 --full   # without shortening long texts
    python show_trace.py runs/<run>                   # list the run's tasks and results

Works for single-agent and multi-agent runs. Needs only the standard library, so it runs anywhere.
"""
import argparse
import json
import sys
from pathlib import Path

# The tools that change the databases (workbench.WRITE_TOOLS, with dots as underscores).
WRITE_TOOLS = {"email_send_email", "email_delete_email", "email_forward_email", "email_reply_email",
               "calendar_create_event", "calendar_delete_event", "calendar_update_event",
               "analytics_create_plot", "project_management_create_task", "project_management_delete_task",
               "project_management_update_task", "customer_relationship_manager_update_customer",
               "customer_relationship_manager_add_customer", "customer_relationship_manager_delete_customer"}
SUMMARIZE_PREFIX = "Based on your exploration so far, write a summary"

_COLOR = sys.stdout.isatty()


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t): return c("1", t)
def dim(t): return c("2", t)
def green(t): return c("32", t)
def yellow(t): return c("33", t)
def cyan(t): return c("36", t)


class Printer:
    def __init__(self, full):
        self.full = full

    def short(self, text, limit):
        text = " ".join((text or "").split()) if not self.full else (text or "").strip()
        return text if self.full or len(text) <= limit else text[:limit] + dim(f" … [{len(text) - limit} more chars]")

    def block(self, indent, label, text, limit=400):
        body = self.short(text, limit)
        pad = " " * indent
        lines = body.splitlines() or [""]
        print(f"{pad}{label} {lines[0]}" + "".join(f"\n{pad}   {line}" for line in lines[1:]))


def rounds_of(conversation):
    """Split a worker's conversation into rounds: [(instructions, [(tool, args, output)], summary)]."""
    outputs = {m["tool_call_id"]: m["content"] for m in conversation if m.get("role") == "tool"}
    rounds, current, asked_summary = [], None, False
    for m in conversation:
        if m["role"] == "user":
            asked_summary = m["content"].startswith(SUMMARIZE_PREFIX)
            if not asked_summary:
                current = {"instructions": m["content"], "calls": [], "summary": None, "answer": None}
                rounds.append(current)
        elif current is None:
            continue
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                f = tc["function"]
                current["calls"].append((f["name"], f["arguments"], outputs.get(tc["id"], "")))
        elif m["role"] == "assistant":
            current["summary" if asked_summary else "answer"] = m["content"]
    return rounds


def show_calls(p, calls, indent):
    if not calls:
        print(" " * indent + dim("(no tool calls)"))
    for name, args, out in calls:
        try:
            a = json.loads(args)
            args = ", ".join(f"{k}={json.dumps(v)}" for k, v in a.items()) if isinstance(a, dict) else args
        except (json.JSONDecodeError, TypeError):
            pass
        tag = yellow(" [write]") if name in WRITE_TOOLS else ""  # the output says whether it worked
        p.block(indent, cyan(f"{name}({p.short(args, 160)})") + tag, dim("→ ") + p.short(out, 140), limit=10_000)


def show_multi(p, t):
    conv = {aid: rounds_of(msgs) for aid, msgs in t["conversations"].items()}
    calls = {c_["tag"]: c_["reply"] for c_ in t.get("orchestrator_calls", [])}
    msgs_by_round = {}
    for m in t["messages"]:
        msgs_by_round.setdefault(m["round"], []).append(m)

    if isinstance(t.get("plan"), list):
        print(bold("\nPLAN") + dim("  (orchestrator splits the task)"))
        for i, s in enumerate(t["plan"], 1):
            p.block(2, f"agent_{i}:", f"{s.get('objective', '')}  " + dim(f"(focus: {s.get('focus', '')})"))

    n_rounds = max(len(r) for r in conv.values())
    for r in range(1, n_rounds + 1):
        print(bold(f"\nROUND {r}"))
        incoming = [m for m in msgs_by_round.get(r, []) if m["from"] == "orchestrator" or m["to"] != "orchestrator"]
        # Messages delivered into agents' contexts at the start of the round (guidance, peer answers).
        seen = set()
        for m in incoming:
            if m["from"] != "orchestrator":  # peer messages: print each text once, listing its recipients
                key = (m["from"], m["text"])
                if key in seen:
                    continue
                seen.add(key)
                to = [x["to"] for x in incoming if (x["from"], x["text"]) == key]
                p.block(2, green(f"{m['from']} → {', '.join(to)}:"), m["text"], 300)
            else:
                p.block(2, green(f"orchestrator → {m['to']}:"), m["text"], 400)
        for aid, rs in conv.items():
            if r > len(rs):
                continue
            rnd = rs[r - 1]
            print(f"  {bold(aid)} works:")
            show_calls(p, rnd["calls"], 6)
            if rnd["summary"] is not None:
                sent_to = sorted({m["to"] for m in msgs_by_round.get(r, []) if m["from"] == aid and m["to"] == "orchestrator"})
                label = f"{aid} → orchestrator:" if sent_to else f"{aid} summary" + (" (sent to peers next round):" if t["topology"] == "decentralized" else ":")
                p.block(6, green(label), rnd["summary"], 300)
        if f"stopping_{r}" in calls:
            p.block(2, bold("orchestrator stop decision:"), calls[f"stopping_{r}"], 250)

    print(bold("\nFINAL ANSWER"))
    if "candidates" in t:
        cands = list(t["candidates"].items())
        print(dim("  changes the workers made (each in its own private workspace):"))
        for i, (a, who) in enumerate(cands, 1):
            print(f"    {i}. {p.short(a.replace('.func(', '(', 1), 220)}" + dim(f"  [{'; '.join(who)}]"))
        print(f"  orchestrator chose to apply: {t.get('selection')}")
    if "vote" in t:
        v = t["vote"]
        for aid, acts in v["answers"].items():
            p.block(2, f"{aid}'s answer:", "; ".join(acts) or "(no changes)", 400)
        print(f"  majority vote: {v['winner']}'s answer, {v['votes']} of {len(v['answers'])} votes")
    if t["topology"] == "independent":
        print("  every distinct change from every worker is kept (no cross-checking)")
    if t.get("worker_errors"):
        print(yellow(f"  worker errors: {t['worker_errors']}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("task", nargs="?")
    ap.add_argument("--full", action="store_true", help="don't shorten long texts")
    args = ap.parse_args()
    run = Path(args.run)
    rows = {r["id"]: r for r in map(json.loads, open(run / "results.jsonl"))}
    if not args.task:
        for i, r in rows.items():
            print(f"{'PASS' if r['correct'] else 'FAIL'}  {i}  turns={r['turns']} messages={r.get('agent_messages', 0)}")
        return
    row = rows[args.task]
    t = json.loads((run / "traces" / f"{args.task}.json").read_text())
    p = Printer(args.full)
    topology = t["topology"] if isinstance(t, dict) else "single"
    verdict = green("PASS") if row["correct"] else yellow("FAIL") + (" (side effects)" if row.get("side_effects") else "")
    print(bold(f"=== {topology} · {args.task} · ") + verdict + bold(" ==="))
    first_user = next(m["content"] for m in (t["conversations"]["agent_1"] if isinstance(t, dict) else t)
                      if m["role"] == "user")
    task = first_user.split("Here is the task for the team to work on:\n", 1)[-1].split("\n\nTo start,")[0]
    print(f"Task: {task}")
    print(dim("Expected changes: ") + ("; ".join(a.replace(".func(", "(", 1) for a in row["expected"]) or "(none)"))
    print(dim(f"turns={row['turns']}  tokens={row['prompt_tokens'] + row['completion_tokens']:,}  "
              f"messages between agents={row.get('agent_messages', 0)}"))
    if isinstance(t, dict):
        show_multi(p, t)
    else:
        print(bold("\nAGENT"))
        show_calls(p, rounds_of(t)[0]["calls"], 2)
    print(bold("\nSCORED CHANGES: ") + ("; ".join(a.replace(".func(", "(", 1) for a in row["answer"]
                                                   if a.split(".func(")[0].replace(".", "_") in WRITE_TOOLS) or "(none)"))


if __name__ == "__main__":
    main()
