"""Re-grade agent-driven BrowseComp-Plus runs made before the fix for truncated team answers.

    python analysis/rescore_driven_browsecomp.py runs/<run dir> [...]   # needs the judge model on localhost:8000

Before commit "Build team answers from full final replies", agent_driven.py built the team's answer from each
agent's final reply as logged, which is cut at 800 characters; the paper's answer format puts "Exact Answer"
after the explanation, so the answer was often cut off before the vote and the judge saw it. Each run keeps
the answer rule it ran under (the last reply for runs before the vote existed; the vote, else the submitted
answer, after), rebuilt from the full replies in the trace's conversations. Only answers that change are
re-judged. The original results.jsonl is kept as results.truncated.jsonl.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import browsecomp as bc  # noqa: E402
from openai import OpenAI  # noqa: E402

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
tasks = {t["id"]: t for t in bc.load_tasks()}


def full_finals(trace):
    out = {}
    for agent, msgs in trace["conversations"].items():
        final = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "assistant"
                      and not m.get("tool_calls") and (m.get("content") or "").strip()), "")
        if final:
            out[agent] = final
    return out


def team_answer(trace, finals):
    subs = trace.get("team_answers") or []
    if subs:
        a = subs[-1]
        return f"Exact Answer: {a['answer']}\nConfidence: {a['confidence'] if a['confidence'] is not None else 100}%", None
    ids = sorted(trace["conversations"])
    if "answer_rule" in trace:  # the vote rule
        answer, votes, keys = bc.majority_answer([(a, finals[a]) for a in ids if a in finals])
        return answer, {"votes": votes, "answers": keys}
    last_agent = next((e["agent"] for e in reversed(trace["events"]) if e["kind"] == "idle" and e["reply"].strip()), None)
    return finals.get(last_agent, ""), None


for run in map(Path, sys.argv[1:]):
    src = run / "results.jsonl"
    backup = run / "results.truncated.jsonl"
    if backup.exists():
        print(f"{run.name}: already rescored, skipping")
        continue
    rows, changed, flipped = [], 0, 0
    for line in open(src):
        r = json.loads(line)
        trace = json.load(open(run / "traces" / f"{r['id']}.json"))
        answer, vote = team_answer(trace, full_finals(trace))
        if answer.strip() and answer != r["answer"]:
            changed += 1
            task = tasks[r["id"]]
            correct, judgment = bc.judge(client, "gpt-oss-20b", task["question"], answer, task["answer"])
            flipped += correct != bool(r["correct"])
            r.update({"answer": answer, "correct": correct, "judgment": judgment,
                      "answer_in_response": bc.answer_in_response(answer, task["answer"]), "rescored": True})
            if vote is not None:
                r["vote"] = vote
        rows.append(r)
    shutil.copy(src, backup)
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{run.name}: {changed} team answers rebuilt from full replies, {flipped} verdicts changed, "
          f"now {sum(bool(r['correct']) for r in rows)}/{len(rows)} correct")
