"""Why agent-driven BrowseComp-Plus teams don't talk: what each agent saw, believed and did, per question.

    python analysis/browsecomp_silence.py --tag split-p1     # needs questions.jsonl on this node

For every agent in every agent-driven run with the tag:
- saw_gold: a page holding the answer (a gold page) came back in its own search results;
- right: its own final reply contains the gold answer; gave_up: it says it couldn't determine one;
- confidence: the confidence it states in its final reply;
- calls: model calls it used (budget 30), and messages it sent.
Per team: whether someone on the team saw a gold page or had the right answer while the team's answer was wrong
("lost inside the team"), and whether anyone told anyone. Prints counts and question ids only.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import browsecomp as bc  # noqa: E402
from browsecomp import parse_verdict  # noqa: E402

GAVE_UP = re.compile(r"unable to|could not|couldn.t|cannot (be )?determine|not (be )?determin|no (individual|match|candidate)|"
                     r"not (been )?(found|identified|located)|none\b|insufficient|unknown", re.I)
CONF = re.compile(r"confidence\W*:?\W*(\d{1,3})", re.I)


def contains(text, gold):
    g, t = bc._norm(gold), bc._norm(text or "")
    return bool(g) and f" {g} " in f" {t} "


def agent_view(msgs, gold_docs, gold):
    seen = set()
    for m in msgs:
        if m.get("role") == "tool" and m.get("name") == "search_documents":
            seen.update(re.findall(r'"docid": "([^"]+)"', m.get("content") or ""))
    replies = [m.get("content") or "" for m in msgs if m.get("role") == "assistant" and not m.get("tool_calls")]
    final = next((r for r in reversed(replies) if r.strip()), "")
    conf = CONF.search(final)
    exact = re.search(r"exact answer\W*:?\**\s*(.+)", final, re.I)
    head = exact.group(1) if exact else final[:300]
    return {"saw_gold": bool(seen & gold_docs), "right": contains(final, gold),
            "gave_up": bool(GAVE_UP.search(head)) and not contains(final, gold),
            "confidence": int(conf.group(1)) if conf and int(conf.group(1)) <= 100 else None,
            "calls": sum(1 for m in msgs if m.get("role") == "assistant"),
            "answered": bool(final.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="split-p1")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in bc.load_tasks()}
    for d in sorted((ROOT / "runs").glob(f"*-agent-driven-*-browsecomp-gpt-oss-20b-{args.tag}")):
        team = re.search(r"agent-driven-(.+?)-browsecomp", d.name).group(1)
        agents, teams = [], []
        for line in open(d / "results.jsonl"):
            r = json.loads(line)
            t = json.load(open(d / "traces" / f"{r['id']}.json"))
            task = tasks[r["id"]]
            gold_docs, gold = set(task["gold_docs"]), task["answer"]
            views = {a: agent_view(m, gold_docs, gold) for a, m in t["conversations"].items()}
            sent = Counter(e["agent"] for e in t["events"] if e["kind"] in ("message", "a2a_send", "state_write"))
            for a, v in views.items():
                v.update({"agent": a, "q": r["id"], "sent": sent.get(a, 0)})
                agents.append(v)
            team_right = parse_verdict(r.get("judgment", "")) if r.get("judgment") else bool(r["correct"])
            teams.append({"q": r["id"], "team_right": team_right,
                          "someone_saw_gold": any(v["saw_gold"] for v in views.values()),
                          "someone_right": any(v["right"] for v in views.values()),
                          "all_right": all(v["right"] for v in views.values()),
                          "talked": sum(sent.values()) > 0,
                          "gold_parts": len({bc.shard_of(x, 3) for x in gold_docs})})
        n = len(agents)
        wrong = [a for a in agents if not a["right"] and not a["gave_up"]]
        confs = [a["confidence"] for a in wrong if a["confidence"] is not None]
        print(f"\n== {team}: {len(teams)} questions, {n} agents")
        print(f"  agents whose own search results contained a gold page: {sum(a['saw_gold'] for a in agents)}")
        print(f"  agents ending with the right answer: {sum(a['right'] for a in agents)}; "
              f"a wrong answer: {len(wrong)}; giving up: {sum(a['gave_up'] for a in agents)}")
        if confs:
            print(f"  confidence stated with a wrong answer: median {st.median(confs)}, "
                  f"{sum(c >= 70 for c in confs)} of {len(confs)} at 70 or more")
        print(f"  model calls used (budget 30): median {st.median(a['calls'] for a in agents)}, "
              f"ran out: {sum(a['calls'] >= 30 for a in agents)}")
        print(f"  agents that sent anything: {sum(a['sent'] > 0 for a in agents)}")
        lost_gold = [x["q"] for x in teams if x["someone_saw_gold"] and not x["team_right"]]
        lost_ans = [x["q"] for x in teams if x["someone_right"] and not x["team_right"]]
        print(f"  team wrong although a member saw a gold page: {len(lost_gold)} {lost_gold}")
        print(f"  team wrong although a member had the right answer: {len(lost_ans)} {lost_ans}")
        print(f"  of those, questions where nobody said anything: "
              f"{sum(1 for x in teams if x['someone_saw_gold'] and not x['team_right'] and not x['talked'])}")


if __name__ == "__main__":
    main()
