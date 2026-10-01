"""Build the BrowseComp-Plus page from the runs in runs/ (via browsecomp_runs.py).

    python analysis/build_browsecomp_page.py   # writes analysis/browsecomp-final.html from analysis/browsecomp.html

The page gets question ids and numbers only: BrowseComp's questions and answers must not leave /scratch.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from browsecomp_runs import lenient, load, norm, summary  # noqa: E402

# Which run group stands for each system on the page (tag t10: the first 10 of the paper's 100 questions).
SYSTEMS = [
    ("Single agent", "sas", "t10"),
    ("Independent", "independent", "t10"),
    ("Centralized", "centralized", "t10"),
    ("Decentralized", "decentralized", "t10"),
    ("Hybrid", "hybrid", "t10"),
    ("Agent-driven, no channels", "agent-driven-none", "t10b"),
    ("Agent-driven, text + coordinate", "agent-driven-text-coord", "t10b"),
]
groups = load()
first10 = sorted(groups[("sas", "t10")], key=lambda i: int(i.split("-")[1]))
paper_order = [r["id"] for r in map(json.loads, open(next((HERE.parent / "runs").glob("*-sas-browsecomp-gpt-oss-20b-t10")) / "results.jsonl"))]

data = {"systems": [], "grid": {"questions": paper_order, "columns": []}}
for label, kind, tag in SYSTEMS:
    rows = groups[(kind, tag)]
    s = summary(rows)
    s.update({"label": label, "solved_ids": sorted(i for i, r in rows.items() if r["correct"]),
              "lenient_ids": sorted(i for i, r in rows.items() if lenient(r) and not r["correct"]),
              "judge_vs_match": sum(bool(r["correct"]) != bool(r.get("answer_in_response")) for r in rows.values())})
    data["systems"].append(s)
    data["grid"]["columns"].append({"label": label, "cells": {i: ("yes" if rows[i]["correct"] else "lenient" if lenient(rows[i]) else "no")
                                                              for i in paper_order if i in rows}})
trial = groups[("sas", "trial")]
ts = summary(trial)
rs = list(trial.values())
ts.update({"median_seconds": st.median(r["seconds"] for r in rs), "max_seconds": max(r["seconds"] for r in rs),
           "hit_turn_limit": sum(r["stop"] == "max_turns" for r in rs),
           "judge_vs_match": sum(bool(r["correct"]) != bool(r.get("answer_in_response")) for r in rs),
           "first10": sum(bool(trial[i]["correct"]) for i in paper_order)})
data["trial"] = ts
data["grid"]["columns"].insert(1, {"label": "Single agent, 100-question run", "cells": {
    i: ("yes" if trial[i]["correct"] else "lenient" if lenient(trial[i]) else "no") for i in paper_order}})

# How often at least one agent's own final answer held the gold answer, against what the setup was credited with.
# Independent: each worker's last answer; agent-driven: each agent's last reply (the team keeps only the last one).
def found_by_some_agent(kind, tag):
    found = 0
    for d in sorted((HERE.parent / "runs").glob(f"*-{kind}-browsecomp-gpt-oss-20b-{tag}*")):
        for line in open(d / "results.jsonl"):
            r = json.loads(line)
            t = json.load(open(d / "traces" / f"{r['id']}.json"))
            if "worker_answers" in t:
                answers = [v[-1] for v in t["worker_answers"].values() if v]
            else:
                last = {}
                for e in t["events"]:
                    if e["kind"] == "idle" and e["reply"].strip():
                        last[e["agent"]] = e["reply"]
                answers = list(last.values())
            gold = norm(r["expected"])
            found += any(gold and f" {gold} " in f" {norm(a)} " for a in answers)
    return found


data["found"] = [{"label": label, "found": found_by_some_agent(kind, tag), "credited": next(s["correct"] for s in data["systems"] if s["label"] == label)}
                 for label, kind, tag in SYSTEMS if kind in ("independent", "agent-driven-none", "agent-driven-text-coord")]

html = (HERE / "browsecomp.html").read_text().replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
(HERE / "browsecomp-final.html").write_text(html)
print("written", len(html) // 1024, "KB;", {s["label"]: f"{s['correct']}/{s['n']}" for s in data["systems"]})
