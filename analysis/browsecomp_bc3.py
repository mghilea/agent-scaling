"""The BrowseComp-Plus comparison: 17 setups on the first 10 of the paper's 100 questions, 3 runs each (tags bc3-r1..r3).

    python analysis/browsecomp_bc3.py                # writes analysis/browsecomp_bc3.json
    python analysis/browsecomp_bc3.py --tag smoke    # any tag prefix, e.g. to test on smoke runs

Per setup: accuracy in each run (by the judge, and leniently), its mean and range, cost, and how many of the runs
solved each question. For agent-driven teams (propose-then-decide), also: how much they communicated by choice
(messages, A2A tasks, shared-state writes), what happened in the propose and decide steps (automatic proposals,
reminders, no submission, talk after the proposals), whether a right answer was among the proposals, and whether
the team submitted it. Needs questions.jsonl on this node for the proposal checks; writes ids and numbers only.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import browsecomp as bc  # noqa: E402
from browsecomp_runs import lenient  # noqa: E402

NAME = re.compile(r"\d{8}-\d{6}-(?P<kind>.+)-browsecomp-gpt-oss-20b-(?P<tag>.+)$")
COMM = ("message", "a2a_send", "a2a_update", "state_write")


def contains(text, gold):
    g, t = bc._norm(gold), bc._norm(text or "")
    return bool(g) and f" {g} " in f" {t} "


def label(kind):
    split = "split" if "-split3" in kind else "unsplit"
    base = kind.replace("-split3", "").replace("-propose", "")
    names = {"sas": "Single agent", "independent": "Independent", "centralized": "Centralized",
             "decentralized": "Decentralized", "hybrid": "Hybrid"}
    if base.startswith("agent-driven-"):
        comm = base[len("agent-driven-"):].replace("-coord", "")
        name = {"text": "Agent-driven, text", "a2a": "Agent-driven, A2A", "shared": "Agent-driven, shared state",
                "all": "Agent-driven, all channels"}.get(comm, base)
    else:
        name = names.get(base, base)
    return name, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bc3-r")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in bc.load_tasks()}
    runs = defaultdict(list)  # (name, split) -> [(tag, rows, run dir)]
    for d in sorted((ROOT / "runs").iterdir()):
        m = NAME.match(d.name)
        if not m or not m["tag"].startswith(args.tag) or not (d / "results.jsonl").exists():
            continue
        rows = [json.loads(line) for line in open(d / "results.jsonl")]
        runs[label(m["kind"])].append((m["tag"], rows, d))

    out = {"setups": [], "questions": {}}
    for (name, split), group in sorted(runs.items()):
        per_run = [sum(bool(r["correct"]) for r in rows) / len(rows) for _, rows, _ in group if rows]
        per_run_len = [sum(lenient(r) for r in rows) / len(rows) for _, rows, _ in group if rows]
        allrows = [r for _, rows, _ in group for r in rows]
        s = {"setup": name, "split": split, "runs": len(group), "questions": sorted({r["id"] for r in allrows}),
             "per_run": per_run, "mean": st.mean(per_run), "min": min(per_run), "max": max(per_run),
             "lenient_mean": st.mean(per_run_len),
             "tokens": st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in allrows),
             "seconds": st.mean(r["seconds"] for r in allrows), "complete": all(len(rows) == 10 for _, rows, _ in group)}
        solved = defaultdict(int)
        for r in allrows:
            solved[r["id"]] += bool(r["correct"])
        s["solved"] = dict(solved)
        driven = [(r, json.load(open(d / "traces" / f"{r['id']}.json"))) for _, rows, d in group for r in rows if r.get("comm")]
        if driven:
            n = len(driven)
            c = lambda key: st.mean((r["comm"].get(key) or 0) for r, _ in driven)  # noqa: E731
            right_proposed = right_submitted = 0
            for r, t in driven:
                gold = tasks[r["id"]]["answer"]
                if any(contains(p["answer"], gold) for p in t.get("proposals", [])):
                    right_proposed += 1
                    right_submitted += bool(r["correct"])
            s.update({
                "chosen_messages": c("messages"), "chosen_a2a": c("a2a_tasks") + c("a2a_updates"),
                "chosen_state_writes": c("state_writes"),
                "question_runs_with_chosen_comm": sum(any(e["kind"] in COMM for e in t["events"]) for _, t in driven),
                "question_runs": n, "proposals": c("proposals"), "auto_proposals": c("auto_proposals"),
                "comm_after_proposals": c("comm_after_proposals"),
                "no_submission": sum(r["comm"].get("answer_rule") == "no_submission" for r, _ in driven),
                "reminded": sum(bool(r["comm"].get("team_nudged")) for r, _ in driven),
                "right_among_proposals": right_proposed, "right_submitted": right_submitted,
                "decide_seconds": st.mean([r["comm"]["decided_at"] - r["comm"]["proposals_at"] for r, _ in driven
                                           if r["comm"].get("decided_at") and r["comm"].get("proposals_at")] or [0]),
            })
        out["setups"].append(s)
    (HERE / "browsecomp_bc3.json").write_text(json.dumps(out, indent=1))
    print(f"{'setup':30s} {'split':8s} {'runs':>4s} {'mean':>5s} {'range':>9s} {'tokens/q':>9s} {'s/q':>5s}  driven: chosen comm per q | proposals (auto) | no submission | right proposed -> submitted")
    for s in out["setups"]:
        extra = ""
        if "proposals" in s:
            extra = (f"  msgs {s['chosen_messages']:.1f} a2a {s['chosen_a2a']:.1f} state {s['chosen_state_writes']:.1f} | "
                     f"{s['proposals']:.1f} ({s['auto_proposals']:.1f}) | {s['no_submission']}/{s['question_runs']} | "
                     f"{s['right_among_proposals']} -> {s['right_submitted']}")
        print(f"{s['setup']:30s} {s['split']:8s} {s['runs']:4d} {s['mean']:5.0%} {s['min']:4.0%}-{s['max']:<4.0%} "
              f"{s['tokens'] / 1000:8.0f}k {s['seconds']:5.0f}{'' if s['complete'] else '  (incomplete)'}{extra}")


if __name__ == "__main__":
    main()
