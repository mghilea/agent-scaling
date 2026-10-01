"""The 64-task hard suite repeated: every system run 10 more times (suites/hard-a.txt and hard-b.txt, --tag repNN).

    python analysis/hard64_repeats.py   # writes analysis/hard64_repeats.json

For each system: accuracy over the repeats (mean, spread, 95% interval of the mean) overall, by task size,
and split by whether the task is cap-bound (its main action is needed more than 5 times, so a single search
can't return every record); how often each run fails each way; and, for the agent-driven teams, how much
they communicated and duplicated. The original single runs (hard64.py's RUNS) are kept alongside for comparison.
"""
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import workbench as wb  # noqa: E402
from hard_common import classify, effective_writes, main_tool, target  # noqa: E402

SYSTEMS = {  # run-directory kind -> label, in the order the page shows them
    "sas": "Single agent", "independent": "Independent", "centralized": "Centralized",
    "decentralized": "Decentralized", "hybrid": "Hybrid", "agent-driven-none": "Agent-driven, no channels",
    "agent-driven-text-coord": "Agent-driven, text + coordinate",
    "agent-driven-all-coord": "Agent-driven, all channels + coordinate",
}
ORIGINAL = {  # the first run of each system, 30 Sep 2026 (same as hard64.py)
    "Single agent": "20260930-201438-sas-workbench-gpt-oss-20b",
    "Independent": "20260930-201858-independent-workbench-gpt-oss-20b",
    "Centralized": "20260930-201433-centralized-workbench-gpt-oss-20b",
    "Decentralized": "20260930-201428-decentralized-workbench-gpt-oss-20b",
    "Hybrid": "20260930-201853-hybrid-workbench-gpt-oss-20b",
    "Agent-driven, no channels": "20260930-201443-agent-driven-none-workbench-gpt-oss-20b",
    "Agent-driven, text + coordinate": "20260930-201903-agent-driven-text-coord-workbench-gpt-oss-20b",
    "Agent-driven, all channels + coordinate": "20260930-201908-agent-driven-all-coord-workbench-gpt-oss-20b",
}
COMM = ("message", "a2a_send", "a2a_update", "state_write")
N_TASKS = 64
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20,
       12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09}


def summary(values):
    """Mean, spread and a 95% t-interval of the mean over runs."""
    n = len(values)
    mean = sum(values) / n if n else None
    sd = st.stdev(values) if n > 1 else 0.0
    half = T95.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": min(values, default=None), "max": max(values, default=None),
            "lo": mean - half if n else None, "hi": mean + half if n else None, "values": values}


def analyze_run(run_dir, label):
    """Per-run metrics, and per-task correctness, for one complete run of one system."""
    rows = {r["id"]: r for r in map(json.loads, open(run_dir / "results.jsonl"))}
    driven = "agent-driven" in run_dir.name
    out = {"tasks": {}, "fail_kinds": Counter(), "cap_passed": 0, "cap_tasks": 0}
    acc = defaultdict(list)
    side, tok, sec, acts, comm_tasks, overlap, changers, lone = [], [], [], [], 0, 0, [], defaultdict(list)
    for i, r in sorted(rows.items()):
        gt, ok = r["expected"], bool(r["correct"])
        tool = main_tool(gt)
        cap_bound = sum(a.startswith(tool) for a in gt) > 5
        size = "5+" if len(gt) >= 5 else "3-4"
        for k in ("all", size, "cap" if cap_bound else "not_cap"):
            acc[k].append(ok)
        out["tasks"][i] = ok
        side.append(bool(r.get("side_effects")))
        tok.append(r["prompt_tokens"] + r["completion_tokens"])
        sec.append(r["seconds"])
        trace = json.load(open(run_dir / "traces" / f"{i}.json"))
        if driven:
            made = [c["change"].replace("(", ".func(", 1) for c in trace["changes"]]
        elif label == "Single agent":
            made = effective_writes(trace)
        else:
            made = [a for a in r["answer"] if wb.is_write(a)]
        out["fail_kinds"][classify(made, gt, ok)] += 1
        if cap_bound:
            out["cap_tasks"] += 1
            out["cap_passed"] += sum(a.startswith(tool) for a in made) > 5
        if label in ("Independent", "Decentralized"):  # each worker's own changes, scored alone
            for c in trace["conversations"].values():
                w = wb.score(effective_writes(c), gt)["correct"]
                for k in ("all", size, "cap" if cap_bound else "not_cap"):
                    lone[k].append(w)
        if driven:
            ev = trace["events"]
            n_comm = sum(e["kind"] in COMM for e in ev)
            acts.append(n_comm)
            comm_tasks += n_comm > 0
            who = defaultdict(set)
            for c in trace["changes"]:
                who[target(c["change"])].add(c["agent"])
            overlap += any(len(a) > 1 for a in who.values())
            changers.append(len({c["agent"] for c in trace["changes"]}))
    n = len(rows)
    m = {f"acc_{k}": sum(v) / len(v) for k, v in acc.items()}
    m.update({"side_effects": sum(side) / n, "tokens": sum(tok) / n, "seconds": sum(sec) / n,
              "cap_passed": out["cap_passed"]})
    if driven:
        m.update({"comm_acts": sum(acts) / n, "tasks_with_comm": comm_tasks, "duplicate_tasks": overlap,
                  "agents_changing": sum(changers) / n})
    out["metrics"] = m
    out["lone"] = {k: (sum(v), len(v)) for k, v in lone.items()}
    out["sizes"] = {k: len(v) for k, v in acc.items()}
    return out


def kind_of(name):
    m = re.match(r"\d{8}-\d{6}-(.+)-workbench-gpt-oss-20b(?:-(rep\d+))?$", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def complete(run_dir):
    try:
        return sum(1 for _ in open(run_dir / "results.jsonl")) == N_TASKS
    except FileNotFoundError:
        return False


runs, incomplete = defaultdict(list), []
for d in sorted((ROOT / "runs").iterdir()):
    kind, rep = kind_of(d.name)
    if kind in SYSTEMS and rep:
        (runs[SYSTEMS[kind]] if complete(d) else incomplete).append(d if complete(d) else d.name)

data = {"systems": {}, "tasks": {}, "incomplete": incomplete, "n_tasks": N_TASKS}
per_task = defaultdict(lambda: defaultdict(list))
for label in SYSTEMS.values():
    rep_out = [analyze_run(d, label) for d in runs[label]]
    orig = analyze_run(ROOT / "runs" / ORIGINAL[label], label)
    if not rep_out:
        continue
    metrics = {k: summary([r["metrics"][k] for r in rep_out]) for k in rep_out[0]["metrics"]}
    kinds = Counter()
    for r in rep_out:
        kinds.update(r["fail_kinds"])
    lone = {}
    for r in rep_out:
        for k, (s, n) in r["lone"].items():
            lone.setdefault(k, [0, 0])
            lone[k][0] += s
            lone[k][1] += n
    for r in rep_out:
        for i, ok in r["tasks"].items():
            per_task[i][label].append(ok)
    data["systems"][label] = {
        "runs": [d.name for d in runs[label]], "metrics": metrics, "original": orig["metrics"],
        "fail_kinds": {k: v / len(rep_out) for k, v in kinds.items()},  # mean tasks per run
        "cap_tasks": rep_out[0]["cap_tasks"], "sizes": rep_out[0]["sizes"],
        "lone": {k: s / n for k, (s, n) in lone.items()} if lone else None,
    }

# How often one system's run beats another's, over all pairs of runs (ties count half).
labels = list(data["systems"])
for key in ("acc_all", "acc_3-4", "acc_5+", "acc_cap", "acc_not_cap"):
    beats = {}
    for a in labels:
        for b in labels:
            if a == b:
                continue
            va, vb = data["systems"][a]["metrics"][key]["values"], data["systems"][b]["metrics"][key]["values"]
            wins = sum((x > y) + 0.5 * (x == y) for x in va for y in vb)
            beats[f"{a} | {b}"] = wins / (len(va) * len(vb))
    data.setdefault("beats", {})[key] = beats

# Per task: how many runs solved it, per system, with its size and whether it is cap-bound.
gts = {t["id"]: t["outcome"] for t in wb.load_tasks(min_changes=3)}
for i, by in sorted(per_task.items()):
    gt = gts[i]
    tool = main_tool(gt)
    data["tasks"][i] = {"changes": len(gt), "cap": sum(a.startswith(tool) for a in gt) > 5,
                        "solved": {s: sum(v) for s, v in by.items()}, "runs": {s: len(v) for s, v in by.items()}}

(HERE / "hard64_repeats.json").write_text(json.dumps(data, indent=1))
print(f"incomplete runs: {incomplete}")
for label, s in data["systems"].items():
    m = s["metrics"]
    print(f"{label:42s} n={m['acc_all']['n']:2d}  all {m['acc_all']['mean']:.0%} ±{m['acc_all']['sd']:.0%}  "
          f"3-4 {m['acc_3-4']['mean']:.0%}  5+ {m['acc_5+']['mean']:.0%}  cap {m['acc_cap']['mean']:.0%}  "
          f"not-cap {m['acc_not_cap']['mean']:.0%}  (original {s['original']['acc_all']:.0%})")
