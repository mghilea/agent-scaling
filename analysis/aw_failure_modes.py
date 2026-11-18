"""Failure modes of the AppWorld ladder (gpt-oss-120b, 50 hard tasks, teams of 3 at L0-L6, runs hlad01/02).

    python analysis/aw_failure_modes.py

For each task, the reference is what the team runs that solved it changed: per API, the fewest and most writes among
those runs. A failed run is classified by its writes against that: missing changes (omission), extra changes (beyond
any solved run; also whether a teammate duplicated a write), both, or neither (same changes, wrong content or
answer). Missing changes are traced to: refused and never redone, claimed in an intent and never made, attempted and
failed, or never attempted. Also: success on the tasks the single agent always solves, and (L3-L6) success with one
agent making every change vs several, within the same tasks. Prints API names and numbers only: run it locally, and
keep task contents off pages (AppWorld license).
"""
from collections import Counter, defaultdict
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
kind = lambda L: "agent-driven-none-rc" if L == "L0" else f"agent-driven-none-{L}-rc"
runs = []   # (level, tag, row, trace)
for L in LEVELS:
    for tag in ("hlad01", "hlad02"):
        d = glob.glob(f"runs/*-{kind(L)}-appworld-test_challenge-gpt-oss-120b-{tag}")[0]
        for line in open(d + "/results.jsonl"):
            r = json.loads(line)
            runs.append((L, tag, r, json.load(open(f"{d}/traces/{r['id']}.json"))))
single = defaultdict(list)
for tag in ("hlad01", "hlad02"):
    d = glob.glob(f"runs/*-sas-rc-appworld-test_challenge-gpt-oss-120b-{tag}")[0]
    for line in open(d + "/results.jsonl"):
        r = json.loads(line); single[r["id"]].append(bool(r["correct"]))

def writes(tr):
    return [e for e in tr["events"] if e["kind"] == "api_write" and e.get("api") != "complete_task"]
def api_counts(tr):
    return Counter((e["app"], e["api"]) for e in writes(tr) if e["ok"])
def dup_keys(tr):
    first, dups = {}, Counter()
    for e in writes(tr):
        k = (e["app"], e["api"], tuple(e["ids"]) or tuple(e["identity"]))
        if e["ok"]:
            if k in first and first[k] != e["agent"]:
                dups[(e["app"], e["api"])] += 1
            first.setdefault(k, e["agent"])
    return dups

# reference: what the team runs that solved a task changed, per API: [min, max] count over those runs
ref = {}
for t in {r["id"] for _, _, r, _ in runs}:
    ok = [api_counts(tr) for _, _, r, tr in runs if r["id"] == t and r["correct"]]
    if ok:
        apis = set().union(*ok)
        ref[t] = {a: (min(c[a] for c in ok), max(c[a] for c in ok)) for a in apis}

modes = defaultdict(Counter); causes = defaultdict(Counter); dup_apis = Counter(); missing_apis = Counter()
for L, tag, r, tr in runs:
    if r["correct"]:
        continue
    t = r["id"]
    if t not in ref:
        modes[L]["no reference (no team run solved it)"] += 1
        continue
    c, R = api_counts(tr), ref[t]
    missing = {a: R[a][0] - c[a] for a in R if c[a] < R[a][0]}
    extra = {a: c[a] - R.get(a, (0, 0))[1] for a in c if c[a] > R.get(a, (0, 0))[1]}
    dups = dup_keys(tr)
    m = ("missing + extra" if missing and extra else "missing changes (omission)" if missing
         else "extra changes" if extra else "same changes, wrong content/answer")
    modes[L][m] += 1
    if extra:
        modes[L]["  of which extra includes a teammate's duplicate"] += any(a in dups for a in extra)
        for a in extra:
            dup_apis[a[1]] += 1
    for a in missing:
        missing_apis[a[1]] += 1
        call = f"{a[0]}.{a[1]}("
        refused = any(e["kind"] == "wal_refused" and str(e.get("call", "")).startswith(call) for e in tr["events"])
        failed = any(e["kind"] == "api_write" and not e["ok"] and e["app"] == a[0] and e["api"] == a[1] for e in tr["events"])
        claimed = any(cl["app"] == a[0] and cl["api"] == a[1] for e in tr.get("wal_log", []) if e["kind"] == "intent" for cl in e["claims"])
        made = c[a] > 0
        causes[L]["refused, never redone" if refused else "claimed in an intent, never (fully) made" if claimed
                  else "attempted, failed" if failed else "made too few times" if made else "never attempted"] += 1

print("Failed team runs by failure mode (each level: 100 task runs; reference = what the team runs that solved the task changed)")
print(f"{'':42s}" + "".join(f"{L:>6s}" for L in LEVELS))
for m in ["missing changes (omission)", "extra changes", "  of which extra includes a teammate's duplicate", "missing + extra",
          "same changes, wrong content/answer", "no reference (no team run solved it)"]:
    print(f"{m:42s}" + "".join(f"{modes[L][m]:>6d}" for L in LEVELS))
print(f"{'failed (total)':42s}" + "".join(f"{sum(1 for l, _, r, _ in runs if l == L and not r['correct']):>6d}" for L in LEVELS))
print("\nWhy changes went missing (per missing API in failed runs):")
allc = sorted({k for L in LEVELS for k in causes[L]})
for k in allc:
    print(f"{k:42s}" + "".join(f"{causes[L][k]:>6d}" for L in LEVELS))
print("\nAPIs most often extra:", dup_apis.most_common(8))
print("APIs most often missing:", missing_apis.most_common(8))

print("\nTasks the single agent solved in BOTH runs: team success by level")
solid = [t for t, v in single.items() if all(v)]
print(f"  {len(solid)} tasks; " + ", ".join(f"{L} {st.mean(bool(r['correct']) for l, _, r, _ in runs if l == L and r['id'] in solid):.0%}" for L in LEVELS))
never = [t for t, v in single.items() if not any(v)]
print(f"  tasks the single agent never solved ({len(never)}): team solved " + ", ".join(
    f"{L} {sum(bool(r['correct']) for l, _, r, _ in runs if l == L and r['id'] in never)}" for L in LEVELS) + " (of 2 runs each)")

print("\nL3-L6: success by how many agents made changes")
w = defaultdict(list)
for L, _, r, tr in runs:
    if L in ("L3", "L4", "L5", "L6"):
        n = len({e["agent"] for e in writes(tr) if e["ok"]})
        w[n].append(bool(r["correct"]))
for n in sorted(w):
    print(f"  {n} agent(s) changed things: {len(w[n])} runs, solved {st.mean(w[n]):.0%}")

# 1. within-task: success with one writer vs several (L3-L6), over tasks that have both kinds of runs
by = defaultdict(lambda: defaultdict(list))
for L, _, r, tr in runs:
    if L in ("L3", "L4", "L5", "L6"):
        n = len({e["agent"] for e in writes(tr) if e["ok"]})
        by[r["id"]]["one" if n == 1 else "several" if n >= 2 else "none"].append(bool(r["correct"]))
both = [t for t in by if by[t]["one"] and by[t]["several"]]
print(f"L3-L6, {len(both)} tasks with both kinds of runs: solved with one writer {st.mean(st.mean(by[t]['one']) for t in both):.0%}, "
      f"with several {st.mean(st.mean(by[t]['several']) for t in both):.0%} (task-averaged)")
# 2. extra changes: written by one agent alone, or by several agents (the same kind of change made by two of them)
kinds = defaultdict(Counter)
for L, _, r, tr in runs:
    t = r["id"]
    if r["correct"] or t not in ref:
        continue
    c, R = api_counts(tr), ref[t]
    for a in c:
        if c[a] > R.get(a, (0, 0))[1]:
            who = {e["agent"] for e in writes(tr) if e["ok"] and (e["app"], e["api"]) == a}
            kinds[L]["several agents made this kind of change" if len(who) > 1 else "one agent made too many"] += 1
            kinds[L]["  ...and it's not in any solved run at all" if R.get(a, (0, 0))[1] == 0 else "  ...more times than any solved run"] += 1
for k in ["several agents made this kind of change", "one agent made too many", "  ...more times than any solved run", "  ...and it's not in any solved run at all"]:
    print(f"{k:46s}" + "".join(f"{kinds[L][k]:>6d}" for L in LEVELS))
# 3. the missing reply_to_email cases: was a send_email made instead?
sub = Counter()
for L, _, r, tr in runs:
    t = r["id"]
    if r["correct"] or t not in ref:
        continue
    c, R = api_counts(tr), ref[t]
    for (app, api), (lo, hi) in R.items():
        if c[(app, api)] < lo and api == "reply_to_email":
            sub[f"{L}: send_email instead" if c[(app, "send_email")] > R.get((app, "send_email"), (0, 0))[1] else f"{L}: no substitute"] += 1
print("missing reply_to_email:", dict(sorted(sub.items())))
