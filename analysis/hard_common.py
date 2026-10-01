"""Helpers shared by the hard-suite analyses (hard64.py, hard64_repeats.py): what a run changed, and why it failed."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import workbench as wb  # noqa: E402
from src.evals.actions import convert_intermediate_step_to_function_call as to_action  # noqa: E402

DOTTED = {re.sub(r"[^a-zA-Z0-9_-]", "_", t.name): t.name for t in wb.TOOLS_IN_ORDER}
REJECT = re.compile(r"not valid|not found|missing|not provided|invalid|error|does not exist|no .* found|please", re.I)
KEYS = ["email_id", "event_id", "task_id", "customer_id", "value_to_plot", "event_name", "task_name", "customer_name", "recipient"]


def norm(a):
    return a.lower()


def main_tool(gt):
    return Counter(a.split(".func(")[0] for a in gt).most_common(1)[0][0] if gt else None


def classify(made, gt, correct):
    """Why a run failed, by comparing its changes with the correct ones (multisets, case-insensitive)."""
    if correct:
        return "correct"
    M, G = Counter(map(norm, made)), Counter(map(norm, gt))
    if not M:
        return "no changes"
    tool = main_tool(gt)
    n_main_gt = sum(a.startswith(tool) for a in gt)
    n_main = sum(a.startswith(tool) for a in made)
    if n_main_gt > 5 and n_main == 5:
        return "stopped at the 5-result cap"
    extra, missing = M - G, G - M
    if not missing and extra:
        return "right changes plus extras"
    if not (set(M) & set(G)):
        return "only wrong changes"
    if missing and not extra:
        return "some changes missing"
    return "mix of right, wrong and missing"


def target(change):
    args = dict(re.findall(r'(\w+)="([^"]*)"', change))
    for k in KEYS:
        if k in args:
            return (change.split("(")[0], k, args[k].lower())
    return (change,)


def effective_writes(conv):
    outs = {m["tool_call_id"]: m["content"] for m in conv if m.get("role") == "tool"}
    acts = []
    for m in conv:
        for tc in m.get("tool_calls") or []:
            d = DOTTED.get(tc["function"]["name"])
            try:
                a = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                a = {}
            if d in wb.WRITE_TOOLS and isinstance(a, dict) and not REJECT.search(outs.get(tc["id"], "") or ""):
                acts.append(to_action(d, {k: str(v) for k, v in a.items() if v is not None}))
    return acts
