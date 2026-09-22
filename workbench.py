"""WorkBench (Styles et al. 2024, arXiv 2405.00823), one of the benchmarks in arXiv 2512.08296.

690 workplace tasks ("Reply to Carlos's last email about 'Task Update'...") over five sandbox
databases (email, calendar, web analytics, project management, CRM) plus a company directory,
with 27 tools. A task is scored by replaying the agent's tool calls on a fresh sandbox and
comparing the final database state with the ground truth's, using WorkBench's own evaluator.
"Side effects" means the agent changed the state but got it wrong (e.g. emailed the wrong person).

Setup mirrors the WorkBench Revisited (2026) runs: all tools on every task, native tool calling,
the benchmark's date prefix as the system prompt, and the "act without confirmation" suffix.

The WorkBench repo is cloned by setup.sh into $WORKBENCH_DIR (default /scratch/$USER/WorkBench).
"""
import ast
import getpass
import os
import random
import re
import sys
from pathlib import Path

WORKBENCH_DIR = Path(os.environ.get("WORKBENCH_DIR", f"/scratch/{getpass.getuser()}/WorkBench"))
if not (WORKBENCH_DIR / "src" / "tools").is_dir():
    sys.exit(f"WorkBench not found at {WORKBENCH_DIR}. Source setup.sh on this node, or set WORKBENCH_DIR.")
sys.path.insert(0, str(WORKBENCH_DIR))

import pandas as pd  # noqa: E402

from src.evals.actions import convert_intermediate_step_to_function_call  # noqa: E402
from src.evals.evaluation import has_side_effects, is_correct, is_exact_match  # noqa: E402
from src.tools import state as _state  # noqa: E402
from src.tools.state import reset_state  # noqa: E402
from src.tools.tool import tool_to_openai_schema  # noqa: E402
from src.tools.toolkits import all_tools  # noqa: E402

# WorkBench loads its databases from paths relative to its repo root; make them absolute.
_state._CSV_PATHS = {k: str(WORKBENCH_DIR / v) for k, v in _state._CSV_PATHS.items()}

DOMAINS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager", "multi_domain"]

# Copied from WorkBench's src/evals/inference.py and agent.py (datetime_prefix and
# ACT_WITHOUT_CONFIRMATION_SUFFIX), which are what the 2026 runs used as the system prompt.
SYSTEM_PROMPT = (
    "Today's date is Thursday, 2023-11-30 and the current time is 00:00:00. "
    "Remember the current date and time when completing tasks. "
    "Meetings must not start before 9am or end after 6pm. "
    "Do not ask for confirmation before executing actions. "
    "Execute actions immediately and continue until the task is fully complete. "
    "Do not stop after a search or lookup step."
)

# WorkBench's own tool order when every toolkit is enabled: the five domains, then the directory.
_TOOL_ORDER = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager",
               "company_directory"]
TOOLS_IN_ORDER = [t for prefix in _TOOL_ORDER for t in all_tools if t.name.split(".")[0] == prefix]


def load_tasks(domains=None, limit=None, seed=0):
    """Tasks as dicts: id (e.g. email-17), question, outcome (ground-truth action strings), template.

    With limit, returns a seeded random sample, so the same --limit and --seed give the same tasks.
    """
    tasks = []
    for domain in domains or DOMAINS:
        df = pd.read_csv(WORKBENCH_DIR / "data/processed/tasks_and_outcomes" / f"{domain}_tasks_and_outcomes.csv")
        for i, row in df.iterrows():
            tasks.append({"id": f"{domain}-{i}", "question": row["task"],
                          "outcome": ast.literal_eval(row["outcome"]), "template": row["base_template"]})
    if limit:
        tasks = random.Random(seed).sample(tasks, min(limit, len(tasks)))
    return tasks


class Episode:
    """One task's sandbox: fresh databases, tools that log every call, and scoring at the end."""

    def __init__(self):
        reset_state()
        self.actions = []  # WorkBench action strings, e.g. 'email.delete_email.func(email_id="00000479")'
        self.tools = {}
        for t in TOOLS_IN_ORDER:
            schema = tool_to_openai_schema(t)
            # Dots aren't valid in OpenAI function names, so email.send_email becomes email_send_email.
            schema["function"]["name"] = re.sub(r"[^a-zA-Z0-9_-]", "_", t.name)
            self.tools[schema["function"]["name"]] = (self._recorder(t), schema)

    def _recorder(self, t):
        def run(**kwargs):
            # WorkBench passes every argument as a string. JSON nulls mean "not given", so drop them.
            kwargs = {k: str(v) for k, v in kwargs.items() if v is not None}
            out = str(t(**kwargs))
            # Only calls that ran are scored. A call that raised (e.g. an argument the tool doesn't
            # have) changed nothing, and the agent saw the error and could retry. WorkBench's own
            # runner instead ends the task as an error there, so it is stricter than this.
            self.actions.append(convert_intermediate_step_to_function_call(t.name, kwargs))
            return out
        return run

    def score(self, expected, error=False):
        # As in WorkBench, a run that crashed or hit the turn limit counts as wrong.
        correct = is_correct(self.actions, expected, "error" if error else "")
        return {"correct": correct,
                "side_effects": has_side_effects(self.actions, correct),
                "exact_match": is_exact_match(self.actions, expected)}
