"""AppWorld: day-to-day tasks across 9 apps (email, shopping, payments, music, notes, ...) driven through 457 APIs.

AppWorld (Trivedi et al., 2024; github.com/stonybrooknlp/appworld) simulates ~100 people with years of app activity.
A task is an instruction from a "supervisor" (the person the agent works for), with its own copy of the
databases, prepared with distractors. The agent acts by writing Python that calls `apis.<app>.<api>(...)` in a
stateful shell. A task is solved when the final databases contain every expected change and nothing else
(unexpected changes are "collateral damage"), and, for questions, the answer passed to
apis.supervisor.complete_task is right. Metrics: task goal completion (share of tasks solved) and scenario goal
completion (share of scenarios, 3 tasks each, with all 3 solved).

setup.sh with APPWORLD=1 installs AppWorld in its own venv ($APPWORLD_PYTHON; its pinned dependencies would clash
with vLLM's) and its data in $APPWORLD_ROOT (default /scratch/$USER/appworld); run agent.py with that Python. The
data's license requires any public redistribution to be encrypted, so task contents stay in /scratch and runs/:
not in git, not on artifact pages.

The agent gets one tool, execute_python, which runs code in the task's shell, as in AppWorld's code-as-action
baselines. The system prompt below gives AppWorld's key instructions (API docs, logins, pagination,
complete_task); it is ours, not the paper's ReAct prompt with its worked example, so scores are not directly
comparable with the paper's.
"""
import getpass
import os
import random
from pathlib import Path

os.environ.setdefault("APPWORLD_ROOT", f"/scratch/{getpass.getuser()}/appworld")
from appworld import AppWorld, load_task_ids  # noqa: E402

DATASETS = ["train", "dev", "test_normal", "test_challenge"]
# AppWorld's API listings and responses are long (an app's API list alone can pass 4,000 characters), so the agent
# sees up to this many characters of each result instead of agent.py's default.
MAX_TOOL_OUTPUT = 20000

SYSTEM_PROMPT = """You are an AI assistant who completes day-to-day tasks for your supervisor, {first_name} {last_name} (email {email}, phone {phone_number}), by operating their apps through APIs.

You work in a stateful Python shell with the tool execute_python. Variables you define stay available in later calls. Every app is reachable through the `apis` object, e.g. `apis.spotify.login(...)`, and an API's response is returned as Python data; print what you need to see.

How to work:
1. Find the APIs you need before using them: `print(apis.api_docs.show_app_descriptions())`, then `print(apis.api_docs.show_api_descriptions(app_name="spotify"))`, then `print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))` for an API's parameters and response.
2. Credentials are not something you know: get your supervisor's account passwords with `apis.supervisor.show_account_passwords()` and log in to each app you use (its `login` API) to get the access_token the other APIs need. Personal details (addresses, payment cards) are under `apis.supervisor` too.
3. Many APIs return results in pages: loop over page_index until a page comes back empty, so you see everything.
4. Write small pieces of code and check their output before going on. Make only the changes the task asks for: anything else (deleting, buying, sending or editing something not asked for) counts as a mistake.
5. Dates and times: use the ones the apps report, not your own sense of today's date.
6. When you are done, call `apis.supervisor.complete_task()`. If the task asks for information, pass it as the answer, e.g. `apis.supervisor.complete_task(answer=42)`, using just the value (a number, a name, or a short phrase), not a sentence. If the task asks for no information, call it without an answer. Then reply with a short summary and no tool call."""

TASK_PROMPT = "Task from your supervisor:\n{instruction}"
# Sent once if the agent stops (replies without a tool call) before calling complete_task.
COMPLETE_REMINDER = ("The task is not marked complete: apis.supervisor.complete_task() has not been called. If you are "
                     "done, call it now, with answer=... if the task asks for information and without an answer "
                     "otherwise. If you are not done, continue the task.")


def load_tasks(dataset="dev", limit=None, offset=0, seed=0):
    """Task ids of an AppWorld split, as task dicts: in AppWorld's order, or a seeded sample with limit."""
    ids = load_task_ids(dataset)
    if limit and limit < len(ids) and not offset:
        ids = sorted(random.Random(seed).sample(ids, limit), key=ids.index)
    else:
        ids = ids[offset:offset + limit if limit else None]
    return [{"id": i, "question": f"(AppWorld {dataset} task {i})"} for i in ids]


def _schema(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


class Session:
    """One task's world: the execute_python tool, the prompts, and AppWorld's evaluation at the end."""

    def __init__(self, task_id, experiment_name):
        self.world = AppWorld(task_id=task_id, experiment_name=experiment_name)
        self.task = self.world.task
        self.codes = []  # every piece of code the agent ran, in order
        self.reminded = False  # whether the agent got COMPLETE_REMINDER
        self.tools = {"execute_python": (self._execute, _schema(
            "execute_python",
            "Run Python code in the task's shell (variables persist between calls; apps are under `apis`) and "
            "return what it printed, or the error.",
            {"code": {"type": "string", "description": "The Python code to run"}}, ["code"]))}

    def _execute(self, code):
        self.codes.append(code)
        return self.world.execute(code)

    def system_prompt(self):
        return SYSTEM_PROMPT.format(**self.task.supervisor)

    def task_prompt(self):
        return TASK_PROMPT.format(instruction=self.task.instruction)

    def score(self):
        """AppWorld's verdict: solved only if every test passes (the expected changes, nothing else, the answer)."""
        r = self.world.evaluate()
        return {"correct": bool(r.success), "tests_passed": r.pass_count, "tests": r.num_tests,
                "difficulty": r.difficulty, "completed": self.world.task_completed(), "code_blocks": len(self.codes),
                "reminded": self.reminded}

    def close(self):
        self.world.close()


def scenario(task_id):
    """The scenario a task belongs to (AppWorld has 3 tasks per scenario): '50e1ac9_2' -> '50e1ac9'."""
    return task_id.rsplit("_", 1)[0]


def scenario_completion(rows):
    """Share of scenarios whose tasks in these rows were all solved (AppWorld's SGC, over complete scenarios)."""
    by = {}
    for r in rows:
        by.setdefault(scenario(r["id"]), []).append(bool(r["correct"]))
    full = [v for v in by.values() if len(v) == 3] or list(by.values())
    return sum(all(v) for v in full) / len(full) if full else None
