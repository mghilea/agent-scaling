"""CooperBench tasks for our own agent harness (agent.py --benchmark cooperbench), instead of CooperBench's agents.

    $COOPERBENCH_PYTHON agent.py --benchmark cooperbench --subset core [--limit N] --repair-calls

A task is one feature pair of a CooperBench task (dataset/subsets/<subset>.json): two features to add to one repository.
The single agent (CooperBench's "solo" setting) gets both feature descriptions, combined as CooperBench's solo runner
does, and one tool, execute_bash, which runs in its own Apptainer container of the task's image
(cooperbench.infra.apptainer, from cooperbench/apptainer.patch) with the repository at /workspace/repo. Its submission
is what it leaves changed in the working tree (no commit or pull request), taken as a git diff, and graded by
CooperBench's own test_solo: each feature's tests run on that patch in a fresh container; the task counts as solved
only if both features' tests pass.

Runs with CooperBench's venv ($COOPERBENCH_PYTHON, from COOPERBENCH=1 source setup.sh) and source tree
($COOPERBENCH_SRC). CooperBench's dataset license is unknown: task contents stay out of git and off artifact pages.
"""
import getpass
import json
import os
import sys
from pathlib import Path

SRC = Path(os.environ.get("COOPERBENCH_SRC", f"/scratch/{getpass.getuser()}/cooperbench-src"))
DATASET = SRC / "dataset"
sys.path.insert(0, str(SRC / "src"))

from cooperbench.eval.sandbox import test_solo  # noqa: E402
from cooperbench.infra.apptainer import Container  # noqa: E402
from cooperbench.utils import get_image_name  # noqa: E402

REPO = "/workspace/repo"
COMMAND_TIMEOUT = 600        # seconds per command (tests can be slow)
MAX_TOOL_OUTPUT = 12000      # characters of a command's output the agent sees

SYSTEM_PROMPT = """You are a software engineer working in a git repository at /workspace/repo, inside a Linux container.
You have one tool, execute_bash, which runs a bash command in /workspace/repo and returns its output (stdout and stderr
together) and exit code. Each command runs in a new shell: `cd` and environment variables don't carry over, but files
you change do. Commands time out after 10 minutes; don't start servers or anything interactive.

Implement the features described in the task:
1. Read the relevant code first (ls, grep, sed -n, cat).
2. Make the changes: edit files with sed, python, or by writing them out (cat > file <<'EOF' ... EOF).
3. Check your work: build the code and run the relevant existing tests, or a small script of your own.
Your changes to the working tree are your submission: you don't need to commit or open a pull request. Don't edit or
add test files; hidden tests will check each feature. When both features are done, reply with a short summary of what
you changed, without calling a tool."""


def load_tasks(subset="core", limit=None, offset=0):
    """One task per feature pair of the subset, in the subset's order."""
    spec = json.load(open(DATASET / "subsets" / f"{subset}.json"))
    tasks = [{"id": f"{t['repo']}-{t['task_id']}-f{a}-f{b}", "repo": t["repo"], "task_id": t["task_id"],
              "features": [a, b], "question": f"(CooperBench {t['repo']} task {t['task_id']}, features {a} and {b})"}
             for t in spec["tasks"] for a, b in t["pairs"]]
    tasks = tasks[offset:]
    return tasks[:limit] if limit else tasks


def _schema(name, description, params):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": params, "required": list(params)}}}


class Session:
    """One feature pair: the task's container, the agent's tool, its patch and its score."""

    def __init__(self, task):
        self.task = task
        self.container = Container(get_image_name(task["repo"], task["task_id"]))
        self.reminded = False
        self.tools = {"execute_bash": (self.execute_bash, _schema(
            "execute_bash", "Run a bash command in /workspace/repo and return its output and exit code.",
            {"command": {"type": "string", "description": "The bash command to run"}}))}

    def execute_bash(self, command):
        try:
            r = self.container.run(["bash", "-lc", command], cwd=REPO, timeout=COMMAND_TIMEOUT)
        except Exception as e:  # e.g. the timeout
            return f"[error] {type(e).__name__}: {e}"
        return f"{r.stdout}\n[exit code {r.returncode}]" if r.returncode else (r.stdout or "[no output]")

    def task_prompt(self):
        """Both features, combined as CooperBench's solo runner combines them."""
        task_dir = DATASET / self.task["repo"] / f"task{self.task['task_id']}"
        parts = [f"## Feature {f}\n\n{(task_dir / f'feature{f}' / 'feature.md').read_text()}" for f in self.task["features"]]
        return "\n\n---\n\n".join(parts)

    def patch(self):
        """Everything the agent changed in the working tree, new files included, as a diff against the base commit."""
        r = self.container.run(["bash", "-c", "git add -A >/dev/null 2>&1; git diff --cached --binary HEAD"],
                               cwd=REPO, timeout=COMMAND_TIMEOUT)
        return r.stdout if r.returncode == 0 else ""

    def score(self):
        patch = self.patch()
        a, b = self.task["features"]
        r = test_solo(self.task["repo"], self.task["task_id"], a, b, patch=patch, backend="apptainer",
                      dataset_dir=DATASET)
        f1, f2 = r.get("feature1") or {}, r.get("feature2") or {}
        return {"correct": bool(r.get("both_passed")), "feature1_passed": bool(f1.get("passed")),
                "feature2_passed": bool(f2.get("passed")), "patch_lines": r.get("patch_lines", 0),
                "eval_error": r.get("error"), "reminded": self.reminded,
                "_patch": patch, "_test_output": {"feature1": f1.get("test_output", ""), "feature2": f2.get("test_output", "")}}

    def close(self):
        self.container.remove()
