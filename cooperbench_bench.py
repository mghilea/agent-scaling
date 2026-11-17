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
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

SRC = Path(os.environ.get("COOPERBENCH_SRC", f"/scratch/{getpass.getuser()}/cooperbench-src"))
DATASET = SRC / "dataset"
sys.path.insert(0, str(SRC / "src"))

from cooperbench.eval.backends import get_backend  # noqa: E402
from cooperbench.eval.sandbox import _filter_test_files, _run_tests, _write_patch, test_solo  # noqa: E402
from cooperbench.infra.apptainer import APPTAINER, Container  # noqa: E402
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
    """One task per feature set of the subset (pairs, or groups in the team3 subsets), in the subset's order."""
    spec = json.load(open(DATASET / "subsets" / f"{subset}.json"))
    tasks = [{"id": f"{t['repo']}-{t['task_id']}-" + "-".join(f"f{f}" for f in fs), "repo": t["repo"],
              "task_id": t["task_id"], "features": list(fs),
              "question": f"(CooperBench {t['repo']} task {t['task_id']}, features {', '.join(map(str, fs))})"}
             for t in spec["tasks"] for fs in (t.get("pairs") or t.get("groups") or [])]
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


# ---- edit tools: bash can't change the repository; edits go through tools (and, in teams, the write-ahead log) ----

EDIT_PROMPT = """You are a software engineer working in a git repository at /workspace/repo, inside a Linux container.
Tools:
- execute_bash runs a bash command in /workspace/repo and returns its output and exit code. Use it to read code, build
  and run tests. It can't change the repository: anything a command writes there is discarded when it ends (files it
  writes elsewhere, like caches in your home directory, are kept). Each command runs in a new shell and times out after
  10 minutes; don't start servers or anything interactive.
- write_file(path, content) creates a file or replaces its whole content.
- replace_in_file(path, old, new) replaces one exact occurrence of `old` in a file with `new` (copy `old` exactly from
  the file, with enough lines around it to be unique).
- delete_file(path) deletes a file.
Paths are relative to /workspace/repo.

Implement the feature(s) described in the task: read the relevant code first, make the changes with the edit tools,
then check them (build, run the relevant existing tests or a small script). Your changes to the repository are your
submission: you don't need to commit. Don't edit or add test files; hidden tests will check each feature. When you are
done, reply with a short summary of what you changed, without calling a tool."""

# Directories bash may write in the repository without it counting as a change it tried to make (build outputs, caches)
_SCRATCH = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".egg-info", "node_modules", "/target/",
            "/build/", "/dist/", ".git/")


def _rmtree(path):
    subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True)
    shutil.rmtree(path, ignore_errors=True)


class Workspace:
    """The task's repository in one container, shared by the agents working on it. Bash commands see it read-only
    (each runs on a throwaway overlay); the edit methods change it directly. Thread-safe: edits take self.lock."""

    def __init__(self, task):
        self.task = task
        self.container = Container(get_image_name(task["repo"], task["task_id"]))
        self.repo = self.container.root / REPO.lstrip("/")
        self.lock = threading.RLock()
        self.homes = {}

    def _home(self, agent):
        if agent not in self.homes:
            self.homes[agent] = (self.container.dir / f"home-{agent}", self.container.dir / f"tmp-{agent}")
            for d in self.homes[agent]:
                d.mkdir(exist_ok=True)
        return self.homes[agent]

    def bash(self, command, agent="agent", timeout=COMMAND_TIMEOUT):
        """Run a command on a throwaway overlay of the container; returns its output and, if it tried to change files
        in the repository, a note that those changes were discarded."""
        home, tmp = self._home(agent)
        upper = self.container.dir / f"overlay-{uuid.uuid4().hex[:8]}"
        upper.mkdir()
        cmd = [APPTAINER, "-s", "exec", "--containall", "--overlay", str(upper), "--bind", f"{tmp}:/tmp",
               "--home", f"{home}:/root", "--pwd", REPO, str(self.container.root), "bash", "-c", command]
        try:
            r = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = f"{r.stdout}\n[exit code {r.returncode}]" if r.returncode else (r.stdout or "[no output]")
        except subprocess.TimeoutExpired:
            out = f"[error] the command timed out after {timeout} s"
        changed = []
        written = upper / "upper" / REPO.lstrip("/")
        if written.exists():
            for f in written.rglob("*"):
                rel = "/" + str(f.relative_to(written))
                if (f.is_file() or f.is_symlink()) and not any(x in rel for x in _SCRATCH):
                    changed.append(rel.lstrip("/"))
        _rmtree(upper)
        if changed:
            shown = ", ".join(sorted(changed)[:6]) + (f" and {len(changed) - 6} more" if len(changed) > 6 else "")
            out += (f"\n[note] bash can't change the repository: what this command wrote there ({shown}) was discarded. "
                    "Use write_file, replace_in_file or delete_file to change files.")
        return out

    def _resolve(self, path):
        rel = os.path.normpath(str(path).strip().lstrip("/").removeprefix("workspace/repo/"))
        if rel.startswith("..") or rel in (".", ""):
            return None, f"[error] {path!r} is not a file in the repository"
        return self.repo / rel, rel

    def write_file(self, path, content):
        target, rel = self._resolve(path)
        if target is None:
            return rel
        with self.lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            target.write_text(content)
        return f"{'Replaced' if existed else 'Created'} {rel} ({len(content.splitlines())} lines)."

    def replace_in_file(self, path, old, new):
        target, rel = self._resolve(path)
        if target is None:
            return rel
        with self.lock:
            if not target.is_file():
                return f"[error] no such file: {rel}"
            text = target.read_text(errors="replace")
            n = text.count(old) if old else 0
            if n != 1:
                return (f"[error] `old` occurs {n} times in {rel}; it must occur exactly once (copy it exactly from the "
                        "file, with enough surrounding lines to be unique). Nothing was changed.")
            target.write_text(text.replace(old, new, 1))
        return f"Replaced 1 occurrence in {rel}."

    def delete_file(self, path):
        target, rel = self._resolve(path)
        if target is None:
            return rel
        with self.lock:
            if not target.is_file():
                return f"[error] no such file: {rel}"
            target.unlink()
        return f"Deleted {rel}."

    def patch(self):
        """Everything changed in the repository, new files included, as a diff against the base commit."""
        out = self.bash("git add -A >/dev/null 2>&1; git diff --cached --binary HEAD", agent="_scorer")
        return out if not out.startswith("[error]") and "[exit code" not in out else ""

    def remove(self):
        self.container.remove()


def score_features(task, patch, timeout=COMMAND_TIMEOUT):
    """Each feature's hidden tests on the patch, in a fresh container (CooperBench's test_solo, for any number of
    features). The task counts as solved only if every feature's tests pass."""
    task_dir = DATASET / task["repo"] / f"task{task['task_id']}"
    patch = _filter_test_files(patch or "")
    sb = get_backend("apptainer").create_sandbox(get_image_name(task["repo"], task["task_id"]), timeout)
    results = {}
    try:
        base = sb.exec("bash", "-c", "cd /workspace/repo && git rev-parse HEAD").stdout_read().strip()
        _write_patch(sb, "agent.patch", patch)
        for f in task["features"]:
            _write_patch(sb, f"tests{f}.patch", (task_dir / f"feature{f}" / "tests.patch").read_text())
            results[f] = _run_tests(sb, f"tests{f}.patch", "agent.patch", base)
    finally:
        sb.terminate()
    return {"correct": all(r["passed"] for r in results.values()),
            "features_passed": sum(r["passed"] for r in results.values()), "features": len(results),
            **{f"feature{i + 1}_passed": bool(results[f]["passed"]) for i, f in enumerate(task["features"])},
            "patch_lines": len(patch.splitlines()), "eval_error": None,
            "_test_output": {f"feature{f}": r["output"] for f, r in results.items()}}


def edit_tools(workspace, agent="agent", gate=None):
    """The edit tools, each passing through gate(agent, tool, path, apply) when given (the write-ahead log)."""
    def wrap(name, fn):
        def call(path, **kw):
            if gate is None:
                return fn(path, **kw)
            return gate(agent, name, path, lambda: fn(path, **kw))
        return call
    s = lambda n, d, p: _schema(n, d, p)  # noqa: E731
    return {
        "write_file": (wrap("write_file", lambda path, content: workspace.write_file(path, content)), s(
            "write_file", "Create a file, or replace its whole content.",
            {"path": {"type": "string", "description": "Path relative to /workspace/repo"},
             "content": {"type": "string", "description": "The file's full new content"}})),
        "replace_in_file": (wrap("replace_in_file", lambda path, old, new: workspace.replace_in_file(path, old, new)), s(
            "replace_in_file", "Replace one exact occurrence of `old` in a file with `new`.",
            {"path": {"type": "string", "description": "Path relative to /workspace/repo"},
             "old": {"type": "string", "description": "Exact text to replace; must occur exactly once"},
             "new": {"type": "string", "description": "The replacement text"}})),
        "delete_file": (wrap("delete_file", lambda path: workspace.delete_file(path)), s(
            "delete_file", "Delete a file.", {"path": {"type": "string", "description": "Path relative to /workspace/repo"}})),
    }


class EditSession:
    """The single agent with read-only bash and the edit tools (agent.py --edit-tools): Session's interface."""

    def __init__(self, task):
        self.task, self.reminded = task, False
        self.ws = Workspace(task)
        self.tools = {"execute_bash": (lambda command: self.ws.bash(command), _schema(
            "execute_bash", "Run a bash command in /workspace/repo (read-only: changes to the repository are discarded) "
            "and return its output and exit code.", {"command": {"type": "string", "description": "The bash command to run"}})),
            **edit_tools(self.ws)}

    task_prompt = Session.task_prompt

    def score(self):
        r = score_features(self.task, patch := self.ws.patch())
        return {**r, "reminded": self.reminded, "_patch": patch}

    def close(self):
        self.ws.remove()
