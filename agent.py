"""Single-agent (SAS) baseline: one LLM tool-calling loop against a vLLM server.

This is the baseline from "Towards a Science of Scaling Agent Systems"
(arXiv 2512.08296). The multi-agent topologies will reuse the Agent class.

Usage (on a compute node, with vLLM serving on localhost:8000):
    python agent.py                      # run every task in tasks.jsonl
    python agent.py --only primes        # run one task
    python agent.py --reasoning-effort low --quiet
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
MAX_TOOL_OUTPUT = 4000
PYTHON_TIMEOUT = 30

SYSTEM_PROMPT = """You are a careful problem solver with access to tools.
- Use run_python for any calculation, counting, or data processing. Do not do arithmetic in your head.
- Use read_file to look at files. Paths are relative to the project folder.
- When you are done, end your reply with a single line:
FINAL ANSWER: <answer>
Put only the answer on that line (a number or a short phrase), with no units or explanation."""


# ---- tools ------------------------------------------------------------------

def _print_last_expression(code: str) -> str:
    """Print a trailing bare expression's value, like a notebook cell (gpt-oss expects this)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # let Python report it
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body.pop().value
    if isinstance(last, ast.Call) and getattr(last.func, "id", None) == "print":
        return code
    return ast.unparse(tree) + f"\n_ = ({ast.unparse(last)})\nif _ is not None:\n    print(repr(_))\n"


def run_python(code: str) -> str:
    # Runs model-written code as you, in the project folder. Fine for a sandbox
    # account on a compute node; don't point this at anything you care about.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_print_last_expression(code))
        path = Path(f.name)
    try:
        p = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True,
                           text=True, timeout=PYTHON_TIMEOUT)
        out = p.stdout
        if p.stderr:
            out += f"\n[stderr]\n{p.stderr}"
        if p.returncode:
            out += f"\n[exit code {p.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[error] timed out after {PYTHON_TIMEOUT}s"
    finally:
        path.unlink(missing_ok=True)
    return out.strip() or "[no output - use print() to see results]"


def read_file(path: str) -> str:
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        return "[error] path is outside the project folder"
    if not target.is_file():
        return f"[error] no such file: {path}"
    return target.read_text(errors="replace")


def _schema(name, description, params):
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": params, "required": list(params)},
    }}


TOOLS = {
    "run_python": (run_python, _schema(
        "run_python",
        "Run a complete Python 3 script and return its stdout and stderr. "
        "Print anything you want to see. The working directory is the project folder.",
        {"code": {"type": "string", "description": "The full Python script to run"}})),
    "read_file": (read_file, _schema(
        "read_file",
        "Return the text contents of a file in the project folder.",
        {"path": {"type": "string", "description": "Path relative to the project folder, e.g. data/sales.csv"}})),
}


def call_tool(name: str, arguments: str) -> str:
    if name not in TOOLS:
        return f"[error] unknown tool: {name}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        # gpt-oss was trained with a built-in python tool that takes raw code, so it
        # often sends the script itself instead of {"code": ...}. For single-argument
        # tools, treat the raw text as that argument rather than burning a turn.
        required = TOOLS[name][1]["function"]["parameters"]["required"]
        if len(required) != 1:
            return "[error] arguments must be a JSON object"
        args = {required[0]: arguments}
    try:
        out = TOOLS[name][0](**args)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n[truncated {len(out) - MAX_TOOL_OUTPUT} chars]"
    return out


# ---- agent ------------------------------------------------------------------

class Agent:
    """One reasoning loop: ask the model, run any tools it calls, repeat until it answers."""

    def __init__(self, client, model, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
                 max_turns=12, reasoning_effort="medium", verbose=True):
        self.client = client
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose

    def log(self, text):
        if self.verbose:
            print(text, flush=True)

    def run(self, task: str) -> dict:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}]
        stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        schemas = [schema for _, schema in self.tools.values()]

        for _ in range(self.max_turns):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=schemas,
                reasoning_effort=self.reasoning_effort)
            stats["turns"] += 1
            if resp.usage:
                stats["prompt_tokens"] += resp.usage.prompt_tokens
                stats["completion_tokens"] += resp.usage.completion_tokens

            msg = resp.choices[0].message
            entry = {"role": "assistant", "content": msg.content}
            # gpt-oss needs its earlier reasoning passed back between tool calls.
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if reasoning:
                entry["reasoning"] = reasoning

            if not msg.tool_calls:
                messages.append(entry)
                self.log(f"  [answer] {(msg.content or '').strip()[-300:]}")
                return {"final": msg.content or "", "stop": "final", "messages": messages, **stats}

            entry["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls]
            messages.append(entry)
            for tc in msg.tool_calls:
                stats["tool_calls"] += 1
                self.log(f"  [{tc.function.name}] {tc.function.arguments[:200]}")
                out = call_tool(tc.function.name, tc.function.arguments)
                if out.startswith("[error]") or "[exit code" in out:
                    stats["tool_errors"] += 1
                self.log(f"    -> {out[:200]}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tc.function.name, "content": out})

        return {"final": "", "stop": "max_turns", "messages": messages, **stats}


# ---- scoring ----------------------------------------------------------------

def extract_answer(text: str):
    found = re.findall(r"FINAL ANSWER:?\**\s*(.+)", text or "", re.IGNORECASE)
    return found[-1].strip() if found else None


def _normalize(s: str) -> str:
    return s.strip().strip("*`").strip().rstrip(".").replace(",", "").replace("$", "").strip().lower()


def is_correct(answer, expected) -> bool:
    if answer is None:
        return False
    a, e = _normalize(answer), _normalize(str(expected))
    try:
        return abs(float(a) - float(e)) <= 1e-6 * max(1.0, abs(float(e)))
    except ValueError:
        return a == e


# ---- main -------------------------------------------------------------------

def git_commit():
    """Short commit hash of the code that produced a run, with -dirty if there are uncommitted edits."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        return (commit + ("-dirty" if git("status", "--porcelain") else "")) or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", help="vLLM's --served-model-name (default: whatever the server is running)")
    ap.add_argument("--tasks", default=str(ROOT / "tasks.jsonl"))
    ap.add_argument("--only", help="run just the task with this id")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--quiet", action="store_true", help="hide per-step tool output")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    if args.only:
        tasks = [t for t in tasks if t["id"] == args.only]
    if not tasks:
        sys.exit("no tasks to run")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    if not args.model:
        args.model = client.models.list().data[0].id

    run_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-sas-{args.model}"
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"topology": "single", "git_commit": git_commit(), **vars(args)}, indent=2))

    agent = Agent(client, args.model, max_turns=args.max_turns,
                  reasoning_effort=args.reasoning_effort, verbose=not args.quiet)

    rows = []
    for t in tasks:
        print(f"\n=== {t['id']} ===", flush=True)
        start = time.time()
        try:
            r = agent.run(t["question"])
        except Exception as e:
            r = {"final": "", "stop": f"error: {type(e).__name__}: {e}", "messages": [],
                 "turns": 0, "tool_calls": 0, "tool_errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        answer = extract_answer(r["final"])
        row = {"id": t["id"], "correct": is_correct(answer, t["answer"]), "answer": answer,
               "expected": t["answer"], "stop": r["stop"], "turns": r["turns"],
               "tool_calls": r["tool_calls"], "tool_errors": r["tool_errors"],
               "prompt_tokens": r["prompt_tokens"],
               "completion_tokens": r["completion_tokens"], "seconds": round(time.time() - start, 1)}
        rows.append(row)
        (run_dir / "traces" / f"{t['id']}.json").write_text(json.dumps(r["messages"], indent=2, default=str))
        with open(run_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"{mark}  answer={answer!r} expected={t['answer']!r}  turns={row['turns']} "
              f"tools={row['tool_calls']} tool_errors={row['tool_errors']} tokens={row['prompt_tokens'] + row['completion_tokens']} "
              f"{row['seconds']}s  stop={row['stop']}", flush=True)

    n = len(rows)
    summary = {
        "tasks": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "mean_turns": round(sum(r["turns"] for r in rows) / n, 2),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "total_tool_errors": sum(r["tool_errors"] for r in rows),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}\nsaved to {run_dir}")


if __name__ == "__main__":
    main()
