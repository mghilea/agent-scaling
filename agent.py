"""Single-agent (SAS) baseline: one LLM tool-calling loop against a vLLM server.

This is the baseline from "Towards a Science of Scaling Agent Systems"
(arXiv 2512.08296). The multi-agent topologies will reuse the Agent class.

Usage (on a compute node, with vLLM serving on localhost:8000):
    python agent.py                                  # run every task in tasks.jsonl
    python agent.py --only primes --show-reasoning   # one task, including the model's thinking
    python agent.py --ask "How many days until 2027?" # your own question (logged, not scored)
    python agent.py --reasoning-effort low --quiet
    python agent.py --benchmark workbench --limit 10  # 10 random WorkBench tasks (see workbench.py)
    python agent.py --benchmark workbench --limit 10 --topology centralized  # a multi-agent system (mas.py)
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from openai import NOT_GIVEN, OpenAI

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


def call_tool(name: str, arguments: str, tools=TOOLS) -> str:
    if name not in tools:
        return f"[error] unknown tool: {name}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        # gpt-oss was trained with a built-in python tool that takes raw code, so it
        # often sends the script itself instead of {"code": ...}. For single-argument
        # tools, treat the raw text as that argument rather than burning a turn.
        required = tools[name][1]["function"]["parameters"].get("required", [])
        if len(required) != 1:
            return "[error] arguments must be a JSON object"
        args = {required[0]: arguments}
    try:
        out = tools[name][0](**args)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    if len(out) > MAX_TOOL_OUTPUT:
        out = out[:MAX_TOOL_OUTPUT] + f"\n[truncated {len(out) - MAX_TOOL_OUTPUT} chars]"
    return out


# ---- display ----------------------------------------------------------------

_COLOR = sys.stdout.isatty()
BOLD, DIM, GREEN, YELLOW, CYAN = "1", "2", "32", "33", "36"


def _c(color, text):
    return f"\033[{color}m{text}\033[0m" if _COLOR else text


def _block(label, text, color, limit=1500):
    """Render text as a labelled, indented block: '  run_python │ line 1\n             │ line 2'."""
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more chars]"
    lines = text.splitlines() or [""]
    head, pad = _c(color, f"  {label:>10} │ "), _c(color, f"  {'':>10} │ ")
    return head + lines[0] + "".join(f"\n{pad}{line}" for line in lines[1:])


def _show_args(arguments: str) -> str:
    """Show a tool call's arguments readably: the code itself for run_python, not escaped JSON."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    if isinstance(args, dict) and len(args) == 1:
        return str(next(iter(args.values())))
    return json.dumps(args, indent=2)


# ---- agent ------------------------------------------------------------------

def new_stats():
    return {"turns": 0, "tool_calls": 0, "tool_errors": 0, "prompt_tokens": 0, "completion_tokens": 0}


class Agent:
    """One reasoning loop: ask the model, run any tools it calls, repeat until it answers.

    The conversation stays in self.messages, so a multi-agent system (mas.py) can send the same
    agent another message later and it carries on from where it stopped.
    """

    def __init__(self, client, model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, max_turns=12,
                 reasoning_effort="medium", verbose=True, show_reasoning=False, name=""):
        self.client = client
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.name = name  # shown in logs when several agents run at once
        self.messages = [{"role": "system", "content": system_prompt}]
        self.stats = new_stats()
        self._stats_lock = threading.Lock()  # an orchestrator's calls can run in parallel threads
        # Optional: a function returning text that arrived while the agent was working (messages from
        # teammates), or None. It is checked after each round of tool calls and added as a user message.
        self.inbox = None

    def log(self, text):
        if self.verbose:
            print(text, flush=True)

    def run(self, task: str) -> dict:
        """Start a fresh conversation on task."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.stats = new_stats()
        return self.send(task)

    def send(self, text: str) -> dict:
        """Add a user message and loop, running tools, until the model answers without calling one."""
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_turns):
            msg, entry = self._call(self.messages, use_tools=True)
            if not msg.tool_calls:
                self.messages.append(entry)
                self.log(_block("answer", msg.content, GREEN))
                return {"final": msg.content or "", "stop": "final", "messages": self.messages, **self.stats}

            if msg.content:
                self.log(_block("says", msg.content, DIM))
            for tc in msg.tool_calls:
                # vLLM's gpt-oss parser sometimes leaves format tokens on the name, like
                # "email_search_emails<|channel|>commentary". Keep just the tool name.
                tc.function.name = tc.function.name.split("<|")[0].strip()
            entry["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls]
            self.messages.append(entry)
            for tc in msg.tool_calls:
                self.log(_block(tc.function.name, _show_args(tc.function.arguments), CYAN, limit=3000))
                out = call_tool(tc.function.name, tc.function.arguments, self.tools)
                failed = out.startswith("[error]") or "[exit code" in out
                with self._stats_lock:
                    self.stats["tool_calls"] += 1
                    self.stats["tool_errors"] += failed
                self.log(_block("output", out, YELLOW if failed else DIM, limit=800))
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "name": tc.function.name, "content": out})
            arrived = self.inbox() if self.inbox else None
            if arrived:
                self.log(_block("arrived", arrived, DIM, limit=800))
                self.messages.append({"role": "user", "content": arrived})

        self.log(_c(YELLOW, f"  {self.name or 'agent'} stopped: hit --max-turns ({self.max_turns}) "
                            "without a final answer"))
        return {"final": "", "stop": "max_turns", "messages": self.messages, **self.stats}

    def reply(self, text: str) -> str:
        """Add a user message and get one plain-text answer, with no tools (e.g. a summary)."""
        self.messages.append({"role": "user", "content": text})
        msg, entry = self._call(self.messages, use_tools=False)
        self.messages.append(entry)
        return msg.content or ""

    def ask(self, messages: list) -> tuple[str, str]:
        """One plain-text call on a standalone prompt, outside this agent's conversation.

        Returns the reply and the model's reasoning (empty if it gave none)."""
        msg, entry = self._call(messages, use_tools=False)
        return msg.content or "", entry.get("reasoning", "")

    def _call(self, messages, use_tools):
        started = time.time()
        schemas = [schema for _, schema in self.tools.values()] if use_tools else []
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=schemas or NOT_GIVEN,
            reasoning_effort=self.reasoning_effort)
        used = ""
        with self._stats_lock:
            self.stats["turns"] += 1
            turn = self.stats["turns"]
            if resp.usage:
                self.stats["prompt_tokens"] += resp.usage.prompt_tokens
                self.stats["completion_tokens"] += resp.usage.completion_tokens
                used = f"{resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion tokens, "
        who = f"{self.name} " if self.name else ""
        self.log(_c(BOLD, f"── {who}turn {turn} ──") + _c(DIM, f" {used}{time.time() - started:.1f}s"))

        msg = resp.choices[0].message
        entry = {"role": "assistant", "content": msg.content}
        # gpt-oss needs its earlier reasoning passed back between tool calls.
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning:
            entry["reasoning"] = reasoning
            if self.show_reasoning:
                self.log(_c(DIM, _block("thinking", reasoning, DIM)))
        return msg, entry


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
    ap.add_argument("--benchmark", default="toy", choices=["toy", "workbench"],
                    help="toy: the task file (--tasks); workbench: WorkBench tasks, scored by its evaluator")
    ap.add_argument("--tasks", default=str(ROOT / "tasks.jsonl"))
    ap.add_argument("--only", help="run just the task with this id")
    ap.add_argument("--limit", type=int, help="workbench: run a random sample of this many tasks")
    ap.add_argument("--seed", type=int, default=0, help="workbench: which random sample --limit takes")
    ap.add_argument("--domain", action="append", help="workbench: only this domain (repeatable), e.g. email")
    ap.add_argument("--topology", default="single",
                    choices=["single", "independent", "centralized", "decentralized", "hybrid", "agent-driven"],
                    help="one agent, one of the paper's multi-agent systems (mas.py), or an agent-driven team "
                         "with no rounds and a shared workspace (agent_driven.py); workbench only")
    ap.add_argument("--comm", default="all", choices=["all", "text", "a2a", "shared", "none"],
                    help="agent-driven: channels the agents may use (all = free choice)")
    ap.add_argument("--coordinate", action="store_true",
                    help="agent-driven: ask agents to agree on who does what before changing anything")
    ap.add_argument("--agents", type=int, default=3, help="multi-agent: number of worker agents (paper: 3)")
    ap.add_argument("--rounds", type=int,
                    help="multi-agent: max rounds (default as in the paper: 5 centralized/hybrid, 3 decentralized)")
    ap.add_argument("--max-turns", type=int,
                    help="turns per agent (per round, for multi-agent): default 12, or 20 for workbench")
    ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--ask", help="ask your own question instead of running the task file")
    ap.add_argument("--show-reasoning", action="store_true", help="also print the model's reasoning each turn")
    ap.add_argument("--quiet", action="store_true", help="only print one result line per task")
    args = ap.parse_args()

    wb = None
    if args.ask:
        tasks = [{"id": "ask", "question": args.ask}]
    elif args.benchmark == "workbench":
        import workbench as wb
        tasks = wb.load_tasks(args.domain, args.limit, args.seed)
    else:
        tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    if args.only:
        tasks = [t for t in tasks if t["id"] == args.only]
    if not tasks:
        sys.exit("no tasks to run")
    multi = args.topology != "single"
    if multi and not wb:
        sys.exit("the multi-agent topologies run on WorkBench: add --benchmark workbench")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    if not args.model:
        args.model = client.models.list().data[0].id

    args.max_turns = args.max_turns or (20 if wb else 12)
    driven = args.topology == "agent-driven"
    if multi and not driven:
        import mas
        args.rounds = args.rounds or mas.DEFAULT_ROUNDS[args.topology]
    if driven:
        import agent_driven
    topo_name = f"agent-driven-{args.comm}{'-coord' if args.coordinate else ''}" if driven else args.topology
    kind = "ask" if args.ask else ("sas" if not multi else topo_name) + ("-workbench" if wb else "")
    run_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{args.model}"
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"git_commit": git_commit(), **vars(args)}, indent=2))

    def make_agent(**kwargs):
        return Agent(client, args.model, max_turns=args.max_turns, reasoning_effort=args.reasoning_effort,
                     verbose=not args.quiet, show_reasoning=args.show_reasoning, **kwargs)

    def log_message(label, text):  # messages between agents, in multi-agent runs
        if not args.quiet:
            print(_block(label, text, DIM, limit=600), flush=True)

    rows = []
    for t in tasks:
        print(_c(BOLD, f"\n=== {t['id']} ===") + f"\n{t['question']}", flush=True)
        start = time.time()
        actions = []
        try:
            if driven:
                r = agent_driven.run(t["question"], make_agent, args.agents, args.comm, log_message, seed=t["id"],
                                     coordinate=args.coordinate)
                actions = r["actions"]
            elif multi:
                r = mas.run(args.topology, t["question"], make_agent, args.agents, args.rounds, log_message)
                actions = r["actions"]
            elif wb:
                sandbox = wb.Sandbox()
                r = make_agent(tools=sandbox.tools, system_prompt=wb.SYSTEM_PROMPT).run(t["question"])
                actions = sandbox.actions
            else:
                r = make_agent().run(t["question"])
        except Exception as e:
            r = {"final": "", "stop": f"error: {type(e).__name__}: {e}", "messages": [], **new_stats()}
        if wb:
            # Scored on what the agents did (their tool calls), not on what they said.
            verdict = wb.score(actions, t["outcome"], error=r["stop"] != "final")
            correct, answer, expected = verdict["correct"], actions, t["outcome"]
        else:
            verdict = {}
            answer, expected = extract_answer(r["final"]), t.get("answer")
            # Tasks without an "answer" (like --ask) are run and logged but not scored.
            correct = is_correct(answer, t["answer"]) if "answer" in t else None
        row = {"id": t["id"], "correct": correct, **verdict, "answer": answer,
               "expected": expected, "stop": r["stop"], "turns": r["turns"],
               "tool_calls": r["tool_calls"], "tool_errors": r["tool_errors"],
               "prompt_tokens": r["prompt_tokens"],
               "completion_tokens": r["completion_tokens"], "agent_messages": r.get("agent_messages", 0),
               **{k: r[k] for k in ("agents", "rounds", "comm", "duplicate_changes") if k in r},
               "seconds": round(time.time() - start, 1)}
        rows.append(row)
        # Multi-agent runs trace every agent, not one conversation.
        trace = r["trace"] if "trace" in r else r["messages"]
        (run_dir / "traces" / f"{t['id']}.json").write_text(json.dumps(trace, indent=2, default=str))
        with open(run_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        mark = {True: _c(GREEN, "PASS"), False: _c(YELLOW, "FAIL"), None: "DONE"}[correct]
        if wb:
            shown = f"actions={answer}  expected={expected}  side_effects={verdict['side_effects']}"
        else:
            shown = f"answer={answer!r}" + (f" expected={expected!r}" if "answer" in t else "")
        extra = f"agents={row['agents']} rounds={row['rounds']} messages={row['agent_messages']} " if multi else ""
        print(f"{mark}  {shown}  {extra}turns={row['turns']} "
              f"tools={row['tool_calls']} tool_errors={row['tool_errors']} tokens={row['prompt_tokens'] + row['completion_tokens']} "
              f"{row['seconds']}s  stop={row['stop']}", flush=True)

    n = len(rows)
    scored = [r for r in rows if r["correct"] is not None]
    summary = {
        "tasks": n,
        "accuracy": round(sum(r["correct"] for r in scored) / len(scored), 3) if scored else None,
        **({"side_effect_rate": round(sum(r["side_effects"] for r in rows) / n, 3)} if wb else {}),
        "mean_turns": round(sum(r["turns"] for r in rows) / n, 2),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "total_tool_errors": sum(r["tool_errors"] for r in rows),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        "mean_agent_messages": round(sum(r["agent_messages"] for r in rows) / n, 2),
        # The paper's message density: messages between agents per model call (turn).
        "message_density": round(sum(r["agent_messages"] for r in rows) / max(1, sum(r["turns"] for r in rows)), 3),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}\nsaved to {run_dir}")


if __name__ == "__main__":
    main()
