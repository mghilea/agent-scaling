"""Offline test of the Claude backend: agent.py --model claude-... on a real AppWorld task, with a scripted stand-in
for the Anthropic client (nothing is sent, nothing is spent). Checks the request shape, the tool loop, the cost
accounting, the results row, and that --max-dollars stops the run.

    $APPWORLD_PYTHON scripts/test_claude_agent.py
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APPWORLD_ROOT", f"/scratch/{os.environ['USER']}/appworld")
import claude_agent  # noqa: E402
import agent  # noqa: E402


class Block(dict):
    def __getattr__(self, k):
        return self[k]

    def model_dump(self, **_):
        return dict(self)


class Usage:
    def __init__(self):
        self.input_tokens, self.cache_creation_input_tokens, self.cache_read_input_tokens, self.output_tokens = 3000, 2000, 10000, 500
        self.iterations = None


SCRIPT = [  # what the stand-in "model" does, one reply per call
    [Block(type="text", text="Let me look."), Block(type="tool_use", id="t1", name="execute_python",
                                                     input={"code": "print(len(apis.api_docs.show_app_descriptions()))"})],
    [Block(type="tool_use", id="t2", name="execute_python", input={"code": "apis.supervisor.complete_task()"})],
    [Block(type="text", text="Done.")],
]
calls = []


class Stream:
    def __init__(self, params):
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        i = sum(1 for p in calls if p is not None) - 1
        content = SCRIPT[min(i, len(SCRIPT) - 1)]
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return Block(content=content, stop_reason=stop, stop_details=None, usage=Usage(), model=self.params["model"])


class Messages:
    def stream(self, **params):
        calls.append(params)
        return Stream(params)


class Client:
    messages = Messages()

    class beta:
        messages = Messages()


claude_agent.make_client = lambda: Client()
task = sorted(json.load(open(ROOT / "analysis" / "aw_ladder.json"))["by_size"] and
              [l.split(",")[0] for l in os.popen(f"python3 {ROOT}/analysis/aw_big_tasks.py")])[0]

# 1) one task, generous budget
sys.argv = ["agent.py", "--benchmark", "appworld", "--dataset", "dev", "--model", "claude-opus-5-5", "--max-dollars", "5",
            "--only", task, "--tag", "offlinetest", "--quiet"]
agent.main()
run = sorted(glob.glob(str(ROOT / "runs" / "*-sas-appworld-dev-claude-opus-5-5-offlinetest")))[-1]
row = json.loads(open(f"{run}/results.jsonl").readline())
p = calls[0]
assert p["model"] == "claude-opus-5-5" and p["betas"] == [claude_agent.FALLBACK_BETA] and p["fallbacks"] == "default"
assert p["output_config"] == {"effort": "high"} and p["cache_control"] == {"type": "ephemeral"}
assert p["tools"][0]["name"] == "execute_python" and "input_schema" in p["tools"][0] and isinstance(p["system"], str)
assert "thinking" not in p and "temperature" not in p
assert row["turns"] == 3 and row["tool_calls"] == 2 and row["completed"] is True, row
per_call = (3000 * 4 + 2000 * 4 * 1.25 + 10000 * 0.20 + 500 * 20) / 1e6
assert abs(row["dollars"] - 3 * per_call) < 1e-6, (row["dollars"], 3 * per_call)
assert row["prompt_tokens"] == 3 * 15000 and row["completion_tokens"] == 1500 and row["fallbacks"] == 0
tr = json.load(open(f"{run}/traces/{task}.json"))
assert tr[0]["role"] == "system" and any(isinstance(m["content"], list) for m in tr[1:])
summary = json.load(open(f"{run}/summary.json"))
assert abs(summary["total_dollars"] - 3 * per_call) < 1e-4
print(f"one task: OK ({row['turns']} calls, ${row['dollars']:.3f}, completed={row['completed']}, tests {row['tests_passed']}/{row['tests']})")

# 2) the budget: a limit below one task's cost stops the run before the second task
calls.clear()
two = ",".join(os.popen(f"python3 {ROOT}/analysis/aw_big_tasks.py").read().strip().split(",")[:2])
sys.argv = ["agent.py", "--benchmark", "appworld", "--dataset", "dev", "--model", "claude-opus-5-5", "--max-dollars", "0.05",
            "--only", two, "--tag", "offlinetest2", "--quiet"]
agent.main()
run2 = sorted(glob.glob(str(ROOT / "runs" / "*-sas-appworld-dev-claude-opus-5-5-offlinetest2")))[-1]
rows = [json.loads(l) for l in open(f"{run2}/results.jsonl")]
assert len(rows) == 1 and rows[0]["stop"].startswith("error: BudgetExceeded"), rows
assert len(calls) == 2, len(calls)  # $0.034 per call: the third call is refused, and the second task never starts
print(f"budget: OK (stopped after {len(calls)} calls, ${rows[0]['dollars']:.3f}; second task not started)")
for r in (run, run2):
    os.system(f"rm -rf '{r}'")
print("all Claude backend tests passed")
