"""Offline test of CooperBench teams (cooperbench_team.py) with scripted agents, no model.

    $COOPERBENCH_PYTHON scripts/test_cooperbench_team.py [L6|L0]

On the core pair dottxt_ai_outlines 1655 f1+f3, with the smoke run's passing patch (one file) as the work:
- L6 (full catch-up and open-claim conflicts; named L4 before 2026-11-18): agent_1 claims the file and edits it three times under one claim; agent_2's edit without a claim, its catch-up and
  its claim on the same file are refused until agent_1 stops; then it claims the file and edits it. Both features pass.
- L0: no log; both agents edit the file, and the edit metrics count the crossing.
Both levels also check that bash can't change the repository, and that each agent is told only its own feature.
"""
import glob
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cooperbench_team as ct  # noqa: E402

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L6"
TASK = {"id": "dottxt_ai_outlines_task-1655-f1-f3", "repo": "dottxt_ai_outlines_task", "task_id": 1655,
        "features": [1, 3]}
F = "outlines/types/__init__.py"
GOOD = open(sorted(glob.glob(str(ROOT / "runs" / "*cooperbench-core-gpt-oss-20b-smoke2" / "patches" / "*.patch")))[-1]).read()
assert GOOD.count("diff --git") == 1 and F in GOOD
held, tried = threading.Event(), threading.Event()


class Fake:
    """Agent's interface (run, stats, limit_hits, messages) running a script instead of a model."""

    def __init__(self, name, tools, system_prompt, **_):
        self.name, self.tools, self.system_prompt = name, tools, system_prompt
        self.stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.limit_hits, self.messages, self.max_turns = 0, [], 100
        self.sent = []

    def send(self, text):  # a reminder from the team: recorded, and the agent stops again
        self.sent.append(text)
        print(f"  {self.name} reminded: {text[:160]!r}", flush=True)
        return {"final": "done", "stop": "final", "messages": self.messages, **self.stats}

    def call(self, tool, **kw):
        self.stats["tool_calls"] += 1
        out = self.tools[tool][0](**kw)
        self.messages.append({"tool": tool, "args": {k: str(v)[:80] for k, v in kw.items()}, "out": out[:300]})
        print(f"  {self.name} {tool}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in kw.items())}) -> {out[:160]!r}",
              flush=True)
        return out

    def until(self, tool, ok, tries=60, **kw):
        """Call until ok(output): the way an agent retries after a refusal (reading what it was shown)."""
        for _ in range(tries):
            out = self.call(tool, **kw)
            if ok(out):
                return out
            time.sleep(0.5)
        raise AssertionError(f"{self.name}: {tool} never went through: {out}")

    def run(self, prompt):
        mine = {"agent_1": 1, "agent_2": 3}[self.name]
        assert f"## Feature {mine}" in prompt and prompt.count("## Feature") == 1, "an agent sees only its feature"
        assert "write-ahead log" in self.system_prompt.lower() if LEVEL != "L0" else "wal_intend" not in self.tools
        globals()[f"{self.name}_{'wal' if LEVEL != 'L0' else 'plain'}"](self)
        return {"final": "done", "stop": "final", "messages": self.messages, **self.stats}


def good_content(a):
    out = a.call("execute_bash", command=f"cat > /tmp/p.patch <<'PATCH_END'\n{GOOD}PATCH_END\n"
                                         f"git apply /tmp/p.patch >/dev/null 2>&1 && cat {F}")
    assert "[note] bash can't change the repository" in out and F in out, "bash's write must be discarded"
    return out.split("\n[note]")[0]


def edits_three_times(a, new):
    first = new.splitlines()[0]
    assert new.count(first) == 1
    assert a.call("write_file", path=F, content=new).startswith(F + ": replaced the whole file")
    assert "replaced line 1" in a.call("replace_in_file", path=F, old=first, new=first + "  # tmp")
    assert "replaced line 1" in a.call("replace_in_file", path=F, old=first + "  # tmp", new=first)
    return first


def agent_1_wal(a):
    new = good_content(a)
    assert a.call("wal_read") == "The WAL is empty."
    assert a.call("wal_intend", plan="feature 1", files=[F, "./" + F]).startswith("Posted as intent #1")
    held.set()
    assert tried.wait(60)
    edits_three_times(a, new)                     # one claim covers every edit of the file
    print("  log:", a.call("wal_read", all=True).replace("\n", " || ")[:400])
    assert "doesn't claim" in a.call("delete_file", path="README.md")   # refused and never made: reminded at the end


def agent_2_wal(a):
    assert held.wait(60)
    assert f"doesn't claim the file {F}" in a.call("replace_in_file", path=F, old="x", new="y")       # L2: no claim
    out = a.call("wal_intend", plan="feature 3", files=[F])
    assert "hadn't read" in out, out                                                       # L3: catch up first
    out = a.call("wal_intend", plan="feature 3", files=[F])
    assert "conflicts with agent_1's open claim" in out and "sleep 60" in out, out         # L6: agent_1 holds it
    tried.set()
    a.until("wal_intend", lambda o: o.startswith("Posted"), plan="feature 3", files=[F])   # free once agent_1 stops
    first = a.call("execute_bash", command=f"head -1 {F}").strip()
    a.until("replace_in_file", lambda o: not o.startswith("[WAL]"), path=F, old=first + "\n", new=first + "\n")


def agent_1_plain(a):
    new = good_content(a)
    edits_three_times(a, new)
    held.set()


def agent_2_plain(a):
    assert held.wait(60)
    first = a.call("execute_bash", command=f"head -1 {F}").strip()
    assert "replaced line 1" in a.call("replace_in_file", path=F, old=first + "\n", new=first + "\n")
    assert "occurs 0 times" in a.call("replace_in_file", path=F, old="zzz-not-there", new="y")


run_dir = Path(tempfile.mkdtemp(prefix="cbteam-"))
r = ct.run(TASK, lambda **kw: Fake(**kw), run_dir, level=None if LEVEL == "L0" else LEVEL)
v, c = r["verdict"], r["comm"]
print("verdict:", {k: v for k, v in v.items() if not k.startswith("_")})
print("comm:", c)
assert v["correct"] and v["features_passed"] == 2, "both features pass on the team's final tree"
assert [l for l in v["_patch"].splitlines() if l.startswith("diff --git")] == [f"diff --git a/{F} b/{F}"]
assert c["shared_files"] == 1 and c["crossed_edits"] == 1 and c["overwrites"] == 0 and c["bash_writes_discarded"] == 1
if LEVEL == "L0":
    assert c["edits"] == 4 and c["failed_edits"] == 1 and "wal_level" not in c
else:
    assert c["edits"] == 4 and c["wal_refused_intent"] == 2 and c["wal_refused_read_before_post"] >= 1  # 2: incl. the delete
    assert c["wal_refused_no_conflict"] >= 1 and c["wal_released_stopped"] >= 1 and c["wal_released_expired"] == 0
    entries = r["trace"]["wal_log"]
    assert [e["kind"] for e in entries][:4] == ["intent", "change", "change", "change"]
    assert entries[0]["claims"] == [{"app": "repo", "api": "edit", "target": F, "ids": [F]}], entries[0]
    assert (run_dir / "wal" / f"{TASK['id']}.jsonl").exists()
    # agent_2's refused edit of F was made later (cleared); agent_1's refused delete of README.md never was
    assert c["unmade_reminded"] == 1 and c["unmade_at_end"] == 1, c
    reminders = [e for e in r["trace"]["events"] if e["kind"] == "unmade_reminder"]
    assert reminders[0]["agent"] == "agent_1" and "README.md" in reminders[0]["calls"][0], reminders
print(f"cooperbench team test passed ({LEVEL})")
