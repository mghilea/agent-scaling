# Run with AppWorld's Python on a node with AppWorld set up: $APPWORLD_PYTHON scripts/test_completion_all.py
"""End-to-end: the "all" completion rule (--team-complete all) on a real AppWorld task with scripted agents (no model).

With the log (L4) and without it: agent_1 likes a song and calls complete_task (a vote: the task goes on, agent_1 is told
to stop, its remaining claim is released); agent_2 then likes another song (allowed: the task isn't over) and votes with
a different answer; agent_3 stops without voting, is reminded once, and stops again. The task completes at the end, once,
with the latest of the tied answers (agent_2's)."""
import os, sys, threading, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("APPWORLD_ROOT", f"/scratch/{os.environ['USER']}/appworld")
import appworld_team

LOGIN = ('pw = {a["account_name"]: a["password"] for a in apis.supervisor.show_account_passwords()}\n'
         'tok = apis.spotify.login(username=apis.supervisor.show_profile()["email"], password=pw["spotify"])["access_token"]\n')


def scenario(level):
    voted1, voted2 = threading.Event(), threading.Event()
    replies, sends = {}, {}

    class Scripted:
        def __init__(self, name, tools, system_prompt):
            self.name, self.tools, self.prompt = name, tools, system_prompt
            self.stats = dict(turns=0, tool_calls=0, tool_errors=0, prompt_tokens=0, completion_tokens=0)
            self.messages, self.limit_hits, self.max_turns = [], 0, 100

        def call(self, tool, **kw):
            self.stats["turns"] += 1
            self.stats["tool_calls"] += 1
            out = str(self.tools[tool][0](**kw))
            replies.setdefault(self.name, []).append((tool, out[:300]))
            return out

        def send(self, msg):  # the reminder: agent_3 stops again without voting
            sends.setdefault(self.name, []).append(msg)
            return {"final": "done", "stop": "final", "messages": self.messages}

        def run(self, prompt):
            assert "the task ends when all of you have called it" in self.prompt, "agents are told the rule"
            n = self.name
            if n == "agent_1":
                if level:
                    self.call("wal_read")
                    out = self.call("wal_intend", plan="like 54 and 57", claims=[
                        {"app": "spotify", "api": "like_song", "target": "54"},
                        {"app": "spotify", "api": "like_song", "target": "57"}])
                    assert out.startswith("Posted"), out
                out = self.call("execute_python", code=LOGIN + "print(apis.spotify.like_song(song_id=54, access_token=tok))")
                out = self.call("execute_python", code="print(apis.supervisor.complete_task(answer='alpha'))")
                assert "Recorded" in out and "still working" in out, out
                voted1.set()
                out = self.call("execute_python", code="print('after voting')")
                assert "already marked the task complete" in out, out
            elif n == "agent_2":
                assert voted1.wait(60)
                time.sleep(0.3)
                if level:
                    self.call("wal_read")
                    out = self.call("wal_intend", plan="like 55", claims=[{"app": "spotify", "api": "like_song", "target": "55"}])
                    assert out.startswith("Posted"), out
                out = self.call("execute_python", code=LOGIN + "print(apis.spotify.like_song(song_id=55, access_token=tok))")
                assert "Execution failed" not in out and "complete" not in out.lower(), out  # the task goes on
                out = self.call("execute_python", code="print(apis.supervisor.complete_task(answer='Beta'))")
                assert "Recorded" in out, out
                voted2.set()
            else:
                assert voted2.wait(60)
                out = self.call("execute_python", code="print('agent_3 looks around')")
                assert "Execution failed" not in out, out  # still allowed: agent_3 hasn't voted
            return {"final": "done", "stop": "final", "messages": self.messages}

    run_dir = Path(tempfile.mkdtemp()) / f"smoke-completion-all-{level or 'L0'}"
    run_dir.mkdir()
    r = appworld_team.run("57c3486_2", lambda **kw: Scripted(kw["name"], kw["tools"], kw["system_prompt"]), 3, run_dir,
                          level=level, completion="all")
    c, tr = r["comm"], r["trace"]
    for a, rs in sorted(replies.items()):
        for tool, out in rs:
            print(f"  {a:8s} {tool:15s} {out[:140]!r}")
    keys = ["completion_rule", "complete_votes", "answers_distinct", "completed_by", "vote_spread", "writes"]
    if level:
        keys += ["wal_released_completed", "wal_released_stopped", "wal_changes"]
    print(level or "L0", {k: c.get(k) for k in keys}, "votes:", tr["votes"], "reminded:", r["verdict"]["reminded"],
          "completed:", r["verdict"]["completed"])
    assert r["verdict"]["completed"], "the task completes at the end"
    assert c["complete_votes"] == 2 and c["answers_distinct"] == 2 and c["completed_by"] == "agent_2"
    assert sends.get("agent_3") and not sends.get("agent_1") and not sends.get("agent_2"), sends  # only agent_3 reminded
    assert c["writes"] == 3, c["writes"]  # two likes and the team's one complete_task
    calls = [e for e in tr["events"] if e["kind"] == "api_write" and e["api"] == "complete_task"]
    assert len(calls) == 1 and "beta" in calls[0]["call"].lower(), calls
    if level:
        assert c["wal_released_completed"] == 1, c  # agent_1's unused claim on 57
        assert any(e["kind"] == "change" and "complete_task" in e["call"] for e in tr["wal_log"])


def scenario_stuck(level):
    """agent_3 hangs (a model call that never returns): the team ends GRACE seconds after TIME_LIMIT anyway, agent_3
    counts as abandoned, and the two votes decide."""
    limit, grace = appworld_team.TIME_LIMIT, appworld_team.GRACE
    appworld_team.TIME_LIMIT, appworld_team.GRACE = 6, 3
    try:
        class Scripted:
            def __init__(self, name, tools, system_prompt):
                self.name, self.tools = name, tools
                self.stats = dict(turns=0, tool_calls=0, tool_errors=0, prompt_tokens=0, completion_tokens=0)
                self.messages, self.limit_hits, self.max_turns = [], 0, 100

            def send(self, msg):
                return {"final": "done", "stop": "final", "messages": self.messages}

            def run(self, prompt):
                if self.name == "agent_3":
                    time.sleep(60)  # stuck
                else:
                    self.tools["execute_python"][0](code="print(apis.supervisor.complete_task(answer='same'))")
                return {"final": "done", "stop": "final", "messages": self.messages}

        run_dir = Path(tempfile.mkdtemp()) / f"smoke-stuck-{level or 'L0'}"
        run_dir.mkdir()
        t = time.time()
        r = appworld_team.run("57c3486_2", lambda **kw: Scripted(kw["name"], kw["tools"], kw["system_prompt"]), 3,
                              run_dir, level=level, completion="all")
        took, c = time.time() - t, r["comm"]
        print(level or "L0", "stuck agent:", f"team ended after {took:.1f}s,",
              {k: c.get(k) for k in ("agents_abandoned", "complete_votes", "completed_by")}, "stop:", r["stop"],
              "completed:", r["verdict"]["completed"])
        assert took < 30 and c["agents_abandoned"] == 1 and c["complete_votes"] == 2 and r["verdict"]["completed"]
        assert r["stop"] == "time_limit"
    finally:
        appworld_team.TIME_LIMIT, appworld_team.GRACE = limit, grace


def scenario_unmade():
    """L6: agent_2's like of 55 is refused (no claim) and it never makes it: its first complete_task isn't recorded but
    lists the unmade change; its second call is. agent_1's refused like is made later, so it signs off at once."""
    out_vote = {}

    class Scripted:
        def __init__(self, name, tools, system_prompt):
            self.name, self.tools = name, tools
            self.stats = dict(turns=0, tool_calls=0, tool_errors=0, prompt_tokens=0, completion_tokens=0)
            self.messages, self.limit_hits, self.max_turns = [], 0, 100

        def call(self, tool, **kw):
            return str(self.tools[tool][0](**kw))

        def send(self, msg):
            return {"final": "done", "stop": "final", "messages": self.messages}

        def run(self, prompt):
            if self.name == "agent_1":
                assert "doesn't claim" in self.call("execute_python", code=LOGIN + "apis.spotify.like_song(song_id=54, access_token=tok)")
                self.call("wal_read")
                assert self.call("wal_intend", plan="like 54", claims=[{"app": "spotify", "api": "like_song", "target": "54"}]).startswith("Posted")
                assert "Song liked" in self.call("execute_python", code="print(apis.spotify.like_song(song_id=54, access_token=tok))")
                out_vote["agent_1"] = self.call("execute_python", code="print(apis.supervisor.complete_task())")
            elif self.name == "agent_2":
                assert "doesn't claim" in self.call("execute_python", code=LOGIN + "apis.spotify.like_song(song_id=55, access_token=tok)")
                out_vote["agent_2a"] = self.call("execute_python", code="print(apis.supervisor.complete_task())")
                out_vote["agent_2b"] = self.call("execute_python", code="print(apis.supervisor.complete_task())")
            else:
                self.call("execute_python", code="print(apis.supervisor.complete_task())")
            return {"final": "done", "stop": "final", "messages": self.messages}

    run_dir = Path(tempfile.mkdtemp()) / "smoke-unmade"
    run_dir.mkdir()
    r = appworld_team.run("57c3486_2", lambda **kw: Scripted(kw["name"], kw["tools"], kw["system_prompt"]), 3, run_dir,
                          level="L6", completion="all")
    c = r["comm"]
    print("unmade:", {k: c.get(k) for k in ("unmade_reminded", "unmade_at_end", "complete_votes")},
          {k: v[:90] for k, v in out_vote.items()})
    assert "Recorded" in out_vote["agent_1"], out_vote["agent_1"]           # redone: no reminder
    assert out_vote["agent_2a"].startswith("Not recorded yet") and "like_song(song_id='55'" in out_vote["agent_2a"]
    assert "Recorded" in out_vote["agent_2b"]                              # once only
    assert c["unmade_reminded"] == 1 and c["unmade_at_end"] == 1 and c["complete_votes"] == 3 and r["verdict"]["completed"]


scenario("L6")
scenario(None)
scenario_unmade()
scenario_stuck("L6")
scenario_stuck(None)
print("completion (all) test passed")
