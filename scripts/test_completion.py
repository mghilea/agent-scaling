# Run with AppWorld's Python on a node with AppWorld set up: $APPWORLD_PYTHON scripts/test_completion.py
"""End-to-end: deferred completion on a real AppWorld task with scripted agents (no model)."""
import os, sys, json, threading, time, tempfile
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ.setdefault("APPWORLD_ROOT", f"/scratch/{os.environ['USER']}/appworld")
import appworld_team

LOGIN = ('pw = {a["account_name"]: a["password"] for a in apis.supervisor.show_account_passwords()}\n'
         'tok = apis.spotify.login(username=apis.supervisor.show_profile()["email"], password=pw["spotify"])["access_token"]\n')
posted, completed = threading.Event(), threading.Event()
replies = {}

class Scripted:
    def __init__(self, name, tools, system_prompt):
        self.name, self.tools, self.prompt = name, tools, system_prompt
        self.stats = dict(turns=0, tool_calls=0, tool_errors=0, prompt_tokens=0, completion_tokens=0)
        self.messages, self.limit_hits, self.max_turns = [], 0, 100
    def call(self, tool, **kw):
        self.stats["turns"] += 1; self.stats["tool_calls"] += 1
        out = self.tools[tool][0](**kw)
        replies.setdefault(self.name, []).append((tool, str(out)[:300]))
        return str(out)
    def send(self, msg):
        return {"final": "done", "stop": "final", "messages": self.messages}
    def run(self, prompt):
        n = self.name
        if n == "agent_1":
            posted.wait(30)
            self.call("wal_read")
            self.call("wal_intend", plan="like 54", claims=[{"app": "spotify", "api": "like_song", "target": "54"}])
            self.call("wal_read")
            self.call("execute_python", code=LOGIN + "print(apis.spotify.like_song(song_id=54, access_token=tok))")
            self.call("execute_python", code="print(apis.supervisor.complete_task())")
            completed.set()
            self.call("execute_python", code="print('after completing')")          # expect: task complete
        elif n == "agent_2":
            self.call("wal_read")
            self.call("wal_intend", plan="like 55", claims=[{"app": "spotify", "api": "like_song", "target": "55"}])
            self.call("execute_python", code=LOGIN + "print('logged in')")
            posted.set()
            completed.wait(60); time.sleep(0.5)
            self.call("wal_read")
            self.call("execute_python", code="print(apis.spotify.like_song(song_id=55, access_token=tok))")
            # claim closed by the like above: the task completes; anything further is refused or blocked
            self.call("wal_intend", plan="more", claims=[{"app": "spotify", "api": "like_song", "target": "56"}])
            self.call("execute_python", code="print(apis.spotify.like_song(song_id=56, access_token=tok))")
        else:
            completed.wait(60); time.sleep(0.2)
            self.call("execute_python", code="print('agent_3 working')")            # expect: task complete
        return {"final": "done", "stop": "final", "messages": self.messages}

run_dir = Path(tempfile.mkdtemp()) / "smoke-completion"
run_dir.mkdir()
r = appworld_team.run("57c3486_2", lambda **kw: Scripted(kw["name"], kw["tools"], kw["system_prompt"]), 3, run_dir, level="L4")
c = r["comm"]
for a, rs in replies.items():
    for tool, out in rs:
        print(f"{a:8s} {tool:15s} {out[:150]!r}")
print({k: c[k] for k in ("completion_deferred", "finish_wait", "finish_held", "finish_writes", "writes")},
      {k: v for k, v in c.items() if k.startswith("wal_refused") and v}, "completed:", r["verdict"]["completed"])
wal = [json.loads(l) for l in open(run_dir / "wal" / "57c3486_2.jsonl")]
print([ (e.get("kind"), e.get("agent"), e.get("call") or e.get("waiting_for") or e.get("plan")) for e in wal])
assert c["completion_deferred"] == 1 and c["finish_writes"] == 1 and r["verdict"]["completed"]
assert any("Recorded" in o for t, o in replies["agent_1"]) and "task is complete" in replies["agent_1"][-1][1]
assert "A teammate has marked the task complete" in replies["agent_2"][4][1]
assert "task is complete" in replies["agent_3"][0][1]
print("completion test passed")
