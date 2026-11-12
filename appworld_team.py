"""AppWorld teams: several agents on the same task in one shared world, optionally with the write-ahead log (wal2.py).

agent.py --benchmark appworld --topology agent-driven --comm none [--agents 3] [--wal-level L1|L2|L3|L4]

Every agent gets the same task and works on the same apps and accounts: one AppWorld world per task (one set of
databases, scored once at the end by AppWorld's tests). Each agent has its own Python shell (its own variables),
started with the same preamble AppWorld gives its own shell, and its own `apis` object: reads pass straight
through; every write (an API that changes something: POST, PATCH, PUT or DELETE, other than logging in or out) is
recorded, and with a log passes its gate (wal2.Log.commit) first. Agents have no other way to contact each other.

The agents think in parallel threads, but their code runs on the main thread, one cell at a time: AppWorld's
execution timeout uses a signal (main thread only) and it captures printed output process-wide. Cells are short
(the APIs run in-process), and the time each one waited for its turn is recorded.

The task is complete when an agent calls apis.supervisor.complete_task (the first call counts; it is logged, but no
rule applies to it). After that, every agent's code gets "the task is complete" instead of running. An agent that
stops before anyone completed the task is reminded once, as in the single-agent harness.

Records nothing of the task's contents beyond the run's own traces (AppWorld's data must not be redistributed
unencrypted): runs/<run>/traces/<task>.json and the log file runs/<run>/wal/<task>.jsonl.
"""
import getpass
import json
import os
import queue
import threading
from collections import Counter
from pathlib import Path

import appworld_bench as aw
import wal2
from appworld import AppWorld
from IPython.terminal.embed import InteractiveShellEmbed

TIME_LIMIT = 1800  # seconds per task for the whole team
READ_ONLY_WRITES = {"login", "logout"}  # POSTs that only open or close a session

TEAM_TEXT = """You are {me}, one of {n} agents working on this task together; your teammates are {others}. You all received the same task from your supervisor and work on the same apps and accounts: a change any of you makes happens once, and all of you see it in the apps, so a change made twice happens twice. Nobody has assigned roles or a plan: decide for yourself how to get the task done.

{channels}

The task is complete when one of you calls apis.supervisor.complete_task, once for the whole team. After that nobody needs to do anything more."""
NO_CHANNELS = "You have no way to contact your teammates; you can only see the changes they make in the apps."
NO_CHANNELS_WAL = ("Apart from the write-ahead log below, you have no way to contact your teammates; you can only see "
                   "the changes they make in the apps.")
REMINDER = ("The task is not marked complete: nobody on the team has called apis.supervisor.complete_task(). If the "
            "task is done, call it now (once for the team), with answer=... if the task asks for information. If it "
            "isn't done, continue.")
OVER = "The task is complete (it has been marked complete). Stop: reply with a short summary, without calling a tool."


class Adapter:
    """AppWorld's APIs for the log: which ones write, their paths, and the Action for a call."""

    def __init__(self, api_docs):
        self.docs = {(app, api): d for app, apis in api_docs.items() for api, d in apis.items()}

    def is_write(self, app, api):
        d = self.docs.get((app, api))
        return bool(d) and app != "api_docs" and api not in READ_ONLY_WRITES and d["method"] in ("POST", "PUT", "PATCH",
                                                                                                 "DELETE")

    def path(self, app, api):
        return self.docs.get((app, api), {}).get("path", "")

    def has_ids(self, app, api):
        return "{" in self.path(app, api)

    def action(self, app, api, kwargs):
        path = self.path(app, api)
        params = [p.strip("{}") for p in path.split("/") if p.startswith("{")]
        ids = tuple(str(kwargs[p]) for p in params if p in kwargs)
        identity = tuple(str(v) for k, v in sorted(kwargs.items()) if k not in ("access_token", "password"))
        shown = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in kwargs.items() if k != "access_token")
        return wal2.Action(app, api, path, ids, identity, f"{app}.{api}({shown})",
                           exempt=(app, api) == ("supervisor", "complete_task"))


def _empty_like(munch):
    """An empty copy of one of AppWorld's attribute dicts (its class keeps the "not found" message)."""
    cls = type(munch)
    return cls(cls.error_message) if hasattr(cls, "error_message") else cls()


def _capture_preamble(task_id, experiment_name):
    """Open the task's world, and the code AppWorld ran first in its shell (imports and print/input overrides)."""
    cells, run_cell = [], InteractiveShellEmbed.run_cell

    def spy(self, raw_cell, *args, **kwargs):
        cells.append(raw_cell)
        return run_cell(self, raw_cell, *args, **kwargs)

    InteractiveShellEmbed.run_cell = spy
    try:
        world = AppWorld(task_id=task_id, experiment_name=experiment_name)
    finally:
        InteractiveShellEmbed.run_cell = run_cell
    return world, cells[0] if cells else ""


class Team:
    def __init__(self, task_id, make_agent, n_agents, run_dir, level=None, claim_ttl=wal2.CLAIM_TTL, max_turns=100):
        self.t0 = wal2.now()
        self.ids = [f"agent_{i}" for i in range(1, n_agents + 1)]
        self.events, self.max_turns, self.task_id = [], max_turns, task_id
        self.world, preamble = _capture_preamble(task_id, run_dir.name)
        self.adapter = Adapter(self.world.task.api_docs)
        self.lock = threading.RLock()       # the log's lock (writes are committed under it)
        self.jobs = queue.Queue()           # code cells waiting for the main thread
        self.done_agents, self.over, self.expired = set(), False, False
        self.reminded, self.waits = Counter(), []
        work = Path(os.environ.get("WORK", f"/scratch/{getpass.getuser()}"))
        self.wal_path = work / "wal" / run_dir.name / f"{task_id}.jsonl"
        self.wal_copy = run_dir / "wal" / f"{task_id}.jsonl"
        if self.wal_path.exists():
            self.wal_path.unlink()
        self.log = (wal2.Log(level, self.adapter, self.wal_path, self.lock, self.event, t0=self.t0, claim_ttl=claim_ttl)
                    if level else None)
        base = self.world.shell
        self.shells = {}
        for me in self.ids:
            shell = InteractiveShellEmbed(config=base.config)
            shell.ast_node_interactivity = "none"
            if preamble:
                shell.run_cell(preamble)
            shell.user_ns["apis"] = self._apis_for(me)
            self.shells[me] = shell
        self.base_shell = base
        sup = self.world.task.supervisor
        channels = NO_CHANNELS if not level else NO_CHANNELS_WAL + "\n\n" + wal2.text(level, claim_ttl)
        self.agents = {}
        for me in self.ids:
            tools = {"execute_python": (lambda code, me=me: self.execute(me, code), aw._schema(
                "execute_python",
                "Run Python code in your shell (your variables persist between calls; apps are under `apis`) and "
                "return what it printed, or the error.",
                {"code": {"type": "string", "description": "The Python code to run"}}, ["code"]))}
            if self.log:
                tools.update(self.log.tools(me, aw._schema))
            prompt = aw.SYSTEM_PROMPT.format(**sup) + "\n\n" + TEAM_TEXT.format(
                me=me, n=n_agents, others=", ".join(a for a in self.ids if a != me), channels=channels)
            self.agents[me] = make_agent(name=me, tools=tools, system_prompt=prompt)
            self.agents[me].max_turns = max_turns

    def event(self, agent, kind, **data):
        self.events.append({"seq": len(self.events), "t": round(wal2.now() - self.t0, 2), "agent": agent,
                            "kind": kind, **data})

    # ---- the apps, through the log ----------------------------------------------------------------------------
    def _apis_for(self, me):
        """A copy of the world's `apis` for one agent: reads pass through, writes are recorded and gated."""
        apis = self.world.apis
        copy = _empty_like(apis)
        for app in list(apis.keys()):
            orig = apis[app]
            gated = _empty_like(orig)
            for api in list(orig.keys()):
                fn = orig[api]
                gated[api] = self._wrap(me, app, api, fn) if self.adapter.is_write(app, api) else fn
            copy[app] = gated
        return copy

    def _wrap(self, me, app, api, fn):
        def call(*args, **kwargs):
            action = self.adapter.action(app, api, kwargs)

            def execute():
                try:
                    out = fn(*args, **kwargs)
                except Exception as ex:
                    self.event(me, "api_write", app=app, api=api, ids=action.ids, identity=action.identity,
                               call=action.call, ok=False, error=str(ex).strip().splitlines()[-1][:200] if str(ex) else "")
                    raise
                self.event(me, "api_write", app=app, api=api, ids=action.ids, identity=action.identity,
                           call=action.call, ok=True)
                return out
            if not self.log:
                return execute()
            with self.lock:
                return self.log.commit(me, action, execute)
        call.__name__ = api
        return call

    # ---- code execution, on the main thread ---------------------------------------------------------------------
    def execute(self, me, code):
        """Called from an agent's thread: queue the cell for the main thread and wait for its output."""
        if self.over:
            return OVER
        if self.expired:
            return "The team's time for this task is up. Stop: reply with a short summary, without calling a tool."
        box = {"queued": wal2.now()}
        done = threading.Event()
        self.jobs.put((me, code, box, done))
        done.wait()
        return box["out"]

    def _run_cell(self, me, code, box):
        box["waited"] = wal2.now() - box["queued"]
        self.waits.append(box["waited"])
        if self.over:
            box["out"] = OVER
            return
        self.world.shell = self.shells[me]
        try:
            out = self.world.execute(code)
        except Exception as ex:  # AppWorld reports errors in the output; this is a fault of the harness or the world
            out = f"Execution failed. Traceback:\n{type(ex).__name__}: {ex}"
        finally:
            self.world.shell = self.base_shell
        if self.world.task_completed():
            self.over = True
        self.event(me, "tool", tool="execute_python", code=code[:1500], out=out[:600], waited=round(box["waited"], 3))
        box["out"] = out

    # ---- the team ---------------------------------------------------------------------------------------------
    def _agent(self, me, prompt, results):
        agent = self.agents[me]
        self.event(me, "active")
        try:
            r = agent.run(prompt)
            left = self.max_turns - agent.stats["turns"]
            if r["stop"] == "final" and not self.over and not self.expired and left > 0:
                self.reminded[me] += 1
                agent.max_turns = left
                r = agent.send(REMINDER)
            results[me] = r
        except Exception as ex:
            results[me] = {"final": "", "stop": f"error: {type(ex).__name__}: {ex}", "messages": agent.messages}
        finally:
            self.event(me, "idle", reply=str(results.get(me, {}).get("final", ""))[:800],
                       stop=results.get(me, {}).get("stop"))
            if self.log:
                self.log.release_agent(me, "stopped")
            self.done_agents.add(me)
            self.jobs.put(None)  # wake the main thread

    def run(self):
        prompt = aw.TASK_PROMPT.format(instruction=self.world.task.instruction)
        results = {}
        threads = [threading.Thread(target=self._agent, args=(me, prompt, results), daemon=True) for me in self.ids]
        for th in threads:
            th.start()
        while len(self.done_agents) < len(self.ids):
            if wal2.now() - self.t0 > TIME_LIMIT:
                self.expired = True
            try:
                job = self.jobs.get(timeout=1)
            except queue.Empty:
                continue
            if job is not None:
                me, code, box, done = job
                self._run_cell(me, code, box)
                done.set()
        while not self.jobs.empty():  # release anyone still waiting (only possible after a crash)
            job = self.jobs.get()
            if job is not None:
                job[2]["out"] = OVER
                job[3].set()
        return self.result(results)

    def _concurrency(self):
        """How much the agents worked at the same time, and how the log's refusals played out.

        An agent works from its 'active' event to its 'idle' one. overlap: share of the team's time with two or more
        agents working; parallelism: their summed working time over the team's time (1 = one at a time). A refusal
        chain is a run of refusals with no accepted post or write in between; an agent 'stopped after a refusal'
        if a refusal was its last log event."""
        span = {}
        for e in self.events:
            if e["kind"] == "active":
                span.setdefault(e["agent"], [e["t"], e["t"]])
            elif e["kind"] == "idle" and e["agent"] in span:
                span[e["agent"]][1] = e["t"]
        end = max((b for _, b in span.values()), default=0.0)
        points = sorted({t for ab in span.values() for t in ab})
        both = sum(b - a for a, b in zip(points, points[1:])
                   if sum(s <= a and b <= t for s, t in span.values()) >= 2)
        out = {"team_seconds": round(end, 1),
               "overlap": round(both / end, 3) if end else 0.0,
               "parallelism": round(sum(b - a for a, b in span.values()) / end, 2) if end else 0.0}
        if self.log:
            chains, last = [], {}
            run = Counter()
            for e in self.events:
                a = e["agent"]
                if e["kind"] == "wal_refused":
                    run[a] += 1
                    last[a] = "refused"
                elif e["kind"] == "wal_intend" or (e["kind"] == "api_write" and e["ok"]):
                    if run[a]:
                        chains.append(run[a])
                    run[a] = 0
                    last[a] = "ok"
            chains += [c for c in run.values() if c]
            out.update({"refusal_chains": len(chains), "refusal_chain_max": max(chains, default=0),
                        "refusal_chains_3plus": sum(c >= 3 for c in chains),
                        "stopped_after_refusal": sum(v == "refused" for v in last.values())})
        return out

    def result(self, results):
        r = self.world.evaluate()
        stats = Counter()
        for a in self.agents.values():
            stats.update({k: a.stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens",
                                                   "completion_tokens")})
        # Conflicts between agents, in time order: a write to a record (or of a new item) that a teammate had
        # already written, which either went through again (a duplicate) or failed (a collision, e.g. "already
        # downloaded"). Keyed like the log's Action.key: the record's ids, or the call's identifying arguments.
        first, dup, coll = {}, 0, 0
        writes = [e for e in self.events if e["kind"] == "api_write"]
        for e in writes:
            k = (e["app"], e["api"], tuple(e["ids"]) or tuple(e["identity"]))
            if k in first and first[k] != e["agent"]:
                dup += e["ok"]
                coll += not e["ok"]
            if e["ok"]:
                first.setdefault(k, e["agent"])
        comm = {"writes": sum(e["ok"] for e in writes), "failed_writes": sum(not e["ok"] for e in writes),
                "writers": len({e["agent"] for e in writes if e["ok"]}),
                "duplicate_writes": dup, "collisions": coll, "conflicts": dup + coll,
                "cell_wait_mean": round(sum(self.waits) / len(self.waits), 3) if self.waits else 0,
                "cell_wait_max": round(max(self.waits), 3) if self.waits else 0}
        comm.update(self._concurrency())
        if self.log:
            comm.update(self.log.counts(self.events))
            self.log.close(copy_to=self.wal_copy)
        verdict = {"correct": bool(r.success), "tests_passed": r.pass_count, "tests": r.num_tests,
                   "difficulty": r.difficulty, "completed": self.world.task_completed(),
                   "reminded": sum(self.reminded.values()) > 0}
        trace = {"topology": "agent-driven", "benchmark": "appworld", "task": self.task_id, "comm": "none",
                 "wal_level": self.log.level if self.log else None, "events": self.events,
                 **({"wal_log": self.log.entries} if self.log else {}),
                 "conversations": {a: ag.messages for a, ag in self.agents.items()}}
        self.world.close()
        return {"verdict": verdict, "final": "", "stop": "final" if not self.expired else "time_limit",
                **{k: stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens")},
                "agents": len(self.ids), "rounds": 1, "agent_messages": 0, "comm": comm,
                "out_of_calls": sum(a.limit_hits > 0 for a in self.agents.values()),
                "duplicate_changes": comm["duplicate_writes"], "trace": trace}


def run(task_id, make_agent, n_agents, run_dir, level=None, claim_ttl=wal2.CLAIM_TTL, max_turns=100):
    return Team(task_id, make_agent, n_agents, run_dir, level, claim_ttl, max_turns).run()
