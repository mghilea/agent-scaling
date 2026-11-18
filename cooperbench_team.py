"""CooperBench teams: one agent per feature in one shared repository, optionally with the write-ahead log (wal2.py).

    $COOPERBENCH_PYTHON agent.py --benchmark cooperbench --topology agent-driven --comm none [--wal-level L1|L2|L3|L4] \
        --repair-calls

A task is a feature set of a CooperBench task: a pair in the core, flash, lite and all subsets, a group of three in the
team3 subsets. The team has one agent per feature, and each agent is told only its own feature. The agents share one
copy of the repository, in one Apptainer container (cooperbench_bench.Workspace), so an edit any of them makes is at
once in the files all of them read, build and test.
- Each agent has its own read-only bash: commands run on a throwaway overlay with the agent's own home and /tmp, and
  anything a command writes in the repository is discarded.
- The only way to change the repository is the three edit tools: write_file, replace_in_file and delete_file.
- Edits run one at a time (the workspace's lock); bash commands run at the same time.
The agents have no way to contact each other, apart from the log.

With the log, every edit passes its gate (wal2.Log.commit) first. Claims are file-level: each claim names one file,
posted through wal_intend(plan, files). A claim covers every edit of its file, and it stays open while its agent works:
each edit restarts its expiry clock. It is released when its agent posts an intent without that file, when the agent
stops, or when the file goes claim_ttl seconds without an edit. The levels are wal2's:
- L1: nothing is checked;
- L2: an edit needs a claim;
- L3: L2, plus the agent must have read the entries about the same files (teammates' intents and edits) before posting
  or editing; other entries only when it calls wal_read;
- L4: L2, plus the agent must have read every entry before posting or editing;
- L5: L3, plus no edit or claim on a file a teammate holds an open claim on;
- L6: L4, plus the same conflict rule.

The work ends when every agent has stopped (replied without a tool call, or run out of turns), or at the time limit.
It is scored on the final repository: every feature's hidden tests (cooperbench_bench.score_features). The task counts
as solved only if all of them pass.

CooperBench's dataset license is unknown: task contents stay in the run's own files (traces, patches, the log), out of
git and off artifact pages.
"""
import getpass
import json
import os
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

import cooperbench_bench as cb
import wal2

TIME_LIMIT = 7200   # seconds for the whole team (a guard against hangs: agents normally stop well before)
CLAIM_TTL = 600     # seconds a file claim stays open without an edit to the file

TEAM_TEXT = """You are {me}, one of {n} agents working in this repository at the same time; your teammates are {others}. Each of you implements a different feature: yours is the one in the task. You all work on the same files: an edit any of you makes is at once in the repository that all of you read, build and test. When you are all done, every feature's tests run on the final repository, so a change that breaks a teammate's feature fails the task as much as one that breaks yours. Nobody has assigned files or a plan: decide for yourself how to get your feature done.

{channels}

When your feature is done, reply with a short summary of what you changed, without calling a tool. The work ends when all of you have stopped."""
NO_CHANNELS = "You have no way to contact your teammates; you can only see their changes in the files."
NO_CHANNELS_WAL = ("Apart from the write-ahead log below, you have no way to contact your teammates; you can only see "
                   "their changes in the files.")
TIME_UP = "The team's time for this task is up. Stop: reply with a short summary, without calling a tool."

WAL_TEXT = """The team keeps a write-ahead log (WAL): an append-only record, shared by all of you, of what each agent intends to change and of every edit made with the edit tools. wal_read shows the entries you haven't seen (all=true shows every entry). wal_intend posts your intent: your plan, and the files you will edit (paths relative to /workspace/repo, new files included), e.g. {{"plan": "add a retry option to the HTTP client", "files": ["src/client.py", "src/config.py"]}}. Your intent claims those files (shown in the WAL as repo.edit <file>); a new intent replaces your previous one. A claim is released when you post an intent without that file, when you stop, or after {ttl} seconds without an edit to the file. Every edit any of you makes is added to the WAL automatically, with its result; reading files and running commands don't need the WAL.
Before editing: 1. read the WAL; 2. post your intent; 3. make the edits.
{rules}"""
CLAIM = "An edit is refused unless your current intent claims its file."
CATCH_UP = ("Posting an intent, and making an edit, are refused while the WAL has entries you haven't read; you are "
            "then shown them.")
CATCH_UP_REL = ("Posting an intent, and making an edit, are refused while the WAL has entries you haven't read about the "
                "same files (teammates' intents and edits); you are then shown them. Other new entries are shown only "
                "when you call wal_read.")
CONFLICT = ("An intent or an edit is also refused if a teammate holds an open claim on the same file. wal_wait(files) "
            "waits until none of those files is claimed by a teammate (up to 10 minutes), so you can claim them then.")
RULES = {  # the ladder of wal2.LEVELS
    "L1": "Nothing checks that you do this: it is up to you.",
    "L2": CLAIM,
    "L3": CLAIM + " " + CATCH_UP_REL,
    "L4": CLAIM + " " + CATCH_UP,
    "L5": CLAIM + " " + CATCH_UP_REL + " " + CONFLICT,
    "L6": CLAIM + " " + CATCH_UP + " " + CONFLICT,
}
HINT = ("Wait for it with wal_wait (it returns as soon as the file is free: when the teammate posts an intent without "
        "it or stops), or work on other files first.")
WAIT_MAX = 600      # seconds one wal_wait call may block


def wal_text(level, claim_ttl=CLAIM_TTL):
    if level not in RULES:
        raise ValueError(f"CooperBench teams have WAL levels {', '.join(RULES)}, not {level}")
    rules = RULES[level]
    if level != "L1":
        rules += " A refused edit returns a message that says why; nothing is changed."
    return WAL_TEXT.format(ttl=claim_ttl, rules=rules)


class FileAdapter:
    """The repository for the log: one write API, repo.edit, whose claims and changes name one file each."""
    keep_claims = True      # a file is edited many times under one claim
    conflict_hint = HINT

    def is_write(self, app, api):
        return (app, api) == ("repo", "edit")

    def has_ids(self, app, api):
        return True

    def path(self, app, api):
        return "file"       # every edit tool acts on the same kind of record, a file: any edit of it matches its claim

    def ids_in(self, app, api, target):
        return {target.strip()} if target.strip() else set()

    def unclaimed(self, action):
        return f"the file {action.ids[0]}"


class EditFailed(Exception):
    """An edit the workspace rejected (e.g. `old` not found); its message is what the agent is told."""


def _schema(name, description, params, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": params, "required": required}}}


class Team:
    def __init__(self, task, make_agent, run_dir, level=None, claim_ttl=CLAIM_TTL, max_turns=100):
        self.t0 = wal2.now()
        self.task, self.features = task, list(task["features"])
        self.ids = [f"agent_{i}" for i in range(1, len(self.features) + 1)]
        self.feature_of = dict(zip(self.ids, self.features))
        self.events, self._ev_lock = [], threading.Lock()
        self.expired = False
        self.ws = cb.Workspace(task)
        self.lock = self.ws.lock            # edits, and the log's checks, run under the workspace's lock
        try:
            self._setup(make_agent, run_dir, level, claim_ttl, max_turns)
        except BaseException:
            self.ws.remove()
            raise

    def _setup(self, make_agent, run_dir, level, claim_ttl, max_turns):
        task = self.task
        self.max_turns = max_turns
        work = Path(os.environ.get("WORK", f"/scratch/{getpass.getuser()}"))
        self.wal_path = work / "wal" / run_dir.name / f"{task['id']}.jsonl"
        self.wal_copy = run_dir / "wal" / f"{task['id']}.jsonl"
        if self.wal_path.exists():
            self.wal_path.unlink()
        self.log = (wal2.Log(level, FileAdapter(), self.wal_path, self.lock, self.event, t0=self.t0, claim_ttl=claim_ttl)
                    if level else None)
        channels = NO_CHANNELS if not level else NO_CHANNELS_WAL + "\n\n" + wal_text(level, claim_ttl)
        self.agents = {}
        for me in self.ids:
            tools = {"execute_bash": (lambda command, me=me: self.bash(me, command), _schema(
                "execute_bash", "Run a bash command in /workspace/repo (read-only: changes to the repository are "
                "discarded) and return its output and exit code.",
                {"command": {"type": "string", "description": "The bash command to run"}}, ["command"])),
                **cb.edit_tools(self.ws, me, gate=self.gate)}
            if self.log:
                tools.update(self._wal_tools(me))
            prompt = cb.EDIT_PROMPT + "\n\n" + TEAM_TEXT.format(
                me=me, n=len(self.ids), others=", ".join(a for a in self.ids if a != me), channels=channels)
            self.agents[me] = make_agent(name=me, tools=tools, system_prompt=prompt)
            self.agents[me].max_turns = max_turns

    def event(self, agent, kind, **data):
        with self._ev_lock:
            self.events.append({"seq": len(self.events), "t": round(wal2.now() - self.t0, 2), "agent": agent,
                                "kind": kind, **data})

    # ---- the tools ------------------------------------------------------------------------------------------------
    def bash(self, me, command):
        if self.expired:
            return TIME_UP
        out = self.ws.bash(command, agent=me)
        self.event(me, "bash", command=str(command)[:300], discarded="[note] bash can't change" in out)
        return out

    def gate(self, me, tool, path, apply):
        """An edit: through the log's gate when there is one. Returns what the agent is told."""
        if self.expired:
            return TIME_UP
        target, rel = self.ws._resolve(path)
        if target is None:
            return rel
        action = wal2.Action("repo", tool, "file", (rel,), (), f"{tool} {rel}")

        def execute():
            out = apply()
            ok = not out.startswith("[error]")
            self.event(me, "edit", tool=tool, file=rel, ok=ok, out=out[:300])
            if not ok:
                raise EditFailed(out)
            action.call = f"{tool} {out}"  # the change's entry: e.g. "replace_in_file src/x.py: replaced lines 3-5 ..."
            return out
        with self.lock:
            try:
                return self.log.commit(me, action, execute) if self.log else execute()
            except (wal2.Refused, EditFailed) as ex:
                return str(ex)

    def _wal_tools(self, me):
        def intend(plan="", files=None):
            if isinstance(files, str):
                try:
                    files = json.loads(files)
                except json.JSONDecodeError:
                    files = [f for f in re.split(r"[,\s]+", files) if f]
            files = [files] if isinstance(files, str) else files if files is not None else []
            if not isinstance(files, list):
                return "[error] files must be a list of paths relative to /workspace/repo"
            claims, seen = [], set()
            for f in files:
                target, rel = self.ws._resolve(f)
                if target is None:
                    return rel
                if rel not in seen:
                    seen.add(rel)
                    claims.append({"app": "repo", "api": "edit", "target": rel})
            return self.log.intend(me, plan, claims)

        def wait(files=None, timeout=300):
            if self.expired:
                return TIME_UP
            if isinstance(files, str):
                try:
                    files = json.loads(files)
                except json.JSONDecodeError:
                    files = [f for f in re.split(r"[,\s]+", files) if f]
            files = [files] if isinstance(files, str) else files or []
            claims = []
            for f in files:
                target, rel = self.ws._resolve(f)
                if target is None:
                    return rel
                claims.append({"app": "repo", "api": "edit", "target": rel})
            if not claims:
                return "[error] files must name at least one file, relative to /workspace/repo"
            try:
                timeout = max(1.0, min(float(timeout), WAIT_MAX))
            except (TypeError, ValueError):
                timeout = 300.0
            return self.log.wait(me, claims, timeout, stop=lambda: self.expired)
        tools = {
            "wal_read": (lambda all=False: self.log.read(me, all), _schema(
                "wal_read", "Read the team's write-ahead log: the entries you haven't seen yet (all=true: every entry).",
                {"all": {"type": "boolean", "description": "Show every entry, not only the ones you haven't seen"}}, [])),
            "wal_intend": (intend, _schema(
                "wal_intend", "Post your intent to the write-ahead log: your plan, and the files you will edit (it "
                "claims them). It replaces your previous intent.",
                {"plan": {"type": "string", "description": "What you will do and why"},
                 "files": {"type": "array", "items": {"type": "string"},
                           "description": "Every file you will edit or create, relative to /workspace/repo"}},
                ["plan", "files"])),
        }
        if "no_conflict" in self.log.rules:  # only where a teammate's claim can block you (L5, L6)
            tools["wal_wait"] = (wait, _schema(
                "wal_wait", "Wait until none of these files is claimed by a teammate (returns as soon as they are free, "
                "or after timeout seconds), so you can claim them with wal_intend.",
                {"files": {"type": "array", "items": {"type": "string"},
                           "description": "The files you are waiting for, relative to /workspace/repo"},
                 "timeout": {"type": "number", "description": f"Seconds to wait at most (default 300, max {WAIT_MAX})"}},
                ["files"]))
        return tools

    # ---- the team -------------------------------------------------------------------------------------------------
    def prompt(self, me):
        f = self.feature_of[me]
        task_dir = cb.DATASET / self.task["repo"] / f"task{self.task['task_id']}"
        return f"## Feature {f}\n\n{(task_dir / f'feature{f}' / 'feature.md').read_text()}"

    def _agent(self, me, results):
        agent = self.agents[me]
        self.event(me, "active", feature=self.feature_of[me])
        try:
            results[me] = r = agent.run(self.prompt(me))
            # once: an agent stopping with edits the log refused and it never made is reminded of them
            pending = self.log.unmade(me) if self.log else []
            left = self.max_turns - agent.stats["turns"]
            if pending and r.get("stop") == "final" and not self.expired and left > 0:
                self.event(me, "unmade_reminder", n=len(pending), at="stop", calls=[u["call"] for u in pending])
                agent.max_turns = left
                results[me] = agent.send(wal2.unmade_reminder(pending))
        except Exception as ex:
            results[me] = {"final": "", "stop": f"error: {type(ex).__name__}: {ex}"}
        finally:
            r = results.get(me, {})
            self.event(me, "idle", reply=str(r.get("final", ""))[:800], stop=r.get("stop"))
            if self.log:
                self.log.release_agent(me, "stopped")

    def run(self):
        results = {}
        threads = [threading.Thread(target=self._agent, args=(me, results), daemon=True) for me in self.ids]
        for th in threads:
            th.start()
        while any(th.is_alive() for th in threads):
            for th in threads:
                th.join(timeout=5)
            if wal2.now() - self.t0 > TIME_LIMIT:
                self.expired = True
            if wal2.now() - self.t0 > TIME_LIMIT + 900:  # an agent stuck in a model call: score without it
                break
        try:
            return self.result(results)
        finally:
            self.ws.remove()

    def _edit_metrics(self):
        """Edits by agent and file, in time order. crossed: an edit to a file whose last edit was a teammate's (the
        team's edits of that file interleave); overwrites: a write_file (the whole file) over a teammate's edit."""
        edits = [e for e in self.events if e["kind"] == "edit"]
        ok = [e for e in edits if e["ok"]]
        last, crossed, overwrites = {}, 0, 0
        by_file = defaultdict(set)
        for e in ok:
            prev = last.get(e["file"])
            if prev is not None and prev != e["agent"]:
                crossed += 1
                overwrites += e["tool"] == "write_file"
            last[e["file"]] = e["agent"]
            by_file[e["file"]].add(e["agent"])
        return {"edits": len(ok), "failed_edits": len(edits) - len(ok), "writers": len({e["agent"] for e in ok}),
                "files_edited": len(by_file), "shared_files": sum(len(a) > 1 for a in by_file.values()),
                "crossed_edits": crossed, "overwrites": overwrites, "conflicts": crossed,
                "bash_commands": sum(e["kind"] == "bash" for e in self.events),
                "bash_writes_discarded": sum(e["kind"] == "bash" and e["discarded"] for e in self.events)}

    def result(self, results):
        patch = self.ws.patch()
        try:
            verdict = cb.score_features(self.task, patch)
        except Exception as ex:
            verdict = {"correct": False, "features_passed": 0, "features": len(self.features),
                       "patch_lines": len(patch.splitlines()), "eval_error": f"{type(ex).__name__}: {ex}",
                       "_test_output": {}}
        verdict["_patch"] = patch
        stats = Counter()
        for a in self.agents.values():
            stats.update({k: a.stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens",
                                                   "completion_tokens")})
            stats.update({k: a.stats.get(k, 0) for k in ("dollars", "cache_read_tokens", "cache_write_tokens",
                                                          "fallbacks", "refusals") if k in a.stats})
        comm = self._edit_metrics()
        comm.update(wal2.team_concurrency(self.events, bool(self.log), lambda e: e["kind"] == "edit" and e["ok"]))
        comm["agent_stops"] = {a: (results.get(a) or {}).get("stop") for a in self.ids}
        comm["unmade_reminded"] = sum(e["kind"] == "unmade_reminder" for e in self.events)
        comm["unmade_at_end"] = sum(len(self.log.unmade(a)) for a in self.ids) if self.log else 0
        if self.log:
            comm.update(self.log.counts(self.events))
            self.log.close(copy_to=self.wal_copy)
        trace = {"topology": "agent-driven", "benchmark": "cooperbench", "task": self.task["id"],
                 "features": self.feature_of, "comm": "none", "wal_level": self.log.level if self.log else None,
                 "events": self.events, **({"wal_log": self.log.entries} if self.log else {}),
                 "conversations": {a: ag.transcript() if hasattr(ag, "transcript") else ag.messages
                                   for a, ag in self.agents.items()}}
        stops = [(results.get(a) or {}).get("stop") for a in self.ids]
        return {"verdict": verdict, "final": "", "stop": "time_limit" if self.expired else
                next((s for s in stops if s and s != "final"), "final"),
                **{k: stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens")},
                **({k: round(stats[k], 6) if k == "dollars" else stats[k]
                    for k in ("dollars", "cache_read_tokens", "cache_write_tokens", "fallbacks", "refusals")
                    if any(k in a.stats for a in self.agents.values())}),
                "agents": len(self.ids), "rounds": 1, "agent_messages": 0, "comm": comm,
                "out_of_calls": sum(a.limit_hits > 0 for a in self.agents.values()),
                "duplicate_changes": comm["crossed_edits"], "trace": trace}


def run(task, make_agent, run_dir, level=None, claim_ttl=CLAIM_TTL, max_turns=100):
    return Team(task, make_agent, run_dir, level, claim_ttl, max_turns).run()
