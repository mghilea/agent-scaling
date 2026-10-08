"""Agent-driven teams: no rounds, no orchestrator, one shared workspace (WorkBench only).

The paper's topologies (mas.py) fix who talks to whom, when, and what they send: every round each
worker writes a summary, and the harness delivers it. Here the agents decide all of that:

- n peers get the same request and run at the same time, each in its own loop (a thread).
- They act on ONE shared copy of WorkBench's databases, like a team sharing an inbox and a CRM.
  A change made twice happens twice; the team is scored on how that workspace ends up.
- Communication is optional and agent-initiated, through the channels the --comm condition allows:
    message   send_message(to, text): free text to one teammate or to everyone.
    a2a       A2A-style structured protocol: a2a_send_task(to, instruction, data) opens a task with
              an id and a state; the receiver moves it through working / input-required / completed /
              failed / rejected with a2a_update_task(task_id, state, text, data); a2a_get_tasks().
    shared    A shared key-value store every teammate can read and write, empty at the start:
              shared_state_read(key), shared_state_write(key, value, expected_version). Writes are
              versioned (a stale expected_version is refused) and teammates are notified of changes.
  --comm all offers all three, so we can see which the agents choose; none offers no channel.
  --coordinate adds one sentence asking agents to agree on who does what first (not how).
- Anything sent to a busy agent is delivered between its turns; an idle agent (one that replied
  without calling a tool) is woken when something arrives. The run ends when every agent is idle
  with an empty inbox, an agent runs out of turns, or the time limit passes.

Every action, message, task update and state change is logged with a timestamp in trace["events"].

BrowseComp-Plus (browsecomp=True): the document collection is read-only, so there is no shared workspace to
see each other's work in; each agent has its own search tools, and teammates learn what others found only
by communicating. The team gives one answer: a shared submit_team_answer slot that any agent can fill, where
each submission replaces the last and the one standing when the team stops is graded; if nobody submits, the
team's answer is a majority vote over the members' final replies (answer_rule="vote", the earlier runs).
With answer_rule="propose" (the default since 7 October) the team answers in two steps instead: each agent
proposes an answer with its confidence and evidence (propose_answer, delivered to every teammate whatever the
channels), and once all have proposed, everyone receives all the proposals, gets DECIDE_CALLS more calls to
discuss and check them, and one agent submits the team's answer (submit_team_answer). The first submission is
final and ends the task; with no submission the team has no answer. With split=n (BrowseComp only) each agent
searches and opens only its own 1/n of the collection (browsecomp.Session(shard=...)), and every agent is told so.
"""
import json
import random
import re
import threading
import time
from collections import Counter
from itertools import count

import workbench as wb

CHANNELS = {"all": ["message", "a2a", "shared"], "text": ["message"], "a2a": ["a2a"],
            "shared": ["shared"], "none": []}
A2A_STATES = ["working", "input-required", "completed", "failed", "rejected", "canceled"]
TURN_BUDGET = 30        # model calls per agent for the whole task
TURNS_PER_WAKE = 12     # model calls per activation before the agent must pause
TIME_LIMIT = 900        # seconds per task
# BrowseComp-Plus: each agent gets the single agent's 30 calls, usable in one go (research takes many searches).
TURN_BUDGET_BC = TURNS_PER_WAKE_BC = 30
DECIDE_CALLS = 8        # propose rule: extra calls per agent once all proposals are in, to discuss and decide

TEAM_PROMPT = """You are {me}, one of {n} agents on a team; your teammates are {others}. You all received the same request from the user, and you all work in one shared workspace: the same email, calendar, CRM, project board and analytics. Every change any of you makes happens once in that workspace and everyone can see it, so a change made twice happens twice.

Nobody has assigned roles or a plan. Decide as a team how to get the request done.{hint}

{channels}

Your turn ends when you reply without calling a tool. If a teammate contacts you afterwards, you will be woken up. The task is over when nobody has anything left to do."""

CHANNEL_TEXT = {
    "message": "- send_message: send a free-text message to one teammate or to all of them.",
    "a2a": ("- a2a_send_task, a2a_update_task, a2a_get_tasks: a structured agent-to-agent protocol modeled on "
            "A2A. Send a teammate a task with an instruction and optional JSON data; it gets an id and the state "
            "\"submitted\". The receiver moves it to working, input-required, completed, failed or rejected, "
            "optionally with a text note and JSON data, and the sender is notified of each update."),
    "shared": ("- shared_state_read, shared_state_write: a key-value store that every teammate can read and write. "
               "It starts empty; create whatever keys you find useful. Each write gets a version number; pass "
               "expected_version to avoid overwriting a teammate's newer update. Teammates are notified when a key changes."),
}
# --coordinate: one sentence asking for coordination, without saying how.
COORDINATE_HINT = " Before you change anything in the workspace, agree with your teammates on who does what, so that no change is made twice."
NO_CHANNELS = "You have no way to contact your teammates; you can only see the changes they make in the workspace."

# BrowseComp-Plus versions of the team prompt, the coordination hint and the no-channels line.
TEAM_PROMPT_BC = """You are {me}, one of {n} agents on a team; your teammates are {others}. You all received the same question, and each of you searches the document collection with your own search tools. Nobody sees another agent's searches, documents or reasoning.{split}

The team gives one answer: submit_team_answer stores it, and each submission replaces the previous one, whoever made it. The answer standing when the team stops is the team's answer. If nobody submits, the team's answer is the most common of its members' final answers.

Nobody has assigned roles or a plan. Decide as a team how to answer the question.{hint}

{channels}

Your turn ends when you reply without calling a tool. If a teammate contacts you afterwards, you will be woken up. The task is over when nobody has anything left to do."""
TEAM_PROMPT_BC_PROPOSE = """You are {me}, one of {n} agents on a team; your teammates are {others}. You all received the same question, and each of you searches the document collection with your own search tools. Nobody sees another agent's searches, documents or reasoning.{split}

The team gives one answer, in two steps. When you have a candidate answer, call propose_answer with it, your confidence and the key evidence; every teammate receives your proposal. When all {n} of you have proposed, everyone receives all the proposals. Then decide together which answer is best supported, and once the team agrees, one of you calls submit_team_answer. The first submission is the team's final answer and ends the task, so submit only when the team agrees. If nobody submits, the team has no answer.

Nobody has assigned roles or a plan. Decide as a team how to answer the question.{hint}

{channels}

Your turn ends when you reply without calling a tool. If a teammate contacts you afterwards, you will be woken up."""
PROPOSE_NUDGE = ("Before you stop, propose your answer with propose_answer (your answer, your confidence and the key "
                 "evidence), so that your teammates see it.")
TASK_OVER = "The team's final answer has been submitted, so the task is over. Reply without calling any tool."
COORDINATE_HINT_BC = " Before you start searching, agree with your teammates on who investigates what, so that no work is done twice."
NO_CHANNELS_BC = "You have no way to contact your teammates."


def _schema(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


def _as_object(data):
    """Tool arguments that should be JSON objects sometimes arrive as strings."""
    if data is None or isinstance(data, dict):
        return data or {}
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, json.JSONDecodeError):
        return {"value": data}


class Team:
    def __init__(self, task, make_agent, n_agents, comm, log, seed, coordinate=False, browsecomp=False, split=0,
                 answer_rule="vote"):
        self.task, self.log, self.channels = task, log, CHANNELS[comm]
        self.split = split
        self.propose = browsecomp and answer_rule == "propose"
        self.answer_rule = answer_rule if browsecomp else None
        self.proposals, self.final, self.announced, self.team_nudged = {}, None, None, False
        self.bc = None
        if browsecomp:
            import browsecomp as bc
            self.bc, self.sessions, self.team_answer, self.submissions = bc, {}, None, []
        self.comm, self.coordinate = comm, coordinate
        self.ids = [f"agent_{i}" for i in range(1, n_agents + 1)]
        self.sandbox = None if browsecomp else wb.Sandbox()  # the one workspace everybody shares
        self.env_lock = threading.Lock()         # WorkBench tools aren't thread-safe on shared state
        self.cond = threading.Condition()        # guards inboxes, idle and done
        self.inbox = {a: [] for a in self.ids}
        self.idle, self.finished, self.done = set(), set(), False
        self.wakes = Counter()
        self.final_replies = {}                  # each agent's latest full final reply (the log keeps 800 characters)
        self.state = {}                          # shared key-value store: key -> {value, version, by}
        self.state_history = []
        self.a2a = {}                            # A2A tasks: id -> {...}
        self._task_ids = count(1)
        self._seq = count()
        self.events = []
        self.changes = []                        # (agent, action) for every change to the workspace
        self.t0 = time.time()
        rng = random.Random(f"{seed}")
        self.agents = {}
        for me in self.ids:
            # Describe the channels in a per-agent random order so no channel is always listed first.
            order = self.channels[:]
            rng.shuffle(order)
            channels = ("You can coordinate with your teammates, as much or as little as you like:\n"
                        + "\n".join(CHANNEL_TEXT[c] for c in order)) if order else (NO_CHANNELS_BC if browsecomp else NO_CHANNELS)
            hint = (COORDINATE_HINT_BC if browsecomp else COORDINATE_HINT) if coordinate and order else ""
            fields = dict(me=me, n=n_agents, others=", ".join(a for a in self.ids if a != me), channels=channels, hint=hint)
            if browsecomp:
                fields["split"] = " " + self.bc.split_note(self.ids.index(me), self.ids) if split else ""
            prompt = (TEAM_PROMPT_BC_PROPOSE if self.propose else TEAM_PROMPT_BC if browsecomp else TEAM_PROMPT).format(**fields)
            tools = {**self._env_tools(me), **self._comm_tools(me)}
            if self.propose:  # once the team's answer is in, every tool just says the task is over
                tools = {n: (self._unless_over(fn), schema) for n, (fn, schema) in tools.items()}
            agent = make_agent(name=me, tools=tools,
                               system_prompt=(self.bc.system_prompt() if browsecomp else wb.SYSTEM_PROMPT) + "\n\n" + prompt)
            agent.inbox = lambda me=me: self._drain(me)
            self.agents[me] = agent

    # ---- bookkeeping -------------------------------------------------------------------------
    def event(self, agent, kind, **data):
        self.events.append({"seq": next(self._seq), "t": round(time.time() - self.t0, 2), "agent": agent,
                            "kind": kind, **data})

    def _deliver(self, to, text, kind):
        with self.cond:
            if to in self.finished:
                return
            self.inbox[to].append({"kind": kind, "text": text})
            self.cond.notify_all()

    def _drain(self, me):
        with self.cond:
            items, self.inbox[me] = self.inbox[me], []
        return "\n\n".join(i["text"] for i in items) if items else None

    def expired(self):
        return time.time() - self.t0 > TIME_LIMIT

    def over(self):
        return self.done or (self.propose and self.final is not None)

    def _unless_over(self, fn):
        def run(**kwargs):
            return TASK_OVER if self.final is not None else fn(**kwargs)
        return run

    def budget(self):
        if not self.bc:
            return TURN_BUDGET
        return TURN_BUDGET_BC + (DECIDE_CALLS if self.announced is not None else 0)

    @staticmethod
    def _proposal_text(me, p):
        note = " (taken from its final reply: it stopped without proposing)" if p.get("auto") else ""
        return (f"{me}{note}: answer: {p['answer']} | confidence: {p['confidence'] if p['confidence'] is not None else '?'}"
                f" | evidence: {p['evidence'] or '(none given)'}")

    def _record_proposal(self, me, answer, confidence, evidence, auto=False):
        with self.cond:
            p = {"answer": str(answer).strip(), "confidence": confidence, "evidence": str(evidence or "")[:1500],
                 "auto": auto, "t": round(time.time() - self.t0, 2)}
            again = me in self.proposals
            self.proposals[me] = p
            self.event(me, "proposal", answer=p["answer"], confidence=confidence, evidence=p["evidence"], auto=auto,
                       revised=again)
            for other in self.ids:
                if other != me:
                    self._deliver(other, f"[{'revised ' if again else ''}proposal from {self._proposal_text(me, p)}]", "proposal")
            if self.announced is None and len(self.proposals) == len(self.ids):
                self.announced = round(time.time() - self.t0, 2)
                listing = "\n".join("- " + self._proposal_text(a, self.proposals[a]) for a in self.ids)
                text = (f"[All {len(self.ids)} proposals are in]\n{listing}\n\nDecide together which answer is best "
                        "supported. A teammate's proposal may rest on pages only they can see, so ask them if you need to. "
                        "When the team agrees, one of you calls submit_team_answer: the first submission is final and ends "
                        f"the task. You each have {DECIDE_CALLS} more calls for this.")
                self.event(me, "proposals_complete")
                for a in self.ids:
                    self._deliver(a, text, "proposal")

    def _auto_propose(self, me):
        """An agent that stops without proposing: its final reply (or one tool-free call) becomes its proposal."""
        reply = self.final_replies.get(me, "")
        if not reply.strip():
            try:
                reply = self.agents[me].reply(self.bc.OUT_OF_TURNS)
            except Exception as e:  # e.g. its context is full
                self.event(me, "error", error=f"auto-propose: {type(e).__name__}: {e}")
                reply = ""
        m = re.search(r"exact answer\W*:?\**\s*(.+)", reply, re.I)
        c = re.search(r"confidence\W*:?\W*(\d{1,3})", reply, re.I)
        answer = (m.group(1) if m else reply).strip().strip("*").strip()[:300] or "(no answer)"
        self._record_proposal(me, answer, int(c.group(1)) if c else None, "", auto=True)

    # ---- tools -------------------------------------------------------------------------------
    def _env_tools(self, me):
        if self.bc:
            return self._search_tools(me)
        tools = {}
        for name, (fn, schema) in self.sandbox.tools.items():
            def run(_fn=fn, _name=name, **kwargs):
                with self.env_lock:
                    before = len(self.sandbox.changes)
                    out = _fn(**kwargs)
                    new = self.sandbox.changes[before:]
                    self.changes += [(me, a) for a in new]
                self.event(me, "tool", tool=_name, args=kwargs, out=out[:600],
                           changes=[wb.show(a) for a in new])
                return out
            tools[name] = (run, schema)
        return tools

    def _search_tools(self, me):
        """BrowseComp-Plus: this agent's own search session, logged, plus the team's one answer slot."""
        session = self.sessions[me] = (self.bc.Session(shard=self.ids.index(me), n_shards=self.split) if self.split
                                       else self.bc.Session())
        tools = {}
        for name, (fn, schema) in session.tools.items():
            if name == "done":
                continue
            def run(_fn=fn, _name=name, **kwargs):
                out = _fn(**kwargs)
                self.event(me, "tool", tool=_name, args=kwargs, out=out[:600])
                return out
            tools[name] = (run, schema)

        if self.propose:
            def propose_answer(answer="", confidence_score=None, evidence=""):
                if not str(answer).strip():
                    return "[error] answer is required"
                self._record_proposal(me, answer, confidence_score, evidence)
                return (f"Proposal sent to {', '.join(a for a in self.ids if a != me)}." +
                        (" All proposals are in; decide together, then one of you submits." if self.announced is not None
                         else " When everyone has proposed, you'll all receive the proposals."))

            def submit_final(answer="", confidence_score=None):
                with self.cond:
                    missing = [a for a in self.ids if a not in self.proposals]
                    if missing:
                        return (f"[error] The team can submit only after everyone has proposed; still waiting for "
                                f"{', '.join(missing)}. You'll be told when all proposals are in.")
                    if self.final is not None:
                        return TASK_OVER
                    self.final = {"answer": str(answer), "confidence": confidence_score, "by": me,
                                  "t": round(time.time() - self.t0, 2)}
                    self.submissions.append(dict(self.final))
                    self.event(me, "team_answer", answer=str(answer), confidence=confidence_score, final=True)
                    for a in self.ids:
                        if a != me:
                            self._deliver(a, f"[{me} submitted the team's final answer: {answer}] {TASK_OVER}", "final")
                    self.done = True
                    self.cond.notify_all()
                return "Submitted. This is the team's final answer, and the task is over."
            tools["propose_answer"] = (propose_answer, _schema(
                "propose_answer", "Propose your answer to the team, with your confidence and the key evidence. Every "
                "teammate receives it. You can propose again to revise it.",
                {"answer": {"type": "string", "description": "Your exact answer"},
                 "confidence_score": {"type": "integer", "description": "Confidence from 0 to 100"},
                 "evidence": {"type": "string", "description": "The key evidence, with the docids it comes from"}},
                ["answer", "confidence_score", "evidence"]))
            tools["submit_team_answer"] = (submit_final, _schema(
                "submit_team_answer", "Submit the team's final answer, once the team agrees. Allowed only after "
                "everyone has proposed. The first submission is final and ends the task.",
                {"answer": {"type": "string", "description": "The exact final answer"},
                 "confidence_score": {"type": "integer", "description": "Confidence from 0 to 100"}}, ["answer"]))
            return tools

        def submit_team_answer(answer="", confidence_score=None):
            with self.cond:
                replaced = self.team_answer
                self.team_answer = {"answer": str(answer), "confidence": confidence_score, "by": me}
                self.submissions.append({"t": round(time.time() - self.t0, 2), **self.team_answer})
            self.event(me, "team_answer", answer=str(answer), confidence=confidence_score,
                       replaced=replaced["by"] if replaced else None)
            return ("Team answer recorded" + (f", replacing {replaced['by']}'s." if replaced else ".") +
                    " Any later submission by anyone replaces it.")
        tools["submit_team_answer"] = (submit_team_answer, _schema(
            "submit_team_answer", "Submit the team's final answer. Each submission replaces the previous one, "
            "whoever made it.", {"answer": {"type": "string", "description": "The exact final answer"},
                                 "confidence_score": {"type": "integer", "description": "Confidence from 0 to 100"}},
            ["answer"]))
        return tools

    def _comm_tools(self, me):
        others = [a for a in self.ids if a != me]
        tools = {}
        if "message" in self.channels:
            def send_message(to="", text=""):
                targets = others if to == "all" else [to] if to in others else None
                if not targets:
                    return f"[error] 'to' must be one of {others} or 'all'"
                if not str(text).strip():
                    return "[error] empty message"
                for t in targets:
                    self._deliver(t, f"[message from {me} to {'everyone' if to == 'all' else 'you'}]\n{text}", "message")
                self.event(me, "message", to=targets, broadcast=to == "all", text=str(text))
                return f"Sent to {', '.join(targets)}."
            tools["send_message"] = (send_message, _schema(
                "send_message", "Send a free-text message to one teammate or to all teammates.",
                {"to": {"type": "string", "description": f"One of {others}, or 'all'"},
                 "text": {"type": "string", "description": "The message"}}, ["to", "text"]))
        if "a2a" in self.channels:
            def a2a_send_task(to="", instruction="", data=None):
                if to not in others:
                    return f"[error] 'to' must be one of {others}"
                tid = f"T{next(self._task_ids)}"
                data = _as_object(data)
                self.a2a[tid] = {"id": tid, "from": me, "to": to, "instruction": str(instruction), "data": data,
                                 "state": "submitted", "updates": []}
                self._deliver(to, f"[A2A task {tid} from {me}, state=submitted]\n"
                                  + json.dumps({"task_id": tid, "instruction": instruction, "data": data}), "a2a")
                self.event(me, "a2a_send", to=[to], task_id=tid, instruction=str(instruction), data=data)
                return json.dumps({"task_id": tid, "state": "submitted"})
            def a2a_update_task(task_id="", state="", text="", data=None):
                t = self.a2a.get(task_id)
                if not t or me not in (t["from"], t["to"]):
                    return f"[error] no task {task_id!r} of yours"
                if state not in A2A_STATES:
                    return f"[error] state must be one of {A2A_STATES}"
                data = _as_object(data)
                t["state"] = state
                t["updates"].append({"by": me, "state": state, "text": str(text), "data": data})
                other = t["to"] if me == t["from"] else t["from"]
                self._deliver(other, f"[A2A task {task_id} update from {me}: state={state}]\n"
                                     + json.dumps({"task_id": task_id, "text": text, "data": data}), "a2a")
                self.event(me, "a2a_update", to=[other], task_id=task_id, state=state, text=str(text), data=data)
                return json.dumps({"task_id": task_id, "state": state})
            def a2a_get_tasks():
                mine = [t for t in self.a2a.values() if me in (t["from"], t["to"])]
                self.event(me, "a2a_get", n=len(mine))
                return json.dumps(mine) if mine else "No tasks yet."
            tools["a2a_send_task"] = (a2a_send_task, _schema(
                "a2a_send_task", "Send a teammate a task: an instruction plus optional JSON data. Returns its task id.",
                {"to": {"type": "string", "description": f"One of {others}"},
                 "instruction": {"type": "string", "description": "What you want the teammate to do"},
                 "data": {"type": "object", "description": "Optional structured data (JSON object)"}}, ["to", "instruction"]))
            tools["a2a_update_task"] = (a2a_update_task, _schema(
                "a2a_update_task", "Update a task you received (or cancel one you sent). The other side is notified.",
                {"task_id": {"type": "string"},
                 "state": {"type": "string", "enum": A2A_STATES},
                 "text": {"type": "string", "description": "Optional note"},
                 "data": {"type": "object", "description": "Optional results as a JSON object"}}, ["task_id", "state"]))
            tools["a2a_get_tasks"] = (a2a_get_tasks, _schema(
                "a2a_get_tasks", "List the tasks you sent or received, with their state and updates.", {}, []))
        if "shared" in self.channels:
            def shared_state_read(key=None):
                if key:
                    entry = self.state.get(key)
                    self.event(me, "state_read", key=key, found=entry is not None)
                    return json.dumps({key: entry}) if entry else f"No key {key!r}. Keys: {sorted(self.state)}"
                self.event(me, "state_read", key=None, found=bool(self.state))
                return json.dumps(self.state) if self.state else "The shared state is empty."
            def shared_state_write(key="", value="", expected_version=None):
                if not str(key).strip():
                    return "[error] key is required"
                value = value if isinstance(value, str) else json.dumps(value)
                with self.cond:
                    current = self.state.get(key)
                    if expected_version not in (None, "") and current and int(expected_version) != current["version"]:
                        self.event(me, "state_conflict", key=key, expected=expected_version, actual=current["version"])
                        return json.dumps({"error": "version conflict", "current": current})
                    version = (current["version"] + 1) if current else 1
                    self.state[key] = {"value": value, "version": version, "by": me}
                    self.state_history.append({"t": round(time.time() - self.t0, 2), "key": key, "value": value,
                                               "version": version, "by": me})
                for other in others:
                    self._deliver(other, f"[shared state] {me} {'created' if version == 1 else 'updated'} "
                                         f"\"{key}\" (version {version})", "state")
                self.event(me, "state_write", key=key, value=value, version=version, created=version == 1)
                return json.dumps({"key": key, "version": version})
            tools["shared_state_read"] = (shared_state_read, _schema(
                "shared_state_read", "Read one key of the shared state, or every key if none is given.",
                {"key": {"type": "string", "description": "Optional key"}}, []))
            tools["shared_state_write"] = (shared_state_write, _schema(
                "shared_state_write", "Create or update a key in the shared state. Teammates are notified.",
                {"key": {"type": "string"}, "value": {"type": "string", "description": "Text or JSON"},
                 "expected_version": {"type": "integer", "description": "Refuse the write if the key's version differs"}},
                ["key", "value"]))
        return tools

    # ---- running -----------------------------------------------------------------------------
    def _loop(self, me):
        agent = self.agents[me]
        message = self.bc.task_prompt(self.task) if self.bc else f"Request from the user: {self.task}"
        asked_to_propose = False
        try:
            while not self.over() and not self.expired():
                left = self.budget() - agent.stats["turns"]
                if left > 0:
                    agent.max_turns = min(TURNS_PER_WAKE_BC if self.bc else TURNS_PER_WAKE, left)
                    self.wakes[me] += 1
                    self.event(me, "active", wake=self.wakes[me])
                    r = agent.send(message)
                    if (r["final"] or "").strip():
                        self.final_replies[me] = r["final"]
                    if self.propose and not self.over() and me not in self.proposals:
                        if not asked_to_propose and self.budget() - agent.stats["turns"] > 0:
                            asked_to_propose = True
                            self.event(me, "idle", reply=(r["final"] or "")[:800], stop=r["stop"])
                            message = PROPOSE_NUDGE
                            continue
                        self._auto_propose(me)
                    self.event(me, "idle", reply=(r["final"] or "")[:800], stop=r["stop"])
                elif not self.propose or self.announced is not None:
                    break
                elif me not in self.proposals:  # out of calls before everyone proposed: propose, then wait
                    self._auto_propose(me)
                with self.cond:
                    self.idle.add(me)
                    self.cond.notify_all()
                    while not self.inbox[me] and not self.over():
                        waiting = set(self.ids) - self.finished
                        if self.idle >= waiting and not any(self.inbox[a] for a in waiting):
                            if (self.propose and self.final is None and self.announced is not None
                                    and not self.team_nudged):
                                # Everyone went quiet without submitting: remind them once.
                                self.team_nudged = True
                                self.event(me, "team_nudge")
                                for a in waiting:
                                    self._deliver(a, "[Nobody has submitted the team's final answer. Agree on one and "
                                                     "submit it with submit_team_answer now; if nobody submits, the "
                                                     "team has no answer.]", "nudge")
                                continue
                            self.done = True           # everyone idle, nothing in flight
                            self.cond.notify_all()
                            break
                        if self.expired():
                            self.done = True
                            self.cond.notify_all()
                            break
                        self.cond.wait(timeout=2)
                    if not self.inbox[me] or self.over():
                        break
                    self.idle.discard(me)
                message = self._drain(me) or "Continue."
        except Exception as e:  # one crashed agent shouldn't hang the team
            self.event(me, "error", error=f"{type(e).__name__}: {e}")
        finally:
            if self.propose and me not in self.proposals and not self.over():
                try:  # a crashed agent's last reply still becomes its proposal, so the team isn't left waiting
                    self._auto_propose(me)
                except Exception as e:
                    self.event(me, "error", error=f"final auto-propose: {type(e).__name__}: {e}")
            with self.cond:
                self.finished.add(me)
                self.idle.add(me)
                self.cond.notify_all()

    def run(self):
        threads = [threading.Thread(target=self._loop, args=(me,), daemon=True) for me in self.ids]
        for th in threads:
            th.start()
        for th in threads:
            th.join(TIME_LIMIT + 60)
        if self.bc and not self.propose and self.team_answer is None and not any(
                e["kind"] == "idle" and e["reply"].strip() for e in self.events):
            # Nobody submitted or said anything: one tool-free call to the first agent, as the single agent gets.
            first = self.agents[self.ids[0]]
            reply = first.reply("The team stopped without submitting an answer. " + self.bc.OUT_OF_TURNS)
            if reply.strip():
                self.final_replies[self.ids[0]] = reply
            self.event(self.ids[0], "idle", reply=reply[:800], stop="fallback_answer")
        return self.result()

    def result(self):
        stats = Counter()
        for a in self.agents.values():
            stats.update({k: a.stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens")})
        kinds = Counter(e["kind"] for e in self.events)
        deliveries = sum(len(e.get("to", [])) for e in self.events if e["kind"] in ("message", "a2a_send", "a2a_update"))
        dup = Counter(a for _, a in self.changes)
        comm = {
            "messages": kinds["message"], "broadcasts": sum(1 for e in self.events if e["kind"] == "message" and e["broadcast"]),
            "a2a_tasks": kinds["a2a_send"], "a2a_updates": kinds["a2a_update"],
            "state_keys": len(self.state), "state_writes": kinds["state_write"], "state_reads": kinds["state_read"],
            "state_conflicts": kinds["state_conflict"], "deliveries": deliveries,
            "chars_sent": sum(len(e.get("text", "")) + len(e.get("instruction", "")) + len(json.dumps(e.get("data") or {}))
                              + len(e.get("value", "")) for e in self.events
                              if e["kind"] in ("message", "a2a_send", "a2a_update", "state_write")),
        }
        extra, trace_extra = {}, {}
        if self.bc:
            # The team's answer is the last submission; if nobody submitted, a majority vote over each member's
            # last reply (ties to the first agent).
            last = self.final_replies  # full replies: the logged ones are cut at 800 characters, often before the answer
            a = self.team_answer
            if self.propose:
                f = self.final
                answer = (f"Exact Answer: {f['answer']}\nConfidence: {f['confidence'] if f['confidence'] is not None else 100}%"
                          if f else "")
                rule, vote = ("submitted" if f else "no_submission"), None
            elif a:
                answer, rule, vote = (f"Exact Answer: {a['answer']}\nConfidence: "
                                      f"{a['confidence'] if a['confidence'] is not None else 100}%"), "submitted", None
            else:
                answer, votes, vote = self.bc.majority_answer([(x, last[x]) for x in self.ids if x in last])
                rule, vote = "vote", {"votes": votes, "answers": vote}
            seen = Counter(d for x in self.sessions.values() for d in {d for q in x.searches for d in q["docids"]})
            queries = Counter(q["query"].strip().lower() for x in self.sessions.values() for q in x.searches)
            comm.update({"team_answers": len(self.submissions),
                         "answer_authors": len({x["by"] for x in self.submissions}),
                         "docs_seen_by_several": sum(1 for c in seen.values() if c > 1),
                         "repeated_queries": sum(c - 1 for c in queries.values() if c > 1)})
            extra = {"answer": answer, "sessions": list(self.sessions.values())}
            if self.propose:
                after = [e for e in self.events if e["kind"] in ("message", "a2a_send", "a2a_update", "state_write")
                         and self.announced is not None and e["t"] >= self.announced]
                comm.update({"proposals": sum(e["kind"] == "proposal" for e in self.events),
                             "auto_proposals": sum(e["kind"] == "proposal" and e.get("auto") for e in self.events),
                             "comm_after_proposals": len(after), "team_nudged": self.team_nudged,
                             "decided_at": self.final["t"] if self.final else None, "proposals_at": self.announced})
            trace_extra = {"team_answers": self.submissions, "answer_rule": rule, "vote": vote, "split": self.split,
                           "proposals": [e for e in self.events if e["kind"] == "proposal"], "final": self.final,
                           "rule": self.answer_rule,
                           "blocked_opens": {x: s.blocked for x, s in self.sessions.items() if s.blocked}}
            comm["answer_rule"] = rule
        return {
            "actions": [] if self.bc else list(self.sandbox.actions), **extra,
            "stop": "final" if not self.expired() else "time_limit",
            **{k: stats[k] for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens")},
            "agents": len(self.ids), "rounds": max(self.wakes.values(), default=0),
            "agent_messages": deliveries, "comm": comm,
            "duplicate_changes": sum(c - 1 for c in dup.values() if c > 1),
            "trace": {"topology": "agent-driven", "comm": self.comm, "coordinate": self.coordinate,
                      "task": self.task, "events": self.events,
                      "changes": [{"agent": a, "change": wb.show(c)} for a, c in self.changes],
                      "shared_state": self.state, "shared_state_history": self.state_history,
                      "a2a_tasks": list(self.a2a.values()), **trace_extra,
                      "conversations": {a: ag.messages for a, ag in self.agents.items()}},
        }


def run(task, make_agent, n_agents=3, comm="all", log=lambda label, text: None, seed=0, coordinate=False,
        browsecomp=False, split=0, answer_rule="vote"):
    return Team(task, make_agent, n_agents, comm, log, seed, coordinate, browsecomp, split, answer_rule).run()
