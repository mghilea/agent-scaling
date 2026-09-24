"""Multi-agent systems from "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296), on WorkBench.

The paper defines four topologies only abstractly. The message flow here follows the authors'
released code (github.com/ybkim95/agent-scaling at 6f3bfb7, agent_scaling/agents/), and the prompts
are adapted from its prompts/multi-agent/ files:

- Every worker is a full tool-using agent with its own private copy of the WorkBench databases (as
  in the paper's code, where each worker gets its own environment) and a conversation that carries
  over between rounds. At the end of each round it writes a summary of its findings; that summary
  is the only thing it ever sends to another agent.
- independent:   n workers each solve the whole task alone, in parallel, once. No communication.
- centralized:   an orchestrator (an LLM with no tools) splits the task into n subtasks, then each
                 round writes a message to each worker based on that worker's and the rest of the
                 team's latest summaries. After each round from round 2 on it decides whether to stop
                 (like the paper's code, it never stops after round 1).
- decentralized: no orchestrator. Round 1 is independent; in each later round every worker gets all
                 its peers' previous-round summaries and revises its answer (debate).
- hybrid:        centralized, then one extra round in which each worker also gets its peers'
                 summaries directly (first 400 characters of each).

WorkBench is scored on how the databases end up, which the paper's WorkBench adapter doesn't do
(its tools are stubs and it grades the final text), so how each system's final changes are chosen
is ours, following each topology's aggregation policy:
- independent:   every distinct change any worker made. The paper concatenates the workers'
                 answers with no cross-checking; identical changes count once.
- centralized, hybrid: the orchestrator picks which of the workers' changes to apply (its synthesis
                 step, where the paper's orchestrator writes the final answer).
- decentralized: majority vote over the workers' answers, ties to the first agent (as in the paper).
                 A worker's answer is every change it made: as in the paper's code, its workspace
                 carries over between debate rounds, so revising means adding or undoing changes.
                 (Resetting it each round failed: gpt-oss-20b didn't redo changes it had already made.)
"""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import workbench as wb

TOPOLOGIES = ["independent", "centralized", "decentralized", "hybrid"]
# Paper appendix E.2: up to 5 orchestration rounds, 3 debate rounds. (The released configs use 10.)
DEFAULT_ROUNDS = {"independent": 1, "centralized": 5, "decentralized": 3, "hybrid": 5}
PEER_CHARS = 400  # how much of each peer summary the hybrid's peer round passes on

# ---- prompts (adapted from the paper's prompts/multi-agent/*.yaml) ------------------------------

WORKER_ROLE = {
    "independent": "You are working as part of a multi-agent team. "
                   "Use the available tools to work thoroughly and systematically.",
    "centralized": "You are working as part of a multi-agent team. Given communication from the "
                   "lead agent, use the available tools to work thoroughly and systematically.",
    "decentralized": "You are working as part of a multi-agent team of peers. Given your peers' "
                     "answers, use the available tools to work thoroughly and systematically.",
}
WORKER_ROLE["hybrid"] = WORKER_ROLE["centralized"]

WORKER_START = """Here is the task for the team to work on:
{task}

To start, you are given the following objective and guidance
Objective:
{objective}
Guidance:
{guidance}
Begin!"""

SUMMARIZE = ("Based on your exploration so far, write a summary of your findings to be the most helpful "
             "for your multi-agent team.\nThis summary should include the actions you took, the results "
             "you observed, and any conclusions you've drawn.")

ORCHESTRATOR_SYSTEM = """You are an intelligent lead agent who can orchestrate multi-agent tasks.

{date}

For this multi-agent orchestration task, you will coordinate work across {n} agents. Each agent can use these tools:
{tools}

You have several key responsibilities:
1. PLANNING: Analyze the task and create a strategic task decomposition
2. COORDINATION: Provide guidance and feedback to individual agents based on their progress and team context
3. DECISION-MAKING: Determine when the team has done enough to proceed to synthesis
4. SYNTHESIS: Combine the agents' work into the final result

Here is the task: {task}"""

PLANNING = """Create a plan to work on the task.

Create exactly {n} distinct subtasks. Each subtask should be specific and focused.

Return ONLY a JSON object with this structure:
{{
    "subtasks": [
        {{
            "agent_id": "agent_1",
            "objective": "Specific objective here",
            "focus": "Distinct focus area"
        }}
    ],
    "reasoning": "Brief explanation"
}}"""

COORDINATION = """ORCHESTRATOR COORDINATION TASK:

You are coordinating a multi-agent team working on: {task}

CURRENT SITUATION:
- Round: {round}
- Agent: {agent}
- Agent's objective: {objective}
- Agent's strategy/focus: {focus}
- Agent iterations completed: {iterations}

AGENT'S PROGRESS:
{progress}

TEAM CONTEXT (for strategic coordination):
{team}

COORDINATION TASK:
Based on the agent's progress and the broader team context, provide specific guidance to {agent} for their next phase of work.

Consider:
- Is this agent making good progress on their specific objective?
- Are their findings relevant and useful for the overall goal?
- How do their findings complement or conflict with other team findings?
- What specific direction should they take next given the team's overall progress?
- Should they continue exploring, focus on a specific area, pivot approach, or wrap up?

Provide a clear, actionable message (2-3 sentences max) to guide {agent}'s next steps.
Do not include JSON or structured data - just the direct coordination message."""

STOPPING = """ORCHESTRATOR STOPPING DECISION:

You are the orchestrator of a multi-agent team working on: {task}

ALL FINDINGS COLLECTED after {round} rounds (across all rounds and agents):
{findings}

DECISION TASK:
Should you stop the work and synthesize the result, or continue with more rounds?

Consider the COMPLETE picture:
- Has the team done everything the task needs?
- Would more rounds likely improve the result significantly?
- Is there diminishing returns from additional rounds?

Respond with either "STOP" or "CONTINUE" followed by a brief reason based on the complete team progress."""

SELECTION = """SYNTHESIS TASK:
Your multi-agent team has gathered the following information.

{findings}

Each agent worked in its own private copy of the workspace, so none of the changes below have been applied to the real workspace yet. These are all the changes the agents made:
{candidates}

Choose which changes to apply to the real workspace to complete the original task. Pick each needed change exactly once, leave out duplicates and mistakes, and choose none if the task needs no changes.

Return ONLY a JSON object like {{"apply": [1, 3]}}, or {{"apply": []}} for no changes."""

INDEPENDENT_GUIDANCE = "Solve the task completely on your own. Work systematically and complete the task."

DEBATE_STRATEGIES = [  # from the paper's multiagent_decentralized.py
    "Analyze the problem systematically and produce a complete solution",
    "Explore the problem broadly, identify the root cause, then solve it",
    "Focus on the target outcome and work backwards to a solution",
]
DEBATE_FIRST = ("Round 1: produce your best candidate answer to the task independently. "
                "Use the available tools and complete the task when ready.")
DEBATE_ROUND = """Debate round {round} of {rounds}.

Below are your peers' answers from the previous round. Read them carefully. Identify points where you agree, points where they erred, and points where you missed something. Then produce your updated final answer for this round. You may defend your previous answer, refine it, or replace it.

{peers}

Your workspace still has the changes you made in earlier rounds, and your answer is the set of all changes you have made. If your updated answer needs more changes, make them now; if an earlier change was wrong, undo it if you can. Do not repeat changes you have already made.

Produce your updated final answer now."""

PEER_ROUND = "Final round with peer insights. Review your peers' work and finalize your solution."

# ---- building blocks ----------------------------------------------------------------------------


class Worker:
    """A tool-using sub-agent with its own sandbox and a conversation that persists across rounds."""

    def __init__(self, make_agent, agent_id, task, objective, focus, role):
        self.id, self.task, self.objective, self.focus = agent_id, task, objective, focus
        self.sandbox = wb.Sandbox()
        self.agent = make_agent(name=agent_id, tools=self.sandbox.tools,
                                system_prompt=wb.SYSTEM_PROMPT + "\n\n" + role)
        self.summaries = []  # what it sent the team at the end of each round
        self.writes = []     # (round, action) for every change it made (rejected writes excluded)
        self.error = None

    def work(self, round_num, message):
        """One round: act on the message with tools, then summarize its findings for the team."""
        if not self.summaries:
            message = WORKER_START.format(task=self.task, objective=self.objective, guidance=message)
        before = len(self.sandbox.changes)
        try:
            self.agent.send(message)
            self.summaries.append(self.agent.reply(SUMMARIZE))
        except Exception as e:  # one failed worker (e.g. context overflow) shouldn't sink the team
            self.error = f"{type(e).__name__}: {e}"
            self.summaries.append("")
        self.writes += [(round_num, a) for a in self.sandbox.changes[before:]]

    @property
    def last_summary(self):
        return self.summaries[-1] if self.summaries else ""

    @property
    def answer(self):
        """Every change this worker made, in order."""
        return [a for _, a in self.writes]


def in_parallel(fn, items):
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        return list(pool.map(fn, items))


def distinct(actions):
    return list(dict.fromkeys(actions))


def parse_json(text):
    """The first {...} object in a model reply, or None."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


class Team:
    """Bookkeeping for one multi-agent run: its agents, the messages between them, and a trace."""

    def __init__(self, topology, task, make_agent, n_agents, log):
        self.topology, self.task, self.make_agent, self.n, self.log = topology, task, make_agent, n_agents, log
        self.workers = []
        self.orchestrator = None
        self.messages = []  # every inter-agent message: {"round", "from", "to", "text"}
        self.calls = []     # every orchestrator prompt and reply
        self.trace = {}
        self.rounds = 0

    def add_worker(self, agent_id, objective, focus):
        w = Worker(self.make_agent, agent_id, self.task, objective, focus, WORKER_ROLE[self.topology])
        self.workers.append(w)
        return w

    def send(self, round_num, sender, recipients, text):
        """Record a message from one agent to others. Each recipient counts as one message."""
        for to in recipients:
            self.messages.append({"round": round_num, "from": sender, "to": to, "text": text})
        self.log(sender + " → " + ", ".join(recipients), text)

    def ask_orchestrator(self, tag, prompt):
        if self.orchestrator is None:
            tools = "\n".join(f"- {name}: {schema['function']['description'].strip().splitlines()[0]}"
                              for name, (_, schema) in wb.Sandbox().tools.items())
            system = ORCHESTRATOR_SYSTEM.format(date=wb.SYSTEM_PROMPT, n=self.n, tools=tools, task=self.task)
            self.orchestrator = self.make_agent(name="orchestrator", tools={}, system_prompt=system)
        reply = self.orchestrator.ask([{"role": "system", "content": self.orchestrator.system_prompt},
                                       {"role": "user", "content": prompt}])
        self.calls.append({"tag": tag, "prompt": prompt, "reply": reply})
        return reply

    def findings(self):
        return "\n".join(f"- Agent {w.id} (round {i}): {s}" for w in self.workers
                         for i, s in enumerate(w.summaries, 1) if s.strip()) or "No findings yet."

    def result(self, actions):
        agents = self.workers + ([self.orchestrator] if self.orchestrator else [])
        stats = {k: 0 for k in ("turns", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens")}
        for a in agents:
            for k in stats:
                stats[k] += (a.agent.stats if isinstance(a, Worker) else a.stats)[k]
        errors = {w.id: w.error for w in self.workers if w.error}
        return {
            "actions": actions, "stop": "final", **stats,
            "agents": len(self.workers), "rounds": self.rounds, "agent_messages": len(self.messages),
            "trace": {"topology": self.topology, **self.trace, "worker_errors": errors,
                      "messages": self.messages, "orchestrator_calls": self.calls,
                      "conversations": {w.id: w.agent.messages for w in self.workers}},
        }

# ---- orchestrator steps (centralized and hybrid) ------------------------------------------------


def plan(team):
    """Ask the orchestrator for n subtasks; fall back to generic ones if it can't produce valid JSON."""
    for attempt in range(3):
        reply = team.ask_orchestrator(f"planning_{attempt}", PLANNING.format(n=team.n))
        subtasks = (parse_json(reply) or {}).get("subtasks")
        if isinstance(subtasks, list) and subtasks and all(isinstance(s, dict) and s.get("objective")
                                                           for s in subtasks):
            team.trace["plan"] = subtasks[:team.n]
            return [(s["objective"], s.get("focus", "")) for s in subtasks[:team.n]]
    team.trace["plan"] = "fallback"
    return [(f"Work on aspect {i} of the main task", f"Aspect {i}") for i in range(1, team.n + 1)]


def coordinate(team, worker, round_num):
    """The orchestrator's message to one worker for this round."""
    others = [f"Agent {w.id}: {w.last_summary}" for w in team.workers if w is not worker and w.last_summary]
    progress = (f"Agent {worker.id} recent findings:\n{worker.last_summary}" if worker.last_summary
                else f"Agent {worker.id}: No significant findings yet.")
    prompt = COORDINATION.format(task=team.task, round=round_num, agent=worker.id, objective=worker.objective,
                                 focus=worker.focus, iterations=worker.agent.stats["turns"], progress=progress,
                                 team="\n".join(f"- {o}" for o in others) or "No team findings yet.")
    reply = team.ask_orchestrator(f"coordination_{round_num}_{worker.id}", prompt).strip()
    return reply if len(reply) > 10 else f"Work on: {worker.objective}"


def should_stop(team, round_num):
    reply = team.ask_orchestrator(f"stopping_{round_num}",
                                  STOPPING.format(task=team.task, round=round_num, findings=team.findings()))
    return reply.strip().lstrip("*\"'").upper().startswith("STOP")


def select_changes(team):
    """The orchestrator's synthesis: which of the workers' changes become the system's answer."""
    candidates = {}  # action -> who made it
    for w in team.workers:
        for r, a in w.writes:
            candidates.setdefault(a, []).append(f"{w.id}, round {r}")
    team.trace["candidates"] = candidates
    if not candidates:
        return []
    actions = list(candidates)
    listing = "\n".join(f"{i}. {wb.show(a)}  [{'; '.join(candidates[a])}]" for i, a in enumerate(actions, 1))
    prompt = SELECTION.format(findings=team.findings(), candidates=listing)
    for attempt in range(2):
        picked = (parse_json(team.ask_orchestrator(f"selection_{attempt}", prompt)) or {}).get("apply")
        if isinstance(picked, list) and all(isinstance(i, int) and 1 <= i <= len(actions) for i in picked):
            team.trace["selection"] = picked
            return [actions[i - 1] for i in distinct(picked)]
    team.trace["selection"] = "fallback: all distinct changes"
    return actions

# ---- topologies ---------------------------------------------------------------------------------


def independent(team, rounds):
    for i in range(1, team.n + 1):
        team.add_worker(f"agent_{i}", "Solve the task completely on your own", f"Independent agent {i}")
    in_parallel(lambda w: w.work(1, INDEPENDENT_GUIDANCE), team.workers)
    team.rounds = 1
    # synthesis_only: every worker's answer is kept, with no cross-checking or voting.
    return distinct(a for w in team.workers for _, a in w.writes)


def centralized(team, rounds, peer_round=False):
    for i, (objective, focus) in enumerate(plan(team), 1):
        team.add_worker(f"agent_{i}", objective, focus)
    for r in range(1, rounds + 1):
        team.rounds = r
        guidance = in_parallel(lambda w: coordinate(team, w, r), team.workers)
        for w, g in zip(team.workers, guidance):
            team.send(r, "orchestrator", [w.id], g)
        in_parallel(lambda wg: wg[0].work(r, wg[1] if r == 1 else f"Message from the lead agent (round {r}):\n{wg[1]}"),
                    list(zip(team.workers, guidance)))
        for w in team.workers:
            team.send(r, w.id, ["orchestrator"], w.last_summary)
        if r >= 2 and should_stop(team, r):
            break
    if peer_round:  # hybrid: one more round where workers hear from each other directly
        r = team.rounds = team.rounds + 1
        messages = []
        for w in team.workers:
            peers = [p for p in team.workers if p is not w and len(p.last_summary.strip()) > 10]
            for p in peers:
                team.send(r, p.id, [w.id], p.last_summary[:PEER_CHARS])
            section = "".join(f"\n[PEER {p.id}]: {p.last_summary[:PEER_CHARS]}" for p in peers)
            messages.append(PEER_ROUND + (f"\n\n## Direct Peer Findings (shared peer-to-peer){section}" if peers else ""))
        in_parallel(lambda wm: wm[0].work(r, wm[1]), list(zip(team.workers, messages)))
        for w in team.workers:
            team.send(r, w.id, ["orchestrator"], w.last_summary)
    return select_changes(team)


def hybrid(team, rounds):
    return centralized(team, rounds, peer_round=True)


def decentralized(team, rounds):
    for i in range(1, team.n + 1):
        team.add_worker(f"agent_{i}", "Independently produce your best answer to the task",
                        DEBATE_STRATEGIES[(i - 1) % len(DEBATE_STRATEGIES)])
    for r in range(1, rounds + 1):
        team.rounds = r
        if r == 1:
            in_parallel(lambda w: w.work(1, DEBATE_FIRST), team.workers)
            continue
        for w in team.workers:  # everyone hears every peer's previous-round answer
            team.send(r, w.id, [p.id for p in team.workers if p is not w], w.last_summary)
        messages = []
        for w in team.workers:
            peers = "\n\n".join(f"--- Peer response from {p.id} (round {r - 1}) ---\n{p.last_summary}"
                                for p in team.workers if p is not w and p.last_summary)
            messages.append(DEBATE_ROUND.format(round=r, rounds=rounds,
                                                peers=peers or "(no peer responses available from the previous round)"))
        in_parallel(lambda wm: wm[0].work(r, wm[1]), list(zip(team.workers, messages)))
    # Consensus: majority vote over the workers' final answers (their changes), ties to the first agent.
    answers = [(w.id, w.answer) for w in team.workers]
    key = lambda actions: tuple(sorted(a.lower() for a in actions))  # noqa: E731  (as WorkBench compares)
    votes = Counter(key(actions) for _, actions in answers)
    winner_key, count = votes.most_common(1)[0]
    winner, actions = next((aid, acts) for aid, acts in answers if key(acts) == winner_key)
    team.trace["vote"] = {"answers": {aid: [wb.show(a) for a in acts] for aid, acts in answers},
                          "winner": winner, "votes": count}
    return actions


def run(topology, task, make_agent, n_agents=3, rounds=None, log=lambda label, text: None):
    """Run one task through a multi-agent topology. Returns its final changes, stats and trace."""
    team = Team(topology, task, make_agent, n_agents, log)
    actions = globals()[topology](team, rounds or DEFAULT_ROUNDS[topology])
    return team.result(actions)
