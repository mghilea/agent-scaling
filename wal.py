"""The write-ahead log (--wal): an append-only list of a team's intents and changes, which every agent reads.

Before changing the workspace an agent must read the log (wal_read) and append its intent (wal_append: a plan
plus each resource it will change and how, e.g. create calendar "meeting with Lena on 12-01 at 10:00"); then
it may make the changes its latest intent declares. A change is refused, and the entries the agent missed are
shown, if the agent hasn't read every entry, or if its latest intent doesn't declare that kind of change (domain
and create/update/delete; the target is free text and isn't checked). Appending also needs every entry read,
so of two agents planning at once the second always sees the first's intent. Every change is logged
automatically. Nobody is notified of new entries, and the log decides nothing: the agents judge overlaps.

With drafts (agent_driven.py --wal-drafts), an intent whose post was refused for unread entries is kept, and is
recorded when the agent, now caught up, makes a change it covers without posting again.

With piggybacking (agent_driven.py --wal-piggyback), the entries an agent hasn't seen are also appended to the result
of every other tool it calls, so it is usually caught up before it posts or writes; the checks are unchanged.

agent_driven.py: the agents share one workspace, so the log tells them what is about to happen in it.
mas.py: each worker has its own private copy (as in the paper), so the log is how a worker knows what the
others intend and have done in theirs; how the team combines their work is unchanged.
"""
import json
import re
import threading
import time
from collections import Counter

import workbench as wb

# The protocol, as each agent is told it: the shared workspace (agent_driven.py) and private copies (mas.py).
TEXT = """The team keeps a write-ahead log (WAL): an append-only list, shared by all of you, of what each agent intends to change and what it has changed. Before you change anything in the workspace:
1. Read the WAL (wal_read) to see your teammates' intents and changes.
2. Append your intent (wal_append): your plan, and each resource you will change and how, for example: create calendar "meeting with Lena on 2023-12-01 at 10:00", update crm "customer 00000076".
3. Then make the change.
The workspace refuses a change if the WAL has entries you haven't read, or if your latest intent doesn't declare that change; it then tells you why and shows you any entries you missed. Every change you make is added to the WAL automatically. Searches and lookups don't need the WAL."""
# --wal-piggyback: new entries ride on every tool result, so step 1 rarely needs a call of its own.
TEXT_PIGGYBACK = """The team keeps a write-ahead log (WAL): an append-only list, shared by all of you, of what each agent intends to change and what it has changed. Before you change anything in the workspace:
1. Check the WAL for your teammates' intents and changes. New entries are added to the end of your tool results automatically, so you are usually up to date already; wal_read shows any you haven't seen, or the whole log with all=true.
2. Append your intent (wal_append): your plan, and each resource you will change and how, for example: create calendar "meeting with Lena on 2023-12-01 at 10:00", update crm "customer 00000076".
3. Then make the change.
The workspace refuses a change if the WAL has entries you haven't read, or if your latest intent doesn't declare that change; it then tells you why and shows you any entries you missed. Every change you make is added to the WAL automatically. Searches and lookups don't need the WAL."""
TEXT_PRIVATE = """Each agent on the team works in its own private copy of the workspace: your teammates can't see the changes you make, and you can't see theirs. So that you all know what the others are doing, the team keeps a write-ahead log (WAL): an append-only list, shared by all of you, of what each agent intends to change and what it has changed in its own copy. Before you change anything in your workspace:
1. Read the WAL (wal_read) to see your teammates' intents and changes.
2. Append your intent (wal_append): your plan, and each resource you will change and how, for example: create calendar "meeting with Lena on 2023-12-01 at 10:00", update crm "customer 00000076".
3. Then make the change.
Your workspace refuses a change if the WAL has entries you haven't read, or if your latest intent doesn't declare that change; it then tells you why and shows you any entries you missed. Every change you make is added to the WAL automatically. Searches and lookups don't need the WAL."""

# Ablations (--wal-mode): the protocol text for the log without intents, and for the log with nothing enforced.
TEXT_CHANGES_ONLY = """The team keeps a write-ahead log (WAL): an append-only list, shared by all of you, of every change any of you has made. Before you change anything in the workspace:
1. Read the WAL (wal_read) to see your teammates' changes.
2. Then make the change.
The workspace refuses a change if the WAL has entries you haven't read; it then shows you the entries you missed. Every change you make is added to the WAL automatically. Searches and lookups don't need the WAL."""
TEXT_ADVISORY = """The team keeps a write-ahead log (WAL): an append-only list, shared by all of you, of what each agent intends to change and what it has changed. Before you change anything in the workspace:
1. Read the WAL (wal_read) to see your teammates' intents and changes.
2. Append your intent (wal_append): your plan, and each resource you will change and how, for example: create calendar "meeting with Lena on 2023-12-01 at 10:00", update crm "customer 00000076".
3. Then make the change.
Nothing checks that you do this: it is up to you. Every change you make is added to the WAL automatically. Searches and lookups don't need the WAL."""
MODES = ["full", "changes-only", "post-anytime", "advisory"]

# Each WorkBench write tool as the (resource, access) an intent must declare to use it.
WRITES = {
    "email_send_email": ("email", "create"), "email_forward_email": ("email", "create"),
    "email_reply_email": ("email", "create"), "email_delete_email": ("email", "delete"),
    "calendar_create_event": ("calendar", "create"), "calendar_update_event": ("calendar", "update"),
    "calendar_delete_event": ("calendar", "delete"), "analytics_create_plot": ("analytics", "create"),
    "project_management_create_task": ("projects", "create"), "project_management_update_task": ("projects", "update"),
    "project_management_delete_task": ("projects", "delete"),
    "customer_relationship_manager_add_customer": ("crm", "create"),
    "customer_relationship_manager_update_customer": ("crm", "update"),
    "customer_relationship_manager_delete_customer": ("crm", "delete"),
}
# The sandbox names tools like calendar_create_event (WorkBench's calendar.create_event).
if set(WRITES) != {re.sub(r"[^a-zA-Z0-9_-]", "_", t) for t in wb.WRITE_TOOLS}:
    raise RuntimeError("wal.WRITES doesn't match WorkBench's write tools")
RESOURCES = ["email", "calendar", "crm", "projects", "analytics"]
ACCESS = ["create", "update", "delete", "read"]
# What agents may write instead of the canonical names.
_RESOURCE_ALIASES = {"emails": "email", "inbox": "email", "calendar_events": "calendar", "events": "calendar",
                     "event": "calendar", "customer_relationship_manager": "crm", "customers": "crm",
                     "customer": "crm", "project_management": "projects", "project": "projects",
                     "project_tasks": "projects", "tasks": "projects", "task": "projects", "plots": "analytics",
                     "plot": "analytics"}
_ACCESS_ALIASES = {"add": "create", "send": "create", "forward": "create", "reply": "create", "insert": "create",
                   "modify": "update", "edit": "update", "change": "update", "reschedule": "update",
                   "remove": "delete", "cancel": "delete", "search": "read", "view": "read", "lookup": "read"}


def _schema(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


def parse_resources(resources):
    """The declared resources as [{resource, access, target}], or an error message."""
    if isinstance(resources, str):
        try:
            resources = json.loads(resources)
        except json.JSONDecodeError:
            return None, "[error] resources must be a JSON list of {resource, access, target} objects"
    if isinstance(resources, dict):
        resources = [resources]
    if not isinstance(resources, list):
        return None, "[error] resources must be a list of {resource, access, target} objects"
    out = []
    for r in resources:
        if not isinstance(r, dict):
            return None, f"[error] each resource must be an object with resource, access and target, not {r!r}"
        res = str(r.get("resource", "")).strip().lower()
        res = _RESOURCE_ALIASES.get(res, res)
        acc = str(r.get("access", "")).strip().lower()
        acc = _ACCESS_ALIASES.get(acc, acc)
        if res not in RESOURCES or acc not in ACCESS:
            return None, (f"[error] resource must be one of {RESOURCES} and access one of {ACCESS}; "
                          f"got resource={r.get('resource')!r}, access={r.get('access')!r}")
        out.append({"resource": res, "access": acc, "target": str(r.get("target") or "").strip()})
    return out, None


class Log:
    """One team's log. Entry i has id i + 1; seen[agent] is the id of the last entry the agent has seen.

    lock guards the log (agent_driven.py passes the lock it also holds while changing the shared workspace);
    event(agent, kind, **data) records the log's events (by default in self.events).
    gate() and add() expect the caller to hold the lock; the tools take it themselves.
    """

    def __init__(self, t0=None, lock=None, event=None, private=False, drafts=False, mode="full"):
        """mode (ablations): full; changes-only (no intents: only changes are logged, and a change needs every
        entry read); post-anytime (an intent is appended even while the agent has unread entries); advisory
        (nothing is checked or refused: the log and the instructions only)."""
        if mode not in MODES:
            raise ValueError(f"unknown WAL mode {mode!r}")
        self.mode = mode
        self.entries, self.seen, self.intent = [], {}, {}
        # drafts: an intent whose post was refused for unread entries is kept, and recorded when the agent makes
        # a change it covers without re-posting (it has seen the entries it missed by then)
        self.drafts, self.draft = drafts, {}
        self.t0 = time.time() if t0 is None else t0
        self.lock = lock or threading.Lock()
        self.private = private
        self.events = []
        self.event = event or self._event

    def _event(self, agent, kind, **data):
        self.events.append({"seq": len(self.events), "t": round(time.time() - self.t0, 2), "agent": agent,
                            "kind": kind, **data})

    def line(self, e):
        if e["kind"] == "intent":
            note = " (posted with its first change)" if e.get("from_draft") else ""
            res = "; ".join(f"{r['access']} {r['resource']}" + (f" {json.dumps(r['target'])}" if r["target"] else "")
                            for r in e["resources"]) or "none declared"
            return f"#{e['id']} {e['agent']} INTENT{note}: {e['plan']} | resources: {res}"
        where = " in its own copy" if self.private else ""
        why = f" (intent #{e['intent']})" if e.get("intent") is not None else ""
        return f"#{e['id']} {e['agent']} CHANGE{where}{why}: {e['call']} -> {e['result']}"

    def add(self, me, kind, **data):
        e = {"id": len(self.entries) + 1, "t": round(time.time() - self.t0, 2), "agent": me, "kind": kind, **data}
        caught_up = self.seen.get(me, 0) >= len(self.entries)
        self.entries.append(e)
        if caught_up:  # always, unless an ablation lets an agent add entries while it has unread ones
            self.seen[me] = e["id"]
        return e

    def catch_up(self, me):
        """The entries this agent hasn't seen, as text; it has now seen them."""
        unseen = self.entries[self.seen.get(me, 0):]
        self.seen[me] = len(self.entries)
        return "\n".join(map(self.line, unseen)), len(unseen)

    def gate(self, me, tool):
        """None if this change may go ahead, else (reason, what the agent is told)."""
        if self.mode == "advisory":
            return None
        if self.mode == "changes-only":  # no intents: only the check that every entry has been read
            unseen, n = self.catch_up(me)
            if n:
                return "behind", (f"[WAL] Refused: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n"
                                  f"{unseen}\nYou have now seen them. If they don't affect your plan, make the change "
                                  "again.")
            return None
        if self.drafts and me in self.draft:
            return self._gate_with_draft(me, tool)
        intent = self.intent.get(me)
        if intent is None:
            return "no_intent", ("[WAL] Refused: you haven't posted an intent. Read the WAL (wal_read), append your "
                                 "intent with wal_append, declaring each resource you will change and how, then make "
                                 "the change.")
        unseen, n = self.catch_up(me)
        if n:
            return "behind", (f"[WAL] Refused: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n{unseen}\n"
                              "You have now seen them. If they don't affect your plan, make the change again; if they "
                              "do, append a revised intent first.")
        resource, access = WRITES[tool]
        if not any(r["resource"] == resource and r["access"] == access for r in self.entries[intent - 1]["resources"]):
            return "undeclared", (f"[WAL] Refused: your latest intent (#{intent}) doesn't declare {access} {resource}. "
                                  "Append an intent with wal_append that declares it, then make the change.")
        return None

    def _gate_with_draft(self, me, tool):
        """The gate for an agent whose last post was refused: once it is caught up, a change its latest intent
        doesn't cover but its draft does records the draft as its intent (the same checks otherwise)."""
        unseen, n = self.catch_up(me)
        if n:
            return "behind", (f"[WAL] Refused: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n{unseen}\n"
                              "You have now seen them. If they don't affect your plan, make the change again; if they "
                              "do, append a revised intent first.")
        resource, access = WRITES[tool]
        covers = lambda rs: any(r["resource"] == resource and r["access"] == access for r in rs)  # noqa: E731
        intent = self.intent.get(me)
        if intent is not None and covers(self.entries[intent - 1]["resources"]):
            return None
        d = self.draft[me]
        if covers(d["resources"]):
            e = self.add(me, "intent", plan=d["plan"], resources=d["resources"], from_draft=True)
            self.intent[me] = e["id"]
            del self.draft[me]
            self.event(me, "wal_draft_used", id=e["id"], tool=tool)
            return None
        if intent is None:
            return "no_intent", ("[WAL] Refused: you haven't posted an intent. Read the WAL (wal_read), append your "
                                 "intent with wal_append, declaring each resource you will change and how, then make "
                                 "the change.")
        return "undeclared", (f"[WAL] Refused: your latest intent (#{intent}) doesn't declare {access} {resource}. "
                              "Append an intent with wal_append that declares it, then make the change.")

    def piggyback(self, me, fn):
        """fn, with the log entries this agent hasn't seen appended to its result; they then count as seen."""
        def run(**kwargs):
            out = fn(**kwargs)
            with self.lock:
                text, n = self.catch_up(me)
            if not n:
                return out
            self.event(me, "wal_piggyback", n=n)
            return (f"{out}\n\n[WAL] {n} new log entr{'y' if n == 1 else 'ies'} from your teammates since you last looked "
                    f"(you have now seen them):\n{text}")
        return run

    def guard(self, me, sandbox):
        """A sandbox's tools for this agent, with every write going through the log (mas.py's private copies)."""
        tools = {}
        for name, (fn, schema) in sandbox.tools.items():
            if name not in WRITES:
                tools[name] = (fn, schema)
                continue
            def run(_fn=fn, _name=name, **kwargs):
                with self.lock:  # the check, the change and its log entry happen as one step
                    refusal = self.gate(me, _name)
                    if refusal:
                        self.event(me, "wal_refused", tool=_name, args=kwargs, reason=refusal[0])
                        return refusal[1]
                    changes, calls = len(sandbox.changes), len(sandbox.actions)
                    out = _fn(**kwargs)
                    call = wb.show(sandbox.actions[-1]) if len(sandbox.actions) > calls else _name
                    self.add(me, "change", intent=self.intent.get(me), call=call,
                             result="done" if len(sandbox.changes) > changes else f"no change: {out[:150]}")
                return out
            tools[name] = (run, schema)
        return tools

    def tools(self, me):
        def wal_read(all=False):
            with self.lock:
                if all in (True, "true", "True", 1):
                    text, n = "\n".join(map(self.line, self.entries)), len(self.entries)
                    self.seen[me] = len(self.entries)
                else:
                    text, n = self.catch_up(me)
            self.event(me, "wal_read", n=n, whole=bool(all))
            if n:
                return text
            return "The WAL is empty." if not self.entries else "No new entries since you last read the WAL."

        def wal_append(plan="", resources=None):
            if not str(plan).strip():
                return "[error] plan is required"
            declared, error = parse_resources(resources if resources is not None else [])
            if error:
                return error
            with self.lock:
                if self.mode in ("post-anytime", "advisory"):  # appended even while behind; unread entries stay unread
                    e = self.add(me, "intent", plan=str(plan).strip(), resources=declared)
                    self.intent[me] = e["id"]
                    self.event(me, "wal_append", id=e["id"], plan=e["plan"], resources=declared,
                               behind=self.seen.get(me, 0) < len(self.entries))
                    return f"Appended as intent #{e['id']}. You can now make the changes it declares."
                unseen, n = self.catch_up(me)
                if n:
                    self.event(me, "wal_append_refused", unseen=n)
                    if self.drafts:
                        self.draft[me] = {"plan": str(plan).strip(), "resources": declared}
                    return (f"[WAL] Not appended: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n"
                            f"{unseen}\nYou have now seen them. Append your intent again, revised if they overlap with it.")
                e = self.add(me, "intent", plan=str(plan).strip(), resources=declared)
                self.intent[me] = e["id"]
                self.draft.pop(me, None)
            self.event(me, "wal_append", id=e["id"], plan=e["plan"], resources=declared)
            return f"Appended as intent #{e['id']}. You can now make the changes it declares."

        tools = {
            "wal_read": (wal_read, _schema(
                "wal_read", "Read the team's write-ahead log: every intent and change since you last read it.",
                {"all": {"type": "boolean", "description": "Show the whole log, not only the entries you haven't seen"}},
                [])),
            "wal_append": (wal_append, _schema(
                "wal_append", "Append your intent to the write-ahead log, before changing the workspace: your plan "
                "and each resource you will change and how. Your latest intent is the one your changes are checked "
                "against.",
                {"plan": {"type": "string", "description": "What you will do and why"},
                 "resources": {"type": "array", "description": "Each resource you will change (or rely on) and how",
                               "items": {"type": "object", "properties": {
                                   "resource": {"type": "string", "enum": RESOURCES},
                                   "access": {"type": "string", "enum": ACCESS},
                                   "target": {"type": "string", "description": "Which record: its id, or for a new "
                                              "one a short description (e.g. meeting with Lena on 2023-12-01 10:00)"}},
                                   "required": ["resource", "access", "target"]}}},
                ["plan", "resources"])),
        }
        if self.mode == "changes-only":  # no intents to post
            del tools["wal_append"]
        return tools

    def counts(self, events):
        """The log's numbers for a run's results, from the events it recorded."""
        kinds = Counter(e["kind"] for e in events)
        refused = Counter(e["reason"] for e in events if e["kind"] == "wal_refused")
        return {"wal_intents": kinds["wal_append"], "wal_reads": kinds["wal_read"],
                "wal_changes": sum(e["kind"] == "change" for e in self.entries),
                "wal_append_refused": kinds["wal_append_refused"],
                **{f"wal_refused_{r}": refused[r] for r in ("no_intent", "behind", "undeclared")},
                **({"wal_drafts_used": kinds["wal_draft_used"]} if self.drafts else {})}
