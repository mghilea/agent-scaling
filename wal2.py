"""The write-ahead log, second version: one log, enforcement levels L1-L4, used through a benchmark adapter.

(wal.py, the first version, is WorkBench-only and stays as it was, so the earlier runs remain reproducible.)

A team's log is an append-only list of entries, kept in memory for the checks and written to a JSON-lines file
as it grows (each line forced to disk, so the file is the record of what agents were told). Entries:
- intent: an agent's plan and its claims, each {app, api, target}: the API it will call and on what (the record's
  id, or for something new a short description). A new intent replaces the agent's previous one and releases
  whatever that one still claimed.
- change: a write the harness executed, with its outcome and the intent it falls under. The file has a pending
  line before the write runs and the entry after it; agents see the entry.
- release: claims closed because their agent stopped or because they expired.
Changes are logged by the harness, never by agents. Agents read with wal_read and post with wal_intend; they make
changes through the environment as usual, and every write passes commit(), which checks the level's rules, records
the pending line, runs the write and records the change, all under one lock (the caller's).

Levels (cumulative): L1 nothing is checked; L2 a write needs a claim of the agent's current intent (same app and
API, and for an existing record its id in the target); L3a posting needs every entry read; L3b a write needs every
entry read; L4 an intent or write may not conflict with a teammate's: the same record claimed or already changed by
another agent, or a new item another agent already created (same API and identifying arguments).
Claims name their targets at every level; from L2 on, a claim on an existing record must name its id.

The adapter describes the environment: is_write(app, api), has_ids(app, api), the apps and their write APIs.
commit() takes an Action built by the adapter: app, api, path (the API's path template), ids (the values of the
path's id parameters), identity (the call's identifying arguments, for new items) and call (as shown in the log).
"""
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = {
    "L1": set(),
    "L2": {"intent"},
    "L3a": {"intent", "read_before_post"},
    "L3b": {"intent", "read_before_post", "read_before_write"},
    "L4": {"intent", "read_before_post", "read_before_write", "no_conflict"},
}
CLAIM_TTL = 90  # seconds a claim stays open without its change (L4 makes claims exclusive; this frees abandoned ones)


class Refused(Exception):
    """A write the log refused; its message is what the agent is told."""


@dataclass
class Action:
    app: str
    api: str
    path: str
    ids: tuple = ()          # values of the API's path id parameters: an existing record
    identity: tuple = ()     # identifying arguments of a call that creates something new
    call: str = ""
    exempt: bool = False     # logged, but no rule applies (e.g. completing the task)

    @property
    def key(self):
        """What makes two writes 'the same': the record they act on, or the item they create."""
        return (self.app, self.path, self.ids) if self.ids else (self.app, self.api, self.identity)


@dataclass
class Claim:
    app: str
    api: str
    target: str
    ids: set = field(default_factory=set)   # record ids named in the target (APIs with ids only); closed as done
    intent: int = 0
    agent: str = ""
    t: float = 0.0

    def show(self):
        return f"{self.app}.{self.api} {self.target}"


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9@.]+", str(text).lower()))


def _ids_in(text):
    return {m for m in re.findall(r"\d+", str(text))}


class Log:
    def __init__(self, level, adapter, path, lock, event, t0=None, claim_ttl=CLAIM_TTL):
        if level not in LEVELS:
            raise ValueError(f"unknown WAL level {level!r}")
        self.level, self.rules, self.adapter = level, LEVELS[level], adapter
        self.lock, self.event = lock, event
        self.t0 = time.time() if t0 is None else t0
        self.claim_ttl = claim_ttl
        self.entries = []              # what agents see: intents, changes, releases (ids are positions + 1)
        self.seen = Counter()          # agent -> entries seen
        self.intent = {}               # agent -> id of its current intent
        self.claims = {}               # agent -> open claims of its current intent
        self.done = {}                 # Action.key -> (agent, entry id) of writes that went through
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")
        self._lines = 0

    # ---- storage ---------------------------------------------------------------------------------------------
    def _write(self, record):
        self._lines += 1
        self._file.write(json.dumps({"line": self._lines, **record}, default=str) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def _append(self, agent, kind, **data):
        e = {"id": len(self.entries) + 1, "t": round(time.time() - self.t0, 2), "agent": agent, "kind": kind, **data}
        caught_up = self.seen[agent] >= len(self.entries)
        self.entries.append(e)
        self._write(e)
        if caught_up and agent:
            self.seen[agent] = e["id"]
        return e

    def close(self, copy_to=None):
        self._file.close()
        if copy_to:
            Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path, copy_to)

    # ---- what agents see -----------------------------------------------------------------------------------
    def line(self, e):
        if e["kind"] == "intent":
            claims = "; ".join(f"{c['app']}.{c['api']} {c['target']}" for c in e["claims"]) or "none"
            rel = (f" (replaces #{e['replaces']}; released: {'; '.join(e['released'])})" if e.get("released")
                   else f" (replaces #{e['replaces']})" if e.get("replaces") else "")
            return f"#{e['id']} {e['agent']} INTENT: {e['plan']} | claims: {claims}{rel}"
        if e["kind"] == "change":
            under = f" (intent #{e['intent']})" if e.get("intent") else ""
            return f"#{e['id']} {e['agent']} CHANGE{under}: {e['call']} -> {e['result']}"
        return f"#{e['id']} RELEASE {e['owner']}'s claim {e['claim']} ({e['reason']})"

    def catch_up(self, me):
        unseen = self.entries[self.seen[me]:]
        self.seen[me] = len(self.entries)
        return "\n".join(map(self.line, unseen)), len(unseen)

    # ---- claims --------------------------------------------------------------------------------------------
    def _expire(self):
        now = time.time()
        for agent, claims in self.claims.items():
            for c in [c for c in claims if now - c.t > self.claim_ttl]:
                claims.remove(c)
                self._append("", "release", owner=agent, claim=c.show(), intent=c.intent, reason="expired")

    def release_agent(self, me, reason="stopped"):
        """Close an agent's open claims (it stopped working), so teammates see the work is free."""
        with self.lock:
            for c in self.claims.pop(me, []):
                self._append("", "release", owner=me, claim=c.show(), intent=c.intent, reason=reason)

    def _matches(self, claim, action):
        """Whether a claim covers a write: the same app and API (or the same record path), and the record's id."""
        if claim.app != action.app:
            return False
        if action.ids:
            same = claim.api == action.api or self.adapter.path(claim.app, claim.api) == action.path
            return same and bool(set(action.ids) & claim.ids)
        return claim.api == action.api

    def _conflicting_claim(self, me, action):
        for agent, claims in self.claims.items():
            if agent == me:
                continue
            for c in claims:
                if action.ids and self._matches(c, action):
                    return agent, c
                if not action.ids and c.app == action.app and c.api == action.api and action.identity and all(
                        _norm(v) in _norm(c.target) for v in action.identity if len(str(v)) >= 3):
                    return agent, c
        return None

    # ---- the agents' tools -----------------------------------------------------------------------------------
    def read(self, me, all=False):
        with self.lock:
            self._expire()
            if all in (True, "true", "True", 1):
                text, n = "\n".join(map(self.line, self.entries)), len(self.entries)
                self.seen[me] = len(self.entries)
            else:
                text, n = self.catch_up(me)
        self.event(me, "wal_read", n=n, whole=bool(all))
        if n:
            return text
        return "The WAL is empty." if not self.entries else "No new entries since you last read the WAL."

    def _parse_claims(self, claims):
        if isinstance(claims, str):
            try:
                claims = json.loads(claims)
            except json.JSONDecodeError:
                return None, "[error] claims must be a list of {app, api, target} objects"
        if isinstance(claims, dict):
            claims = [claims]
        if not isinstance(claims, list):
            return None, "[error] claims must be a list of {app, api, target} objects"
        out = []
        for c in claims:
            if not isinstance(c, dict):
                return None, f"[error] each claim must be an object with app, api and target, not {c!r}"
            app, api, target = (str(c.get(k, "")).strip() for k in ("app", "api", "target"))
            if not self.adapter.is_write(app, api):
                return None, (f"[error] {app}.{api} is not an API that changes anything. Claims name the API you "
                              "will call to make a change (see apis.api_docs), e.g. {app: 'spotify', "
                              "api: 'download_song', target: '114'}.")
            if not target:
                return None, f"[error] the claim on {app}.{api} needs a target: the record's id, or what you will create"
            ids = _ids_in(target) if self.adapter.has_ids(app, api) else set()
            if self.adapter.has_ids(app, api) and not ids and "intent" in self.rules:
                return None, (f"[error] {app}.{api} acts on an existing record: name its id in the target "
                              f"(e.g. '{api} 114'), one claim per record or several ids in one target.")
            out.append({"app": app, "api": api, "target": target, "ids": sorted(ids)})
        return out, None

    def intend(self, me, plan="", claims=None):
        if not str(plan).strip():
            return "[error] plan is required"
        parsed, error = self._parse_claims(claims if claims is not None else [])
        if error:
            return error
        with self.lock:
            self._expire()
            if "read_before_post" in self.rules:
                unseen, n = self.catch_up(me)
                if n:
                    self.event(me, "wal_refused", rule="read_before_post", unseen=n)
                    return (f"[WAL] Not posted: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n"
                            f"{unseen}\nYou have now seen them. Post your intent again, revised if they overlap with it.")
            if "no_conflict" in self.rules:
                for c in parsed:
                    hit = self._claim_conflict(me, c)
                    if hit:
                        self.event(me, "wal_refused", rule="no_conflict", at="post", claim=c, reason=hit)
                        return (f"[WAL] Not posted: your claim {c['app']}.{c['api']} {c['target']} conflicts with "
                                f"{hit}. Leave that work to them, or claim something else.")
            old = self.claims.pop(me, [])
            e = self._append(me, "intent", plan=str(plan).strip(), claims=parsed,
                             replaces=self.intent.get(me), released=[c.show() for c in old])
            self.intent[me] = e["id"]
            now = time.time()
            self.claims[me] = [Claim(c["app"], c["api"], c["target"], set(c["ids"]), e["id"], me, now) for c in parsed]
        self.event(me, "wal_intend", id=e["id"], claims=parsed)
        return f"Posted as intent #{e['id']}."

    def _claim_conflict(self, me, c):
        """A teammate's open claim or completed change on the same record or new item, as text, or None."""
        same_path = self.adapter.path(c["app"], c["api"])
        for agent, claims in self.claims.items():
            if agent == me:
                continue
            for o in claims:
                if o.app != c["app"]:
                    continue
                if c["ids"] and (o.api == c["api"] or self.adapter.path(o.app, o.api) == same_path) \
                        and o.ids & set(c["ids"]):
                    return f"{agent}'s open claim {o.show()} (intent #{o.intent})"
                if not c["ids"] and o.api == c["api"] and _norm(o.target) == _norm(c["target"]):
                    return f"{agent}'s open claim {o.show()} (intent #{o.intent})"
        for key, (agent, eid) in self.done.items():
            if agent == me or key[0] != c["app"]:
                continue
            if c["ids"] and key[1] == same_path and set(key[2]) & set(c["ids"]):
                return f"a change {agent} already made (#{eid})"
            if not c["ids"] and key[1] == c["api"] and key[2] and all(
                    _norm(v) in _norm(c["target"]) for v in key[2] if len(str(v)) >= 3):
                return f"a change {agent} already made (#{eid})"
        return None

    # ---- the gate ------------------------------------------------------------------------------------------
    def commit(self, me, action, execute):
        """Check the rules, then run execute() and log the change. The caller holds the lock. Raises Refused."""
        self._expire()
        claim = None
        if not action.exempt:
            mine = self.claims.get(me, [])
            claim = next((c for c in mine if self._matches(c, action)), None)
            if "intent" in self.rules and claim is None:
                rule = "intent"
                self.event(me, "wal_refused", rule=rule, call=action.call)
                need = (f"{action.app}.{action.api} with the record's id ({', '.join(action.ids)}) in the target"
                        if action.ids else f"{action.app}.{action.api}")
                raise Refused(f"[WAL] Refused: your current intent doesn't claim {need}. Post an intent with "
                              "wal_intend that claims it, then make the change. Nothing was changed.")
            if "read_before_write" in self.rules:
                unseen, n = self.catch_up(me)
                if n:
                    self.event(me, "wal_refused", rule="read_before_write", call=action.call, unseen=n)
                    raise Refused(f"[WAL] Refused: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n"
                                  f"{unseen}\nYou have now seen them. If they don't affect your plan, make the change "
                                  "again; if they do, post a revised intent first. Nothing was changed.")
            if "no_conflict" in self.rules:
                hit = self._conflicting_claim(me, action)
                prior = self.done.get(action.key)
                if hit or (prior and prior[0] != me):
                    why = (f"{hit[0]}'s open claim {hit[1].show()} (intent #{hit[1].intent})" if hit
                           else f"a change {prior[0]} already made (#{prior[1]})")
                    self.event(me, "wal_refused", rule="no_conflict", at="write", call=action.call, reason=why)
                    raise Refused(f"[WAL] Refused: {action.call} conflicts with {why}. Nothing was changed.")
        self._write({"kind": "pending", "agent": me, "call": action.call, "intent": self.intent.get(me)})
        try:
            out = execute()
        except Exception as ex:  # the environment refused it: logged as a change that changed nothing
            self._append(me, "change", intent=self.intent.get(me), call=action.call,
                         result=f"failed: {str(ex).strip().splitlines()[-1][:160] if str(ex).strip() else type(ex).__name__}")
            raise
        self._append(me, "change", intent=self.intent.get(me), call=action.call, result="done")
        self.done.setdefault(action.key, (me, len(self.entries)))
        if claim is not None and action.ids:
            claim.ids -= set(action.ids)
            if not claim.ids:
                self.claims[me].remove(claim)
        return out

    # ---- tools and numbers ---------------------------------------------------------------------------------
    def tools(self, me, schema):
        return {
            "wal_read": (lambda all=False: self.read(me, all), schema(
                "wal_read", "Read the team's write-ahead log: the entries you haven't seen yet (all=true: every entry).",
                {"all": {"type": "boolean", "description": "Show every entry, not only the ones you haven't seen"}},
                [])),
            "wal_intend": (lambda plan="", claims=None: self.intend(me, plan, claims), schema(
                "wal_intend", "Post your intent to the write-ahead log: your plan, and a claim for each change you "
                "will make. It replaces your previous intent.",
                {"plan": {"type": "string", "description": "What you will do and why"},
                 "claims": {"type": "array", "description": "One claim per change you will make",
                            "items": {"type": "object", "properties": {
                                "app": {"type": "string", "description": "The app, e.g. spotify"},
                                "api": {"type": "string", "description": "The API you will call, e.g. download_song"},
                                "target": {"type": "string", "description": "The record's id (several ids allowed), or "
                                           "for something new a short description (e.g. playlist 'Road trip')"}},
                                "required": ["app", "api", "target"]}}},
                ["plan", "claims"])),
        }

    def counts(self, events):
        kinds = Counter(e["kind"] for e in events)
        refused = Counter(e["rule"] for e in events if e["kind"] == "wal_refused")
        releases = Counter(e["reason"] for e in self.entries if e["kind"] == "release")
        return {"wal_level": self.level, "wal_intents": kinds["wal_intend"], "wal_reads": kinds["wal_read"],
                "wal_changes": sum(e["kind"] == "change" for e in self.entries),
                **{f"wal_refused_{r}": refused[r] for r in ("intent", "read_before_post", "read_before_write",
                                                            "no_conflict")},
                **{f"wal_released_{r}": releases[r] for r in ("stopped", "expired")}}


# The protocol as each agent is told it, by level.
TEXT = """The team keeps a write-ahead log (WAL): an append-only record, shared by all of you, of what each agent intends to change and of every change made through the apps' APIs. wal_read shows the entries you haven't seen (all=true shows every entry). wal_intend posts your intent: your plan, and one claim per change you will make, naming the app, the API you will call and its target: the record's id (e.g. {{"app": "spotify", "api": "download_song", "target": "114, 161"}}), or for something new a short description (e.g. {{"app": "spotify", "api": "create_playlist", "target": "playlist 'Road trip'"}}). A new intent replaces your previous one. Every change any of you makes through the APIs is added to the WAL automatically, with its result; reading data and logging in don't need the WAL.
Before changing anything: 1. read the WAL; 2. post your intent; 3. make the change.
{rules}"""
RULES = {
    "L1": "Nothing checks that you do this: it is up to you.",
    "L2": "A change is refused unless your current intent claims it: the same app and API and, for an existing record, its id in the target.",
    "L3a": "A change is refused unless your current intent claims it: the same app and API and, for an existing record, its id in the target. Posting an intent is refused while the WAL has entries you haven't read; you are then shown them.",
    "L3b": "A change is refused unless your current intent claims it: the same app and API and, for an existing record, its id in the target. Posting an intent, and making a change, are refused while the WAL has entries you haven't read; you are then shown them.",
    "L4": "A change is refused unless your current intent claims it: the same app and API and, for an existing record, its id in the target. Posting an intent, and making a change, are refused while the WAL has entries you haven't read; you are then shown them. An intent or a change is also refused if it conflicts with a teammate's: a record another agent has claimed or already changed, or a new item another agent has claimed or already created. Claims are released when their change is made, when their agent posts a new intent, when it stops, or after {ttl} seconds.",
}


def text(level, claim_ttl=CLAIM_TTL):
    rules = RULES[level].format(ttl=claim_ttl)
    if level != "L1":
        rules += " A refused change raises an error in your code that says why; nothing is changed."
    return TEXT.format(rules=rules)
