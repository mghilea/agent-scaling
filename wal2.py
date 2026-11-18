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

Levels (the ladder of 2026-11-18; L0 is no log at all):
- L1: nothing is checked.
- L2: a write needs a claim of the agent's current intent (same app and API, and for an existing record its id in the
  target).
- L3: L2, and posting or writing is refused while the agent has unread entries relevant to it: teammates' intents and
  changes on the same records (for new items, the same API). It is then shown those, and only those: other unread
  entries stay unread until it calls wal_read ("relevant_catchup").
- L4: L2, and posting or writing needs every entry read.
- L5: L3, and an intent or write may not touch a target a teammate holds an open claim on (the same record, or the same
  new item).
- L6: L4 plus the same conflict rule.
Once a claim closes, its target is free again: the log doesn't stop an agent from redoing what a teammate already did.
Earlier names: runs before 2026-11-18 named L3 used this L4's rules, and the L4 of 2026-11-17 (only in tests and checks)
is this L6. Kept so the runs tagged lad02-lad06 can be reproduced: L4v0 is L2 plus catch-up on relevant entries
("relevant_read", where the other unread entries are shown with the result and count as read) plus refusing both open
claims and work a teammate already did ("no_redo"); L4v0-strict is L4v0 with catch-up on every entry.
Claims name their targets at every level; from L2 on, a claim on an existing record must name its id.

The adapter describes the environment: is_write(app, api), has_ids(app, api), path(app, api). Optional hooks, for
environments other than AppWorld's (whose adapter has none of them, so its log behaves as before):
- ids_in(app, api, target): the ids a claim's target names (default: the numbers in it);
- keep_claims = True: a claim stays open after its changes (default: a claim on records closes once each has been
  changed), and every change it covers restarts its expiry clock: for files, which an agent edits many times;
- conflict_hint: what a refused agent is told to do instead (default: leave that work to the teammate);
- unclaimed(action): how a refusal for a missing claim names what to claim (default: the app, API and record ids).
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
    "L3": {"intent", "relevant_catchup"},
    "L4": {"intent", "read_before_post", "read_before_write"},
    "L5": {"intent", "relevant_catchup", "no_conflict"},
    "L6": {"intent", "read_before_post", "read_before_write", "no_conflict"},
    # earlier definitions (runs tagged lad02-lad06, which were named L4 and L4-strict), kept for reproducibility
    "L4v0": {"intent", "relevant_read", "no_conflict", "no_redo"},
    "L4v0-strict": {"intent", "read_before_post", "read_before_write", "no_conflict", "no_redo"},
}
CLAIM_TTL = 90  # seconds a claim stays open without its change (L4 makes claims exclusive; this frees abandoned ones)

# AppWorld freezes the clock with freezegun for as long as a task's world is open: the time module's clock then
# returns the task's simulated date, constant, wherever the call stack has no threading frame within freezegun's
# 5-frame look-up, i.e. in nearly all of the log's code, so claims never aged (runs before 2026-11-12). The log keeps
# the real clock inside a tuple, where freezegun's scan of module attributes doesn't swap it for the fake one.
try:
    import freezegun.api as _fg
    _CLOCK = (_fg.real_time,)
    del _fg
except ImportError:
    _CLOCK = (time.time,)


def now():
    """Seconds since the epoch, from the real clock even while AppWorld's is frozen."""
    return _CLOCK[0]()


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
        self.keep_claims = getattr(adapter, "keep_claims", False)
        self.conflict_hint = getattr(adapter, "conflict_hint", "Leave that work to them, or claim something else.")
        self.lock, self.event = lock, event
        self.t0 = now() if t0 is None else t0
        self.claim_ttl = claim_ttl
        self.entries = []              # what agents see: intents, changes, releases (ids are positions + 1)
        self.seen = Counter()          # agent -> entries seen (a read position: the first n entries)
        self.shown = {}                # agent -> ids past its read position it has already been shown (L3, L5: the
                                       # relevant entries of a refusal), so wal_read doesn't repeat them
        self.intent = {}               # agent -> id of its current intent
        self.claims = {}               # agent -> open claims of its current intent
        self.done = {}                 # Action.key -> (agent, entry id) of writes that went through
        self._unmade = {}              # agent -> {Action.key: refused change it hasn't made since} (see commit)
        self.finishing = False         # the task was marked complete while teammates held claims: no new intents,
                                       # and only writes an agent has claimed (set by the team, see appworld_team.py)
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
        e = {"id": len(self.entries) + 1, "t": round(now() - self.t0, 2), "agent": agent, "kind": kind, **data}
        shown = self.shown.get(agent, ())
        caught_up = all(x["id"] in shown for x in self.entries[self.seen[agent]:])
        self.entries.append(e)
        self._write(e)
        if caught_up and agent:
            self.seen[agent] = e["id"]
            self.shown.pop(agent, None)
        elif agent and "relevant_catchup" in self.rules:
            self.shown.setdefault(agent, set()).add(e["id"])  # its own entry: nothing for it to read
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
        shown = self.shown.pop(me, ())
        unseen = [e for e in self.entries[self.seen[me]:] if e["id"] not in shown]
        self.seen[me] = len(self.entries)
        return "\n".join(map(self.line, unseen)), len(unseen)

    def _unseen_relevant(self, me, targets):
        """L3, L5: unread entries relevant to any of targets [(app, api, path, ids)], not shown to the agent yet."""
        shown = self.shown.get(me, ())
        return [e for e in self.entries[self.seen[me]:]
                if e["id"] not in shown and any(self._relevant(me, e, *t) for t in targets)]

    def _show(self, me, entries):
        self.shown.setdefault(me, set()).update(e["id"] for e in entries)
        n = len(entries)
        return "\n".join(map(self.line, entries)), f"entr{'y' if n == 1 else 'ies'}", ("it" if n == 1 else "them")

    def _relevant(self, me, e, app, api, path, ids):
        """Whether an entry concerns the same record (ids under the same path) or, for a new item, the same API."""
        if e["kind"] == "intent" and e["agent"] != me:
            for c in e["claims"]:
                if c["app"] != app:
                    continue
                if ids and (c["api"] == api or self.adapter.path(c["app"], c["api"]) == path) and set(c["ids"]) & set(ids):
                    return True
                if not ids and c["api"] == api:
                    return True
        if e["kind"] == "change" and e["agent"] != me and e.get("app") == app:
            if ids and e.get("path") == path and set(e.get("ids", ())) & set(ids):
                return True
            if not ids and e.get("api") == api:
                return True
        return False

    def catch_up_relevant(self, me, targets):
        """L4: the unread entries, split into those relevant to any of targets [(app, api, path, ids)] and the rest.
        All of them count as seen now. Returns (relevant text, other text, n relevant, n other)."""
        unseen = self.entries[self.seen[me]:]
        self.seen[me] = len(self.entries)
        rel = [e for e in unseen if any(self._relevant(me, e, *t) for t in targets)]
        other = [e for e in unseen if e not in rel]
        return "\n".join(map(self.line, rel)), "\n".join(map(self.line, other)), len(rel), len(other)

    @staticmethod
    def _news(text, n):
        return (f"\n[WAL] {n} new entr{'y' if n == 1 else 'ies'} from your teammates since you last read "
                f"(you have now seen {'it' if n == 1 else 'them'}):\n{text}") if n else ""

    # ---- claims --------------------------------------------------------------------------------------------
    def _expire(self):
        t = now()
        for agent, claims in self.claims.items():
            for c in [c for c in claims if t - c.t > self.claim_ttl]:
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

    def _conflicting_claim(self, me, action, own=None):
        """A teammate's open claim on the target of this write: the same record or, for something new, the same target
        as the writer's own claim that covers the write. (The earlier definitions matched the call's identifying
        arguments against the claim's text instead.)"""
        for agent, claims in self.claims.items():
            if agent == me:
                continue
            for c in claims:
                if action.ids and self._matches(c, action):
                    return agent, c
                if action.ids or c.app != action.app or c.api != action.api:
                    continue
                if "no_redo" not in self.rules and own is not None:
                    if _norm(c.target) == _norm(own.target):
                        return agent, c
                elif action.identity and all(_norm(v) in _norm(c.target) for v in action.identity if len(str(v)) >= 3):
                    return agent, c
        return None

    # ---- the agents' tools -----------------------------------------------------------------------------------
    def read(self, me, all=False):
        with self.lock:
            self._expire()
            if all in (True, "true", "True", 1):
                text, n = "\n".join(map(self.line, self.entries)), len(self.entries)
                self.seen[me] = len(self.entries)
                self.shown.pop(me, None)
            else:
                text, n = self.catch_up(me)
        self.event(me, "wal_read", n=n, whole=bool(all))
        if n:
            return text
        return "The WAL is empty." if not self.entries else "No new entries since you last read the WAL."

    def _target_ids(self, app, api, target):
        hook = getattr(self.adapter, "ids_in", None)
        return set(hook(app, api, target)) if hook else _ids_in(target)

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
            ids = self._target_ids(app, api, target) if self.adapter.has_ids(app, api) else set()
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
            if self.finishing:
                self.event(me, "wal_refused", rule="finishing", at="post")
                return ("[WAL] Not posted: a teammate has marked the task complete. No new intents now: finish the "
                        "changes you already claimed, then stop.")
            if "read_before_post" in self.rules:
                unseen, n = self.catch_up(me)
                if n:
                    self.event(me, "wal_refused", rule="read_before_post", unseen=n)
                    return (f"[WAL] Not posted: the WAL has {n} entr{'y' if n == 1 else 'ies'} you hadn't read:\n"
                            f"{unseen}\nYou have now seen them. Post your intent again, revised if they overlap with it.")
            if "relevant_catchup" in self.rules:
                rel = self._unseen_relevant(me, [(c["app"], c["api"], self.adapter.path(c["app"], c["api"]),
                                                  tuple(c["ids"])) for c in parsed])
                if rel:
                    text, word, pron = self._show(me, rel)
                    self.event(me, "wal_refused", rule="relevant_catchup", at="post", unseen=len(rel))
                    return (f"[WAL] Not posted: the WAL has {len(rel)} {word} you hadn't read about what you are "
                            f"claiming:\n{text}\nYou have now seen {pron}. Post your intent again, revised if they "
                            "overlap with it.")
            news = ""
            if "relevant_read" in self.rules:
                targets = [(c["app"], c["api"], self.adapter.path(c["app"], c["api"]), tuple(c["ids"])) for c in parsed]
                rel, other, nr, no = self.catch_up_relevant(me, targets)
                if nr:
                    self.event(me, "wal_refused", rule="relevant_read", at="post", unseen=nr, other=no)
                    return (f"[WAL] Not posted: the WAL has {nr} entr{'y' if nr == 1 else 'ies'} you hadn't read about "
                            f"what you are claiming:\n{rel}{self._news(other, no)}\nYou have now seen them. Post your intent "
                            "again, revised if they overlap with it.")
                if no:
                    self.event(me, "wal_delivered", at="post", n=no)
                news = self._news(other, no)
            for c in parsed:
                hit = self._claim_conflict(me, c)
                if hit:
                    self.event(me, "wal_refused", rule=hit[0], at="post", claim=c, reason=hit[1])
                    return (f"[WAL] Not posted: your claim {c['app']}.{c['api']} {c['target']} conflicts with "
                            f"{hit[1]}. {self.conflict_hint}")
            old = self.claims.pop(me, [])
            e = self._append(me, "intent", plan=str(plan).strip(), claims=parsed,
                             replaces=self.intent.get(me), released=[c.show() for c in old])
            self.intent[me] = e["id"]
            t = now()
            self.claims[me] = [Claim(c["app"], c["api"], c["target"], set(c["ids"]), e["id"], me, t) for c in parsed]
        self.event(me, "wal_intend", id=e["id"], claims=parsed)
        return f"Posted as intent #{e['id']}." + news

    def wait(self, me, claims, timeout=300, poll=2.0, stop=None):
        """Block until none of these claims would conflict with a teammate's open claim (L5, L6), or timeout seconds
        pass, or stop() is true. Doesn't claim anything: the agent posts its intent afterwards. Returns the reply."""
        parsed, error = self._parse_claims(claims)
        if error:
            return error
        t0 = now()
        while True:
            with self.lock:
                self._expire()
                held = [(c, hit[1]) for c in parsed if (hit := self._claim_conflict(me, c))]
            waited = round(now() - t0, 1)
            if not held:
                self.event(me, "wal_wait", waited=waited, free=True)
                return (f"Free now (after {waited:.0f}s): no teammate holds a claim on "
                        f"{', '.join(c['target'] for c in parsed)}. Post your intent with wal_intend to claim it.")
            if waited >= timeout or (stop and stop()):
                self.event(me, "wal_wait", waited=waited, free=False)
                return (f"Still held after {waited:.0f}s: " + "; ".join(f"{c['target']} by {why}" for c, why in held)
                        + ". You can wait again, or work on something else meanwhile.")
            time.sleep(poll)

    def _claim_conflict(self, me, c):
        """(rule, text) for a teammate's open claim on the same target (no_conflict) or, in the earlier definitions,
        a change a teammate already made to it (no_redo); None if neither."""
        same_path = self.adapter.path(c["app"], c["api"])
        if "no_conflict" in self.rules:
            for agent, claims in self.claims.items():
                if agent == me:
                    continue
                for o in claims:
                    if o.app != c["app"]:
                        continue
                    if c["ids"] and (o.api == c["api"] or self.adapter.path(o.app, o.api) == same_path) \
                            and o.ids & set(c["ids"]):
                        return "no_conflict", f"{agent}'s open claim {o.show()} (intent #{o.intent})"
                    if not c["ids"] and o.api == c["api"] and _norm(o.target) == _norm(c["target"]):
                        return "no_conflict", f"{agent}'s open claim {o.show()} (intent #{o.intent})"
        if "no_redo" in self.rules:
            for key, (agent, eid) in self.done.items():
                if agent == me or key[0] != c["app"]:
                    continue
                if c["ids"] and key[1] == same_path and set(key[2]) & set(c["ids"]):
                    return "no_redo", f"a change {agent} already made (#{eid})"
                if not c["ids"] and key[1] == c["api"] and key[2] and all(
                        _norm(v) in _norm(c["target"]) for v in key[2] if len(str(v)) >= 3):
                    return "no_redo", f"a change {agent} already made (#{eid})"
        return None

    # ---- the gate ------------------------------------------------------------------------------------------
    def commit(self, me, action, execute):
        """Check the rules, then run execute() and log the change. The caller holds the lock. Raises Refused.

        A refused change is remembered as unmade until the agent makes a change to the same target (Action.key: the
        same record or file, or the same new item), so the team can remind an agent that stops with work it never
        redid (unmade())."""
        try:
            out = self._commit(me, action, execute)
        except Refused as ex:
            if not action.exempt:
                self._unmade.setdefault(me, {})[action.key] = {
                    "call": action.call, "t": round(now() - self.t0, 2), "why": str(ex).splitlines()[0][:200]}
            raise
        self._unmade.get(me, {}).pop(action.key, None)
        return out

    def unmade(self, me):
        """Changes this agent tried that were refused and that it hasn't made since, oldest first."""
        return sorted(self._unmade.get(me, {}).values(), key=lambda u: u["t"])

    def _commit(self, me, action, execute):
        self._expire()
        claim = None
        if not action.exempt:
            mine = self.claims.get(me, [])
            claim = next((c for c in mine if self._matches(c, action)), None)
            if self.finishing and claim is None:
                self.event(me, "wal_refused", rule="finishing", call=action.call)
                raise Refused("[WAL] Refused: a teammate has marked the task complete; only changes you had already "
                              "claimed are allowed now. Nothing was changed.")
            if "intent" in self.rules and claim is None:
                rule = "intent"
                self.event(me, "wal_refused", rule=rule, call=action.call)
                need = (self.adapter.unclaimed(action) if hasattr(self.adapter, "unclaimed")
                        else f"{action.app}.{action.api} with the record's id ({', '.join(action.ids)}) in the target"
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
            if "relevant_catchup" in self.rules:
                rel = self._unseen_relevant(me, [(action.app, action.api, action.path, action.ids)])
                if rel:
                    text, word, pron = self._show(me, rel)
                    self.event(me, "wal_refused", rule="relevant_catchup", at="write", call=action.call,
                               unseen=len(rel))
                    raise Refused(f"[WAL] Refused: the WAL has {len(rel)} {word} you hadn't read about what you are "
                                  f"changing:\n{text}\nYou have now seen {pron}. If they don't affect your plan, make the "
                                  "change again; if they do, post a revised intent first. Nothing was changed.")
            if "relevant_read" in self.rules:
                rel, other, nr, no = self.catch_up_relevant(me, [(action.app, action.api, action.path, action.ids)])
                if nr:
                    self.event(me, "wal_refused", rule="relevant_read", at="write", call=action.call, unseen=nr,
                               other=no)
                    raise Refused(f"[WAL] Refused: the WAL has {nr} entr{'y' if nr == 1 else 'ies'} you hadn't read about "
                                  f"this record:\n{rel}{self._news(other, no)}\nYou have now seen them. If they don't "
                                  "affect your plan, make the change again; if they do, post a revised intent first. "
                                  "Nothing was changed.")
                if no:
                    self.event(me, "wal_delivered", at="write", n=no)
                    print(self._news(other, no).strip())  # shown in the agent's cell output
            hit = self._conflicting_claim(me, action, claim) if "no_conflict" in self.rules else None
            prior = self.done.get(action.key) if "no_redo" in self.rules else None
            if hit or (prior and prior[0] != me):
                rule, why = (("no_conflict", f"{hit[0]}'s open claim {hit[1].show()} (intent #{hit[1].intent})") if hit
                             else ("no_redo", f"a change {prior[0]} already made (#{prior[1]})"))
                self.event(me, "wal_refused", rule=rule, at="write", call=action.call, reason=why)
                raise Refused(f"[WAL] Refused: {action.call} conflicts with {why}. Nothing was changed."
                              + (f" {self.conflict_hint}" if self.keep_claims else ""))
        if claim is not None and self.keep_claims:
            claim.t = now()  # the claim is in use: its expiry clock restarts
        self._write({"kind": "pending", "agent": me, "call": action.call, "intent": self.intent.get(me)})
        try:
            out = execute()
        except Exception as ex:  # the environment refused it: logged as a change that changed nothing
            self._append(me, "change", intent=self.intent.get(me), call=action.call,
                         result=f"failed: {str(ex).strip().splitlines()[-1][:160] if str(ex).strip() else type(ex).__name__}",
                         app=action.app, api=action.api, path=action.path, ids=list(action.ids))
            raise
        self._append(me, "change", intent=self.intent.get(me), call=action.call, result="done", app=action.app,
                     api=action.api, path=action.path, ids=list(action.ids))
        self.done.setdefault(action.key, (me, len(self.entries)))
        if claim is not None and action.ids and not self.keep_claims:
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
                                                            "relevant_catchup", "relevant_read", "no_conflict", "no_redo",
                                                            "finishing")},
                "wal_delivered": kinds["wal_delivered"],
                **{f"wal_released_{r}": releases[r] for r in ("stopped", "expired", "completed", "abandoned")}}


def team_concurrency(events, logged, accepted):
    """How much a team's agents worked at the same time, and how the log's refusals played out (from its events).

    An agent works from its 'active' event to its 'idle' one. overlap: share of the team's time with two or more
    agents working; parallelism: their summed working time over the team's time (1 = one at a time). With the log
    (logged): a refusal chain is a run of refusals with no accepted post or write in between (accepted(e): the event
    is a write that went through); an agent 'stopped after a refusal' if a refusal was its last log event."""
    span = {}
    for e in events:
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
    if logged:
        chains, last = [], {}
        run = Counter()
        for e in events:
            a = e["agent"]
            if e["kind"] == "wal_refused":
                run[a] += 1
                last[a] = "refused"
            elif e["kind"] == "wal_intend" or accepted(e):
                if run[a]:
                    chains.append(run[a])
                run[a] = 0
                last[a] = "ok"
        chains += [c for c in run.values() if c]
        out.update({"refusal_chains": len(chains), "refusal_chain_max": max(chains, default=0),
                    "refusal_chains_3plus": sum(c >= 3 for c in chains),
                    "stopped_after_refusal": sum(v == "refused" for v in last.values())})
    return out

def unmade_text(unmade):
    """The list of an agent's refused, never-made changes, as the reminders show it."""
    return "\n".join(f"- {u['call']} (refused at {u['t']:.0f}s: {u['why'].removeprefix('[WAL] ')[:120]})" for u in unmade)


UNMADE_REMINDER = ("Before you stop: {n} change{s} you tried {were} refused by the write-ahead log, and you haven't made "
                   "{them} since:\n{list}\nA refused change was not made. If {it_is} still needed, make {them} now (post an "
                   "intent first if needed); if not (for instance a teammate made {them}), you can stop.")


def unmade_reminder(unmade, template=UNMADE_REMINDER):
    n = len(unmade)
    return template.format(n=n, s="" if n == 1 else "s", were="was" if n == 1 else "were", them="it" if n == 1 else "them",
                           it_is="it is" if n == 1 else "they are", list=unmade_text(unmade))


# The protocol as each agent is told it, by level.
TEXT = """The team keeps a write-ahead log (WAL): an append-only record, shared by all of you, of what each agent intends to change and of every change made through the apps' APIs. wal_read shows the entries you haven't seen (all=true shows every entry). wal_intend posts your intent: your plan, and one claim per change you will make, naming the app, the API you will call and its target: the record's id (e.g. {{"app": "spotify", "api": "download_song", "target": "114, 161"}}), or for something new a short description (e.g. {{"app": "spotify", "api": "create_playlist", "target": "playlist 'Road trip'"}}). A new intent replaces your previous one. Every change any of you makes through the APIs is added to the WAL automatically, with its result; reading data and logging in don't need the WAL.
Before changing anything: 1. read the WAL; 2. post your intent; 3. make the change.
{rules}"""
CLAIM = "A change is refused unless your current intent claims it: the same app and API and, for an existing record, its id in the target."
CATCH_UP = "Posting an intent, and making a change, are refused while the WAL has entries you haven't read; you are then shown them."
CATCH_UP_REL = ("Posting an intent, and making a change, are refused while the WAL has entries you haven't read about the "
                "same records (or, for something new, the same API): teammates' intents and changes; you are then shown "
                "them. Other new entries are shown only when you call wal_read.")
CATCH_UP_RELEVANT = ("Posting an intent, and making a change, are refused while the WAL has entries you haven't read about the "
                     "same records (or, for something new, the same API); you are then shown them. Other new entries are shown "
                     "to you with the result instead.")
CONFLICT = ("An intent or a change is also refused if a teammate has an open claim on the same target: the same record, or "
            "the same new item. A claim on an existing record closes once its change is made; any claim is released when "
            "its agent posts a new intent or stops, or after {ttl} seconds.")
CONFLICT_V0 = ("An intent or a change is also refused if it conflicts with a teammate's: a record another agent has claimed "
               "or already changed, or a new item another agent has claimed or already created. Claims are released when "
               "their change is made, when their agent posts a new intent, when it stops, or after {ttl} seconds.")
RULES = {
    "L1": "Nothing checks that you do this: it is up to you.",
    "L2": CLAIM,
    "L3": CLAIM + " " + CATCH_UP_REL,
    "L4": CLAIM + " " + CATCH_UP,
    "L5": CLAIM + " " + CATCH_UP_REL + " " + CONFLICT,
    "L6": CLAIM + " " + CATCH_UP + " " + CONFLICT,
    "L4v0": CLAIM + " " + CATCH_UP_RELEVANT + " " + CONFLICT_V0,
    "L4v0-strict": CLAIM + " " + CATCH_UP + " " + CONFLICT_V0,
}


def text(level, claim_ttl=CLAIM_TTL):
    rules = RULES[level].format(ttl=claim_ttl)
    if level != "L1":
        rules += " A refused change raises an error in your code that says why; nothing is changed."
    return TEXT.format(rules=rules)
