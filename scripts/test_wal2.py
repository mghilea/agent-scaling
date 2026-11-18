# Run with AppWorld's Python: $APPWORLD_PYTHON scripts/test_wal2.py
"""Unit tests for wal2's ladder: L6 (full catch-up, open-claim conflicts), L3/L5 (relevant catch-up only), the earlier
L4v0, and the clock fix."""
import sys, threading, time, tempfile
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import freezegun
import wal2

class Adapter:
    def is_write(self, app, api): return api in ("like_song", "create_playlist", "send_payment")
    def has_ids(self, app, api): return api == "like_song"
    def path(self, app, api): return {"like_song": "/songs/{song_id}/like"}.get(api, f"/{api}")

def team(level, ttl=wal2.CLAIM_TTL):
    lock = threading.RLock()
    log = wal2.Log(level, Adapter(), tempfile.mktemp(suffix=".jsonl"), lock, lambda *a, **k: None, claim_ttl=ttl)
    def write(me, action):
        with lock:
            try:
                return log.commit(me, action, lambda: "ok")
            except wal2.Refused as ex:
                return str(ex)
    return log, write

like = lambda i: wal2.Action("spotify", "like_song", "/songs/{song_id}/like", ids=(str(i),), call=f"like_song({i})")
playlist = lambda name: wal2.Action("spotify", "create_playlist", "/create_playlist", identity=(name,), call=f"create_playlist({name!r})")
ok = lambda r: r.startswith("Posted") or r == "ok"
def deep(f, n=8):  # call f() n frames down, past freezegun's 5-frame look-up
    return f() if n == 0 else deep(f, n - 1)

with freezegun.freeze_time("2023-05-18 12:00:00"):   # as AppWorld does while a world is open
    assert deep(time.time) < 1.7e9 and deep(wal2.now) > 1.7e9, "the log must see the real clock"

    # L6 (was named L4 on 2026-11-17)
    log, write = team("L6")
    assert ok(log.intend("a1", "like 54", [{"app": "spotify", "api": "like_song", "target": "54"}]))
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "hadn't read" in r, r                          # full catch-up before posting
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "open claim" in r, r                           # a teammate's open claim on the same record
    assert ok(write("a1", like(54)))                      # a1's claim on 54 closes
    log.read("a2")
    assert ok(log.intend("a2", "like 54 again", [{"app": "spotify", "api": "like_song", "target": "54"}])), "closed claim frees the target"
    assert ok(write("a2", like(54))), "no 'already done' check any more"
    log.read("a1")
    assert ok(log.intend("a1", "playlist", [{"app": "spotify", "api": "create_playlist", "target": "playlist 'Road trip'"}]))
    log.read("a2")
    r = log.intend("a2", "same playlist", [{"app": "spotify", "api": "create_playlist", "target": "Playlist Road Trip"}])
    assert "open claim" in r, r                           # the same new item (normalized text)
    assert ok(log.intend("a2", "another", [{"app": "spotify", "api": "create_playlist", "target": "playlist 'Road trip 2'"}]))
    r = write("a2", like(99))
    assert "doesn't claim" in r, r                        # claim required
    log.read("a1")
    log.read("a3"); assert ok(log.intend("a3", "unrelated", [{"app": "spotify", "api": "send_payment", "target": "rent"}]))
    r = write("a1", playlist("Road trip"))
    assert "hadn't read" in r, r                          # full catch-up before writing, even for unrelated entries
    assert ok(write("a1", playlist("Road trip")))

    # L3: catch-up on relevant entries only; unrelated entries are neither required nor shown, until wal_read
    claim = lambda i: [{"app": "spotify", "api": "like_song", "target": str(i)}]
    for level in ("L3", "L5"):
        log, write = team(level)
        assert ok(log.intend("a1", "like 54", claim(54)))                                        # 1
        assert ok(log.intend("a3", "pay", [{"app": "spotify", "api": "send_payment", "target": "rent"}])), \
            "unread but unrelated: no catch-up needed"                                        # 2
        r = log.intend("a2", "like 54 too", claim(54))
        assert "hadn't read" in r and "#1 a1" in r and "#2" not in r, r                     # only the relevant entry
        r = log.intend("a2", "like 54 too", claim(54))
        if level == "L5":
            assert "open claim" in r, r                                                       # L5: a1 holds 54
            assert ok(log.intend("a2", "like 55", claim(55))), "a different record is fine"  # 3
            assert ok(write("a2", like(55)))                                                  # 4: #2 still unread, unrelated
            out = log.read("a2")
            assert "#2 a3" in out and "#1 a1" not in out and "#3" not in out and "#4" not in out, out
            continue
        assert ok(r), r                                                                       # 3: L3 has no conflict rule
        assert ok(write("a2", like(54))), "a2 has read what concerns 54; #2 is unrelated"   # 4
        r = write("a1", like(54))
        assert "hadn't read" in r and "#3 a2" in r and "#4 a2" in r and "#2" not in r, r     # write-time catch-up
        assert ok(write("a1", like(54)))                                                      # 5
        out = log.read("a2")
        assert "#2 a3" in out and "#5 a1" in out, out                                        # unread entries, now shown
        assert "#1 a1" not in out and "#3" not in out and "#4" not in out, out               # not repeated, not its own
        assert log.read("a2") == "No new entries since you last read the WAL."
        out = log.read("a1")
        assert "#2 a3" in out and "#3" not in out and "#4" not in out and "#5" not in out, out
    # a relevant refusal is counted under its own rule
    assert log.counts([{"kind": "wal_refused", "rule": "relevant_catchup"}])["wal_refused_relevant_catchup"] == 1

    # earlier L4v0: relevant catch-up, and work already done is refused (no_redo)
    log, write = team("L4v0")
    log.intend("a1", "like 54", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert ok(write("a1", like(54)))
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "hadn't read" in r, r                          # relevant unread entries block
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "already made" in r, r

    # claims expire on the real clock while AppWorld's is frozen
    log, write = team("L6", ttl=0.2)
    log.intend("a1", "like 7", [{"app": "spotify", "api": "like_song", "target": "7"}])
    wal2_real_sleep = lambda s: (lambda t0: [None for _ in iter(lambda: deep(wal2.now) - t0 < s, False)])(deep(wal2.now))
    wal2_real_sleep(0.3)
    log.read("a2")
    assert any(e["kind"] == "release" and e["reason"] == "expired" for e in log.entries), "claim should expire"
    assert all(0 <= e["t"] < 60 for e in log.entries), [e["t"] for e in log.entries]
    r = log.intend("a2", "like 7", [{"app": "spotify", "api": "like_song", "target": "7"}])
    assert ok(r), r
# L3 with CooperBench's file claims (the same adapter as cooperbench_team.FileAdapter): relevance is the same file
class Files:
    keep_claims = True
    def is_write(self, app, api): return (app, api) == ("repo", "edit")
    def has_ids(self, app, api): return True
    def path(self, app, api): return "file"
    def ids_in(self, app, api, target): return {target.strip()}
    def unclaimed(self, action): return f"the file {action.ids[0]}"
lock = threading.RLock()
flog = wal2.Log("L3", Files(), tempfile.mktemp(suffix=".jsonl"), lock, lambda *a, **k: None)
edit = lambda f: wal2.Action("repo", "replace_in_file", "file", ids=(f,), call=f"replace_in_file {f}")
def fwrite(me, action):
    with lock:
        try:
            return flog.commit(me, action, lambda: "ok")
        except wal2.Refused as ex:
            return str(ex)
fclaim = lambda *fs: [{"app": "repo", "api": "edit", "target": f} for f in fs]
assert ok(flog.intend("a1", "core", fclaim("src/core.py")))                      # 1
assert ok(flog.intend("a3", "docs", fclaim("README.md")))                        # 2
assert ok(fwrite("a1", edit("src/core.py"))) and ok(fwrite("a1", edit("src/core.py")))   # 3, 4: one claim, two edits
r = flog.intend("a2", "core too", fclaim("src/core.py", "src/util.py"))
assert "hadn't read" in r and "#1 a1" in r and "#3 a1" in r and "#4 a1" in r and "#2" not in r, r
assert ok(flog.intend("a2", "core too", fclaim("src/core.py", "src/util.py")))   # 5
assert ok(fwrite("a3", edit("README.md"))), "a3: nothing unread about README.md"
assert ok(fwrite("a2", edit("src/util.py")))
out = flog.read("a2")
assert "#2 a3" in out and "#6 a3" in out and "#1 a1" not in out, out
# refused changes are remembered as unmade until the agent makes a change to the same target
log, write = team("L6")
assert ok(log.intend("a1", "like 54", [{"app": "spotify", "api": "like_song", "target": "54"}]))
log.read("a2")
assert "doesn't claim" in write("a2", like(54)) and "doesn't claim" in write("a2", like(55))
assert [u["call"] for u in log.unmade("a2")] == ["like_song(54)", "like_song(55)"], log.unmade("a2")
assert log.unmade("a1") == []
log.read("a2"); assert ok(log.intend("a2", "like 55", [{"app": "spotify", "api": "like_song", "target": "55"}]))
assert ok(write("a2", like(55)))
assert [u["call"] for u in log.unmade("a2")] == ["like_song(54)"], "the same target made later clears it"
assert "like_song(54)" in wal2.unmade_reminder(log.unmade("a2")) and "1 change you tried was refused" in wal2.unmade_reminder(log.unmade("a2"))
print("all wal2 tests passed")

# finishing: no new intents; only writes the agent had claimed
log, write = team("L1")
log.intend("a2", "like 55", [{"app": "spotify", "api": "like_song", "target": "55"}])
log.finishing = True
assert "No new intents" in log.intend("a2", "more", [{"app": "spotify", "api": "like_song", "target": "56"}])
assert "only changes you had already claimed" in write("a2", like(56))
assert ok(write("a2", like(55)))
print("finishing tests passed")
