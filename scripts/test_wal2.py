# Run with AppWorld's Python: $APPWORLD_PYTHON scripts/test_wal2.py
"""Unit tests for wal2's new L4 (open-claim conflicts only, full catch-up), the earlier L4v0, and the clock fix."""
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

    # new L4
    log, write = team("L4")
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

    # earlier L4v0: relevant catch-up, and work already done is refused (no_redo)
    log, write = team("L4v0")
    log.intend("a1", "like 54", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert ok(write("a1", like(54)))
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "hadn't read" in r, r                          # relevant unread entries block
    r = log.intend("a2", "like 54 too", [{"app": "spotify", "api": "like_song", "target": "54"}])
    assert "already made" in r, r

    # claims expire on the real clock while AppWorld's is frozen
    log, write = team("L4", ttl=0.2)
    log.intend("a1", "like 7", [{"app": "spotify", "api": "like_song", "target": "7"}])
    wal2_real_sleep = lambda s: (lambda t0: [None for _ in iter(lambda: deep(wal2.now) - t0 < s, False)])(deep(wal2.now))
    wal2_real_sleep(0.3)
    log.read("a2")
    assert any(e["kind"] == "release" and e["reason"] == "expired" for e in log.entries), "claim should expire"
    assert all(0 <= e["t"] < 60 for e in log.entries), [e["t"] for e in log.entries]
    r = log.intend("a2", "like 7", [{"app": "spotify", "api": "like_song", "target": "7"}])
    assert ok(r), r
print("all wal2 tests passed")

# finishing: no new intents; only writes the agent had claimed
log, write = team("L1")
log.intend("a2", "like 55", [{"app": "spotify", "api": "like_song", "target": "55"}])
log.finishing = True
assert "No new intents" in log.intend("a2", "more", [{"app": "spotify", "api": "like_song", "target": "56"}])
assert "only changes you had already claimed" in write("a2", like(56))
assert ok(write("a2", like(55)))
print("finishing tests passed")
