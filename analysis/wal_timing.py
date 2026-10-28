"""How long the write-ahead log's atomic (locked) operations take, for the concurrency page.

    python analysis/wal_timing.py   # writes analysis/wal_timing.json (about a minute; needs WorkBench)

Times the real wal.Log on a real WorkBench sandbox, single-threaded (so with no lock contention): posting an
intent when caught up, reading one new entry (with 10 and 100 entries in the log), and a gated write (lock,
check, WorkBench's update_customer, log entry) against the same WorkBench call without the log.
"""
import json
import socket
import statistics as st
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import wal  # noqa: E402
import workbench as wb  # noqa: E402

RES = [{"resource": "crm", "access": "update", "target": "00000187"},
       {"resource": "email", "access": "create", "target": "to lena"}]


def summary(seconds, unit):
    s = sorted(seconds)
    return {"median": st.median(s) * unit, "p95": s[int(0.95 * len(s))] * unit, "n": len(s)}


def log_ops(size, n=2000):
    """Post and read times with `size` entries already in the log."""
    log = wal.Log()
    me, other = log.tools("agent_1"), log.tools("agent_2")
    for i in range(size):
        other["wal_append"][0](plan=f"plan {i}", resources=RES)
    me["wal_read"][0](all=True)
    post, read = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        me["wal_append"][0](plan="update a lead", resources=RES)  # caught up, so it is appended
        post.append(time.perf_counter() - t0)
        other["wal_append"][0](plan="x", resources=RES)  # one new entry for agent_1 (agent_2 is behind: refused)
        other["wal_read"][0]()
        other["wal_append"][0](plan="x", resources=RES)
        t0 = time.perf_counter()
        me["wal_read"][0]()
        read.append(time.perf_counter() - t0)
    return {"entries": size, "post_us": summary(post, 1e6), "read_us": summary(read, 1e6)}


def write(with_log, n=300):
    times = []
    for _ in range(n):
        sandbox = wb.Sandbox()
        if with_log:
            log = wal.Log()
            log.tools("agent_1")["wal_append"][0](plan="p", resources=[{"resource": "crm", "access": "update", "target": "x"}])
            fn = log.guard("agent_1", sandbox)["customer_relationship_manager_update_customer"][0]
        else:
            fn = sandbox.tools["customer_relationship_manager_update_customer"][0]
        t0 = time.perf_counter()
        fn(customer_id="00000187", field="status", new_value="Lost")
        times.append(time.perf_counter() - t0)
    return summary(times, 1e3)


def main():
    data = {"host": socket.gethostname().split(".")[0], "date": str(date.today()),
            "log": [log_ops(10), log_ops(100)],
            "write_ms": write(True), "write_without_log_ms": write(False)}
    (ROOT / "analysis" / "wal_timing.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
