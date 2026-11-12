"""Generate figure 1 (architecture and write path) and figure 4 (completion) of analysis/wal_design.html as SVG fragments; paste them into the page (they use its CSS classes)."""
from pathlib import Path

def T(x, y, text, cls, anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{text}</text>'
def R(x, y, w, h, cls="p-box", rx=3):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
def L(x1, y1, x2, y2, cls="p-edge", end="pa", start=None):
    s = f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
    if end: s += f' marker-end="url(#{end})"'
    if start: s += f' marker-start="url(#{start})"'
    return s + "/>"
def N(x, y, n, ref=False):
    return (f'<circle class="{"p-num-ref" if ref else "p-num"}" cx="{x}" cy="{y}" r="8.5"/>'
            f'<text class="t-num" x="{x}" y="{y + 3.6}" text-anchor="middle">{n}</text>')
def cyl(cx, cy, w=34, h=30, cls="p-fill"):
    rx, ry = w / 2, 5
    return (f'<path class="{cls}" d="M{cx-rx},{cy-h/2} V{cy+h/2} A{rx},{ry} 0 0,0 {cx+rx},{cy+h/2} V{cy-h/2}"/>'
            f'<ellipse class="{cls}" cx="{cx}" cy="{cy-h/2}" rx="{rx}" ry="{ry}"/>')
def markers(*ids):
    out = []
    for i, cls in ids:
        out.append(f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>')
    return "<defs>" + "".join(out) + "</defs>"

# ---- figure 1 ------------------------------------------------------------------------------------------------------
f = ['<svg viewBox="0 0 1080 560" role="img" aria-label="Architecture of the write-ahead log. Three agent threads have no channel to each other; each calls wal_intend and wal_read on the log and execute_python on the code runner. In the harness process, the code runner sends reads straight to the AppWorld app databases and every write to the gate, commit, which checks the level rules under the log mutex, records a pending line, applies the write and appends a change entry, all forced to the log file on disk.">',
     markers(("pa", "arrowhead"), ("pr", "arrowhead-ref"))]
# agent threads
f += [R(16, 44, 144, 470, "p-bound", 6), T(28, 64, "Agent threads", "t-bound")]
for i, cy in enumerate([128, 278, 428]):
    f += [R(36, cy - 28, 104, 56), T(88, cy - 2, f"Agent {i + 1}", "t-h", "middle"), T(88, cy + 15, f"a{i + 1}", "t-k", "middle")]
for y0, y1 in [(156, 250), (306, 400)]:
    m = (y0 + y1) / 2
    f += [L(88, y0, 88, y1, "p-none", None), R(56, m - 12, 64, 30, "p-mask", 0), T(88, m + 1, "×", "t-x", "middle"),
          T(88, m + 15, "no channel", "t-v", "middle")]
# harness process
f += [R(318, 20, 538, 520, "p-bound", 6), T(330, 40, "Harness process (one per task)", "t-bound")]
# write-ahead log
f += [R(330, 60, 300, 150, "p-key"), T(344, 82, "Write-ahead log", "t-h"), T(470, 82, "append-only", "t-bound")]
rows = [("entries", "intent · change · release"), ("cursor[a]", "last entry agent a has read"),
        ("claims[a]", "open claims {app, api, target}"), ("mutex", "one lock: reads, posts, commits")]
for j, (k, v) in enumerate(rows):
    f += [T(344, 108 + j * 24, k, "t-k"), T(420, 108 + j * 24, v, "t-v")]
# gate
f += [R(330, 258, 300, 92, "p-key"), T(344, 280, "Gate", "t-h"), T(384, 280, "commit(a, op)", "t-api"),
      T(344, 304, "level rules: claim · catch-up · conflict", "t-v"), T(344, 324, "pending line → apply op → change entry", "t-v"),
      T(344, 342, "complete_task: logged, never checked", "t-v")]
# code runner
f += [R(330, 410, 300, 108), T(344, 432, "Code runner", "t-h"), T(446, 432, "(main thread)", "t-bound")]
for j in range(4):
    f += [R(344 + j * 30, 446, 24, 18, "p-fill", 2)]
f += [T(470, 459, "FIFO, one cell at a time", "t-v"), T(344, 486, "each cell runs in its agent's shell;", "t-v"),
      T(344, 504, "API calls dispatched by HTTP method", "t-v")]
# AppWorld world
f += [R(696, 150, 144, 368), T(710, 172, "AppWorld world", "t-h")]
for j, name in enumerate(["spotify", "venmo", "gmail", "…"]):
    cy = 214 + j * 62
    f += [cyl(728, cy), T(756, cy + 4, name, "t-k")]
f += [T(710, 476, "9 app databases", "t-v"), T(710, 494, "457 APIs, in-process", "t-v")]
# log file
f += [cyl(960, 112, 64, 52), T(960, 160, "wal/&lt;task&gt;.jsonl", "t-k", "middle"), T(960, 177, "JSON Lines, fsync per line", "t-v", "middle")]
# edges: agents <-> log
f += [L(160, 108, 326, 108), T(238, 100, "wal_intend(plan, claims)", "t-api", "middle"),
      L(330, 146, 164, 146), T(240, 162, "wal_read() → entries", "t-api", "middle")]
# agents <-> code runner
f += [L(160, 446, 326, 446), T(228, 436, "execute_python(code)", "t-api", "middle"), N(300, 446, "1"),
      L(330, 486, 164, 486), T(228, 504, "stdout | Refused", "t-api", "middle"), N(300, 486, "7")]
# code runner -> gate (writes), gate -> code runner (refused)
f += [L(364, 410, 364, 354), N(364, 381, "2"), T(378, 385, "POST · PUT · PATCH · DELETE", "t-api"),
      L(604, 352, 604, 406, "p-edge-ref", "pr"), T(612, 385, "Refused", "t-vref")]
# gate <-> log
f += [L(480, 256, 480, 214, "p-edge", "pa", "pa"), N(480, 235, "3"), T(494, 239, "check · append", "t-v")]
# gate -> app DB ; code runner -> app DB (reads)
f += [L(630, 300, 692, 300), N(661, 300, "5"), L(630, 470, 692, 470), T(661, 462, "GET", "t-api", "middle")]
# log -> disk
f += [L(630, 112, 924, 112), N(700, 112, "4"), N(722, 112, "6"), T(800, 104, "append + fsync", "t-v", "middle")]
# key to the write path
key = [("1", "cell queued"), ("2", "write → commit(a, op)"), ("3", "rules checked, under mutex"), ("4", "pending line, fsync"),
       ("5", "op applied to the app DB"), ("6", "change entry, fsync"), ("7", "result, or Refused")]
f += [T(872, 250, "Write path", "t-h")]
for j, (n, text) in enumerate(key):
    f += [N(882, 274 + j * 26, n), T(898, 278 + j * 26, text, "t-v")]
f += [T(872, 466, "Reads (GET) go from the", "t-v"), T(872, 482, "code runner straight to", "t-v"), T(872, 498, "the app DB: no ②–⑥.", "t-v")]
f.append("</svg>")

# ---- figure 4 ------------------------------------------------------------------------------------------------------
g = ['<svg viewBox="0 0 1000 230" role="img" aria-label="Agent a calls complete_task. If no teammate holds an open claim, the call runs and the task is graded. Otherwise the call is recorded and finishing starts: the caller and agents without claims stop, claim holders may only make their claimed writes and cannot post intents. When no open claims remain, the recorded call runs and the task is graded.">',
     markers(("pb", "arrowhead"))]
g += [R(10, 88, 170, 64), T(95, 114, "complete_task(answer)", "t-api", "middle"), T(95, 134, "called by agent a", "t-v", "middle")]
g += ['<path class="p-box" d="M310,64 L400,120 L310,176 L220,120 Z"/>', T(310, 116, "open claims held", "t-v", "middle"), T(310, 132, "by teammates?", "t-v", "middle")]
g += [L(180, 120, 216, 120, end="pb")]
g += [f'<path class="p-edge" d="M310,64 V30 H836" marker-end="url(#pb)"/>', T(318, 52, "no", "t-v")]
g += [L(400, 120, 436, 120, end="pb"), T(406, 112, "yes", "t-v")]
g += [R(440, 80, 190, 80), T(454, 102, "Record the call", "t-h"), T(454, 122, "finishing := true; caller and", "t-v"), T(454, 140, "claim-less agents stop", "t-v")]
g += [L(630, 120, 666, 120, end="pb")]
g += [R(670, 80, 190, 80), T(684, 102, "Claim holders finish", "t-h"), T(684, 122, "only claimed writes;", "t-v"), T(684, 140, "wal_intend refused", "t-v")]
g += [f'<path class="p-edge" d="M860,120 H900 V52" marker-end="url(#pb)"/>', T(908, 92, "claims closed,", "t-v"), T(908, 108, "released or", "t-v"), T(908, 124, "expired (90 s)", "t-v")]
g += [R(840, 10, 150, 40, "p-key"), T(915, 35, "Run call; grade", "t-h", "middle")]
g += [T(10, 205, "Without the log (L0), the first complete_task runs at once and ends the task.", "t-v")]
g.append("</svg>")

out = Path(__file__).resolve().parent / "figures"
(out / "fig1.svg.part").write_text("\n".join(f)); (out / "fig4.svg.part").write_text("\n".join(g)); print("ok")
