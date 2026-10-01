"""Compare BrowseComp-Plus runs question by question.

    python analysis/browsecomp_runs.py                 # every BrowseComp run in runs/
    python analysis/browsecomp_runs.py --tag t10       # only runs whose tag starts with t10

Runs split across jobs with --offset/--limit (tags like t10b-s1 … t10b-s5) are merged back into one.
Besides the judge's verdict, each answer gets two cross-checks: the gold answer appears in the response
(answer_in_response, recorded at run time), and a lenient re-grade that accepts the judge's own extracted
answer when it equals the gold answer up to case, punctuation and accents, or contains it as a whole phrase
in a short answer (the 20b judge rejects "FormFactor, Inc." against "FormFactor"). Verdicts are re-read with
browsecomp.parse_verdict, which also accepts the judge's "**Correctness:** yes".

Writes analysis/browsecomp_runs.json with question ids and numbers only: BrowseComp's questions and answers
must not leave /scratch.
"""
import argparse
import sys
import json
import re
import statistics as st
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from browsecomp import parse_verdict  # noqa: E402
NAME = re.compile(r"\d{8}-\d{6}-(?P<kind>.+)-browsecomp-gpt-oss-20b(?:-(?P<tag>[^/]+))?$")


def norm(text):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def lenient(row):
    """The judge's verdict, or its extracted answer matching the gold answer up to formatting."""
    if row["correct"]:
        return True
    m = re.search(r"extracted_final_answer\W*:?\**\s*(.+)", row.get("judgment", ""), re.I)
    extracted, gold = norm(m.group(1) if m else ""), norm(row["expected"])
    if not extracted or not gold or extracted in ("none", "not determined"):
        return False
    return extracted == gold or (f" {gold} " in f" {extracted} " and len(extracted) <= len(gold) + 25)


def load(tag_prefix=None):
    groups = defaultdict(dict)  # (kind, tag without shard) -> {question id: row}
    for d in sorted((ROOT / "runs").iterdir()):
        m = NAME.match(d.name)
        if not m or "smoke" in (m["tag"] or "") or not (d / "results.jsonl").exists():
            continue
        tag = re.sub(r"-s\d+$", "", m["tag"] or "")
        if tag_prefix and not tag.startswith(tag_prefix):
            continue
        for line in open(d / "results.jsonl"):
            r = json.loads(line)
            # Re-read the verdict: runs before 1 October parsed "**Correctness:** yes" as no.
            if r.get("judgment") and r["judgment"] != "empty response":
                r["correct"] = parse_verdict(r["judgment"])
            groups[(m["kind"], tag)][r["id"]] = r
    return groups


def summary(rows):
    rs = list(rows.values())
    n = len(rs)
    out = {"n": n, "correct": sum(bool(r["correct"]) for r in rs), "lenient": sum(lenient(r) for r in rs),
           "answer_in_response": sum(bool(r.get("answer_in_response")) for r in rs),
           "seconds": st.mean(r["seconds"] for r in rs), "tokens": st.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
           "searches": st.mean(r.get("searches", 0) for r in rs), "reads": st.mean(r.get("reads", 0) for r in rs)}
    comms = [r["comm"] for r in rs if r.get("comm")]
    if comms:
        for k in ("messages", "a2a_tasks", "state_writes", "team_answers", "docs_seen_by_several", "repeated_queries"):
            out[k] = st.mean(c.get(k, 0) for c in comms)
        out["questions_with_messages"] = sum(c.get("messages", 0) > 0 for c in comms)
        out["submitted"] = sum(c.get("answer_rule") == "submitted" for c in comms)
        out["voted"] = sum(c.get("answer_rule") == "vote" for c in comms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="only runs whose tag starts with this")
    args = ap.parse_args()
    groups = load(args.tag)
    table = {f"{k}/{t}": summary(rows) for (k, t), rows in groups.items()}
    print(f"{'run':52s} {'n':>3s} {'judge':>6s} {'lenient':>8s} {'match':>6s} {'s/q':>5s} {'tokens':>7s} {'search':>6s} {'msgs':>5s} {'q w/ msgs':>9s} {'team ans':>8s} {'shared docs':>11s}")
    for name, s in sorted(table.items()):
        print(f"{name:52s} {s['n']:3d} {s['correct']:3d} ({s['correct'] / s['n']:.0%})"[:66].ljust(66)
              + f"{s['lenient']:4d} {s['answer_in_response']:6d} {s['seconds']:5.0f} {s['tokens'] / 1000:6.0f}k {s['searches']:6.1f}"
              + (f" {s['messages']:5.1f} {s['questions_with_messages']:9d} {s['team_answers']:8.1f} {s['docs_seen_by_several']:11.1f}"
                 + (f"  submitted {s['submitted']}, voted {s['voted']}" if s["submitted"] or s["voted"] else "") if "messages" in s else ""))
    # Question by question, on the questions every run with 10+ answers has in common.
    full = {f"{k}/{t}": rows for (k, t), rows in groups.items() if len(rows) >= 10}
    common = sorted(set.intersection(*(set(r) for r in full.values())), key=lambda i: int(i.split("-")[1])) if full else []
    if common and len(common) <= 30:
        names = sorted(full)
        print(f"\n{len(common)} questions in every run with 10+ answers (judge verdict; L = only the lenient re-grade accepts it)")
        print(f"{'question':10s}" + "".join(f"{n[:20]:>21s}" for n in names))
        for i in common:
            cells = []
            for n in names:
                r = full[n][i]
                cells.append("yes" if r["correct"] else "L" if lenient(r) else ".")
            print(f"{i:10s}" + "".join(f"{c:>21s}" for c in cells))
    out = {"runs": table, "per_question": {n: {i: {"correct": bool(r["correct"]), "lenient": lenient(r)} for i, r in rows.items()}
                                          for n, rows in ((f"{k}/{t}", rows) for (k, t), rows in groups.items())}}
    (ROOT / "analysis" / "browsecomp_runs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
