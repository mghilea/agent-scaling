"""Why some BrowseComp-Plus answer pages are never found: where they rank for the queries agents actually issued.

    python analysis/browsecomp_retrieval.py bc-116 bc-100 bc-228 bc-235          # needs the embedder (EMBED_PORT)
    python analysis/browsecomp_retrieval.py --bm25 bc-116 bc-100 bc-228 bc-235   # also keyword (BM25) and hybrid search

For each question: every distinct search query any agent issued in any run, re-ranked over the whole
collection with the same dense search; the best rank any answer (gold) page reached, and how many queries
put one in the top 5 (what search returns), 20, 100 and 1000. Also the rank with the whole question as the
query. With --bm25, the same for keyword search (BM25, k1=0.9 b=0.4 as in Pyserini, over hashed word counts; the
index is cached in $BROWSECOMP_DIR) and for a hybrid that fuses the two rankings (reciprocal rank fusion, k=60).
Prints ids, counts and ranks only.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import browsecomp as bc  # noqa: E402


def queries_for(qid):
    out = set()
    for d in (ROOT / "runs").glob("*-browsecomp-gpt-oss-20b-*"):
        f = d / "traces" / f"{qid}.json"
        if "smoke" in d.name or "invalid" in d.name or not f.exists():
            continue
        t = json.load(open(f))
        convs = {"single": t} if isinstance(t, list) else t.get("conversations", {})
        for msgs in convs.values():
            for m in msgs:
                for tc in m.get("tool_calls") or []:
                    if tc["function"]["name"].split("<|")[0] == "search_documents":
                        try:
                            q = json.loads(tc["function"]["arguments"] or "{}").get("query")
                        except json.JSONDecodeError:
                            q = None
                        if isinstance(q, str) and q.strip():
                            out.add(q.strip())
    return sorted(out)


def ranks(corpus, texts, targets):
    """Best rank (1-based) of any target page for each text, over the whole collection."""
    rows = np.array([corpus.docids.index(t) for t in targets if t in corpus.docids])
    best = []
    for i in range(0, len(texts), 32):
        q = corpus.embed([bc.QUERY_PREFIX + x for x in texts[i:i + 32]])
        scores = q @ corpus.vectors.T                       # (batch, docs)
        target = scores[:, rows].max(axis=1, keepdims=True)
        best += list((scores > target).sum(axis=1) + 1)
    return np.array(best)


def bm25_index(corpus):
    """Documents x terms BM25 weights, as a CSR matrix in the corpus's docid order (built once, cached)."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer
    cache = bc.BROWSECOMP_DIR / "bm25_hashed.npz"
    hv = HashingVectorizer(n_features=2 ** 21, alternate_sign=False, norm=None, stop_words="english", dtype=np.float32)
    if cache.exists():
        return hv, sp.load_npz(cache).tocsr()
    parts = []
    for i in range(0, len(corpus.docids), 2000):
        parts.append(hv.transform([corpus.text.get(d, "") for d in corpus.docids[i:i + 2000]]))
    tf = sp.vstack(parts).tocsr()
    k1, b = 0.9, 0.4
    dl = np.asarray(tf.sum(axis=1)).ravel()
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    n = tf.shape[0]
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(tf.indptr))
    t = tf.data
    tf.data = idf[tf.indices] * t * (k1 + 1) / (t + k1 * (1 - b + b * dl[rows] / dl.mean()))
    sp.save_npz(cache, tf)
    return hv, tf


def all_ranks(scores):
    """For each row of scores (queries x docs), the rank of every doc (1 = best)."""
    order = np.argsort(-scores, axis=1)
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, scores.shape[1] + 1)[None, :].repeat(scores.shape[0], 0), axis=1)
    return r


def ranks3(corpus, hv, W, texts, targets):
    """Best rank of any target page for each text: dense, BM25, and their reciprocal-rank fusion."""
    rows = np.array([corpus.docids.index(t) for t in targets if t in corpus.docids])
    out = {"dense": [], "bm25": [], "hybrid": []}
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        dense = corpus.embed([bc.QUERY_PREFIX + x for x in batch]) @ corpus.vectors.T
        q = hv.transform(batch)
        q.data[:] = 1.0
        bm = np.asarray((W @ q.T).todense()).T
        rd, rb = all_ranks(dense), all_ranks(bm)
        fused = 1 / (60 + rd) + 1 / (60 + rb)
        for name, sc in (("dense", dense), ("bm25", bm), ("hybrid", fused)):
            target = sc[:, rows].max(axis=1, keepdims=True)
            out[name] += list((sc > target).sum(axis=1) + 1)
    return {k: np.array(v) for k, v in out.items()}


def main():
    tasks = {t["id"]: t for t in bc.load_tasks()}
    corpus = bc.Corpus.get()
    if "--bm25" in sys.argv:
        sys.argv.remove("--bm25")
        hv, W = bm25_index(corpus)
        print(f"{'question':9s} {'search':7s} {'queries':>7s} {'best':>5s} {'top 5':>6s} {'top 20':>6s} {'top 100':>7s} {'median':>7s} {'whole question':>14s}")
        for qid in sys.argv[1:]:
            t = tasks[qid]
            qs = queries_for(qid)
            r = ranks3(corpus, hv, W, qs, t["gold_docs"])
            w = ranks3(corpus, hv, W, [t["question"]], t["gold_docs"])
            for name in ("dense", "bm25", "hybrid"):
                x = r[name]
                print(f"{qid:9s} {name:7s} {len(qs):7d} {x.min():5d} {(x <= 5).sum():6d} {(x <= 20).sum():6d} {(x <= 100).sum():7d} "
                      f"{int(np.median(x)):7d} {w[name][0]:14d}")
        return
    print(f"{'question':9s} {'answer pages':>12s} {'queries':>7s} {'best rank':>9s} {'top 5':>6s} {'top 20':>6s} {'top 100':>7s} {'top 1000':>8s} "
          f"{'median rank':>11s} {'whole question':>14s}")
    for qid in sys.argv[1:]:
        t = tasks[qid]
        qs = queries_for(qid)
        r = ranks(corpus, qs, t["gold_docs"])
        whole = ranks(corpus, [t["question"]], t["gold_docs"])[0]
        print(f"{qid:9s} {len(t['gold_docs']):12d} {len(qs):7d} {r.min():9d} {(r <= 5).sum():6d} {(r <= 20).sum():6d} {(r <= 100).sum():7d} "
              f"{(r <= 1000).sum():8d} {int(np.median(r)):11d} {whole:14d}")


if __name__ == "__main__":
    main()
