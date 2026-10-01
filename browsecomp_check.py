"""Check BrowseComp-Plus search before running agents: are our query embeddings compatible with the index?

    python browsecomp_check.py      # needs the embedding server (setup.sh with BROWSECOMP=1)

1. Self-retrieval: a document's own opening text, embedded like the index's documents (no prefix), should
   rank that document first.
2. Questions: how often the paper's 100 questions, searched as-is, surface their evidence and gold documents.
"""
import random
import time

import numpy as np

import browsecomp as bc

c = bc.Corpus.get()
print(f"{len(c.text):,} documents, index {c.vectors.shape}, norms {np.linalg.norm(c.vectors[:1000], axis=1).mean():.3f}")
rng = random.Random(0)
for chars in (2000, 6000):
    ids = rng.sample(c.docids, 30)
    q = c.embed([c.text[i][:chars] for i in ids])
    top1 = [c.docids[int(np.argmax(c.vectors @ v))] for v in q]
    print(f"self-retrieval with the first {chars} characters: top-1 is the document itself for {sum(a == b for a, b in zip(ids, top1))}/30")
tasks = bc.load_tasks()
t0 = time.time()
found = {k: [] for k in (5, 100)}
gold = {k: [] for k in (5, 100)}
for t in tasks:
    s = c.vectors @ c.embed([bc.QUERY_PREFIX + t["question"]])[0]
    order = np.argsort(-s)[:100]
    ranked = [c.docids[i] for i in order]
    for k in (5, 100):
        found[k].append(len(set(ranked[:k]) & set(t["evidence_docs"])) / max(1, len(t["evidence_docs"])))
        gold[k].append(len(set(ranked[:k]) & set(t["gold_docs"])) / max(1, len(t["gold_docs"])))
print(f"{len(tasks)} questions searched in {time.time() - t0:.1f}s")
for k in (5, 100):
    print(f"  evidence recall@{k}: {np.mean(found[k]):.1%}   gold recall@{k}: {np.mean(gold[k]):.1%}")
