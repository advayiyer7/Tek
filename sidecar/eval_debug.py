"""Dissect retrieve() stage-by-stage for failing stress-eval queries.

Builds the same corpus as eval_stress.py into a persistent dir (reused across
runs) and prints vector hits, FTS hits, fused pool, and CE scores per query.

Run:  .venv/Scripts/python eval_debug.py "query 1" "query 2" ...
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from eval_retrieval import CORPUS as CORE_CORPUS
from eval_stress import CLUSTER_CORPUS, EDGE_TEXT_FILES, build_filler, build_long_docs
from tek.config import Config
from tek.embed import FastEmbedEmbedder
from tek.indexer import Indexer
from tek.rag import retrieve
from tek.rerank import Reranker
from tek.store import Store

WORK = Path(__file__).parent / ".debug_stress"


def main() -> int:
    queries = sys.argv[1:] or ["who is my landlord"]
    rng = random.Random(7)
    corpus_dir = WORK / "corpus"
    data_dir = WORK / "data"

    corpus: dict[str, str] = {}
    corpus.update(CORE_CORPUS)
    corpus.update(CLUSTER_CORPUS)
    corpus.update(EDGE_TEXT_FILES)
    corpus.update(build_long_docs(rng))
    corpus.update(build_filler(110, (1, 3), rng))
    if not corpus_dir.exists():
        for rel, content in corpus.items():
            f = corpus_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")

    config = Config(data_dir)
    config.update(folders=[str(corpus_dir)])
    embedder = FastEmbedEmbedder(config.settings.embed_model, str(config.models_dir))
    embedder.ensure_loaded()
    store = Store(config.db_dir, dim=embedder.dim)
    if store.stats()["chunks"] == 0:
        indexer = Indexer(config=config, embedder=embedder, store=store)
        indexer.start_full_index()
        while indexer.running:
            time.sleep(0.2)
        assert indexer.progress.state == "done", indexer.progress.error
    print(f"index: {store.stats()}")

    reranker = Reranker(config.settings.rerank_model, str(config.models_dir))
    reranker.rerank("warmup", ["warmup"])

    for q in queries:
        print(f"\n================ {q!r}")
        vec = embedder.embed_query(q)
        t0 = time.perf_counter()
        vec_hits = store.search(vec, k=24)
        t_vec = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        fts_hits = store.fts_search(q, vec, k=24)
        t_fts = (time.perf_counter() - t0) * 1000
        print(f"  vector top-8 ({t_vec:.0f}ms):")
        for h in vec_hits[:8]:
            print(f"    cos {h['score']:.3f}  {Path(h['path']).name}#{h['chunk_index']}")
        print(f"  fts top-8 ({t_fts:.0f}ms):")
        for h in fts_hits[:8]:
            print(f"    cos {h['score']:.3f}  {Path(h['path']).name}#{h['chunk_index']}")
        t0 = time.perf_counter()
        hits = retrieve(store, embedder, q, k=10, reranker=reranker)
        t_all = (time.perf_counter() - t0) * 1000
        print(f"  retrieve() top ({t_all:.0f}ms total):")
        for h in hits[:6]:
            print(f"    ce {h.get('rerank', float('nan')):.4f} cos {h['score']:.3f}  {Path(h['path']).name}#{h['chunk_index']}")
        if not hits:
            # replicate the pre-rerank pool to see what the CE rejected
            pool: dict[tuple[str, int], dict] = {}
            for rank, h in enumerate(vec_hits):
                e = pool.setdefault((h["path"], h["chunk_index"]), {**h, "rrf": 0.0, "fts_rank": None})
                e["rrf"] += 1.0 / (60 + rank + 1)
            for rank, h in enumerate(fts_hits):
                e = pool.setdefault((h["path"], h["chunk_index"]), {**h, "rrf": 0.0, "fts_rank": None})
                e["rrf"] += 1.0 / (60 + rank + 1)
                if e["fts_rank"] is None:
                    e["fts_rank"] = rank
            cands = sorted(pool.values(), key=lambda c: c["rrf"], reverse=True)
            cands = [c for c in cands if c["score"] >= 0.30 or (c["fts_rank"] is not None and c["fts_rank"] < 3)][:16]
            probs = reranker.rerank(q, [c["text"] for c in cands]) or []
            print("  (empty) pre-rerank pool with CE scores:")
            for c, p in sorted(zip(cands, probs), key=lambda x: -x[1])[:8]:
                print(f"    ce {p:.5f} cos {c['score']:.3f}  {Path(c['path']).name}#{c['chunk_index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
