"""Mine hard negatives by running Tek's REAL hybrid retriever over the training
corpus, no reranker. For each query we keep:
  - positive: the gold doc's chunk the fusion pool surfaces (or its first chunk
    if retrieval missed it) — exactly the text the serving reranker would score
  - hard negatives: the top non-gold chunks fusion ranked above/around it —
    the topically-confusable passages the reranker must push down

Output pairs.jsonl: {anchor_id, path, topic, style, query, doc, label}. Resumable.
"""

from __future__ import annotations

import json
import sys
import time

from .config import (CORPUS_DIR, DATA, EMBED_MODEL, EVAL_MODELS, NEG_PER_POS,
                     PAIRS_PATH, QUERIES_PATH, add_sidecar_to_path)

add_sidecar_to_path()

from tek.chunk import chunk_text  # noqa: E402
from tek.config import Config  # noqa: E402
from tek.embed import FastEmbedEmbedder  # noqa: E402
from tek.indexer import Indexer  # noqa: E402
from tek.rag import CANDIDATE_K, RRF_K  # noqa: E402
from tek.store import Store  # noqa: E402

MINE_POOL = 30  # fused candidates inspected per query


def fuse(store: Store, embedder: FastEmbedEmbedder, query: str) -> list[dict]:
    """Fusion pool BEFORE rerank/floor/per-file-cap — the raw candidate set."""
    vector = embedder.embed_query(query)
    vec_hits = store.search(vector, k=CANDIDATE_K)
    fts_hits = store.fts_search(query, vector, k=CANDIDATE_K)
    pool: dict[tuple[str, int], dict] = {}
    for rank, hit in enumerate(vec_hits):
        key = (hit["path"], hit["chunk_index"])
        pool.setdefault(key, {**hit, "rrf": 0.0})["rrf"] += 1.0 / (RRF_K + rank + 1)
    for rank, hit in enumerate(fts_hits):
        key = (hit["path"], hit["chunk_index"])
        pool.setdefault(key, {**hit, "rrf": 0.0})["rrf"] += 1.0 / (RRF_K + rank + 1)
    return sorted(pool.values(), key=lambda c: c["rrf"], reverse=True)[:MINE_POOL]


def _gold_text(query: str, gold_path: str, cands: list[dict], embedder, store) -> str | None:
    abs_gold = str((CORPUS_DIR / gold_path))
    gold = [c for c in cands if c["path"] == abs_gold]
    if gold:
        return max(gold, key=lambda c: c["rrf"])["text"]
    # Retrieval missed it: fall back to the doc's own chunks (same chunker).
    try:
        text = (CORPUS_DIR / gold_path).read_text(encoding="utf-8")
    except OSError:
        return None
    chunks = chunk_text(text)
    return chunks[0].text if chunks else None


def index_corpus() -> tuple[Store, FastEmbedEmbedder]:
    data_dir = DATA / "mine_data"
    config = Config(data_dir)
    config.update(folders=[str(CORPUS_DIR)])
    EVAL_MODELS.mkdir(exist_ok=True)
    embedder = FastEmbedEmbedder(EMBED_MODEL, str(EVAL_MODELS))
    embedder.ensure_loaded()
    store = Store(config.db_dir, dim=embedder.dim)
    indexer = Indexer(config=config, embedder=embedder, store=store)
    t0 = time.perf_counter()
    indexer.start_full_index()
    while indexer.running:
        time.sleep(0.25)
    assert indexer.progress.state == "done", f"index failed: {indexer.progress.error}"
    s = store.stats()
    print(f"indexed {s['files']} files / {s['chunks']} chunks in {time.perf_counter()-t0:.0f}s", flush=True)
    return store, embedder


def main() -> int:
    queries = [json.loads(l) for l in QUERIES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set[tuple[str, str]] = set()
    if PAIRS_PATH.exists():
        for l in PAIRS_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["anchor_id"], r["style"]))
    todo = [q for q in queries if (q["anchor_id"], q["style"]) not in done]
    print(f"queries={len(queries)} done_pairs_for={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to mine", flush=True)
        return 0

    store, embedder = index_corpus()
    n_pos = n_neg = 0
    with PAIRS_PATH.open("a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            query, gold = q["query"], q["path"]
            abs_gold = str(CORPUS_DIR / gold)
            cands = fuse(store, embedder, query)
            pos = _gold_text(query, gold, cands, embedder, store)
            if not pos:
                continue
            base = {"anchor_id": q["anchor_id"], "path": gold, "topic": q["topic"], "style": q["style"], "query": query}
            out.write(json.dumps({**base, "doc": pos, "label": 1}) + "\n")
            n_pos += 1
            seen_paths = {abs_gold}
            negs = 0
            for c in cands:
                if negs >= NEG_PER_POS:
                    break
                if c["path"] in seen_paths or c["text"] == pos:
                    continue
                seen_paths.add(c["path"])
                out.write(json.dumps({**base, "doc": c["text"], "label": 0}) + "\n")
                n_neg += 1
                negs += 1
            if i % 200 == 0:
                out.flush()
                print(f"  {i}/{len(todo)} queries -> {n_pos} pos / {n_neg} neg", flush=True)
    print(f"done: {n_pos} positives, {n_neg} hard negatives", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
