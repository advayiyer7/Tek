"""Retrieval-augmented chat: hybrid retrieve -> rerank -> ground the LLM -> stream.

Retrieval pipeline (each stage degrades gracefully if unavailable):
  1. Vector search (bge-small cosine) — semantic recall
  2. BM25 full-text search over text + filename — exact-keyword recall
  3. Reciprocal-rank fusion of both lists
  4. Cross-encoder rerank of the fused pool — precision at the top
  5. Per-file cap so one document can't crowd out the rest

Yields NDJSON-able events:
  {"type": "citations", "citations": [...]}   — always first
  {"type": "token", "text": "..."}            — streamed answer (LLM mode)
  {"type": "fallback", "reason": "..."}       — no-LLM extractive mode
  {"type": "done"} / {"type": "error", ...}
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from . import ollama
from .embed import FastEmbedEmbedder
from .rerank import Reranker
from .store import Store

log = logging.getLogger(__name__)

TOP_K = 8
CANDIDATE_K = 24  # per retriever, before fusion
RERANK_POOL = 16  # fused candidates handed to the cross-encoder
RRF_K = 60  # standard reciprocal-rank-fusion constant
MAX_PER_FILE = 3  # diversity: one file can't fill the whole context
MIN_SCORE = 0.30  # cosine floor below which a vector-only hit is noise
MIN_RERANK = 0.02  # CE relevance probability floor (irrelevant pairs score ≪ this)
FTS_TRUST_RANK = 3  # top FTS hits pass the floor even at low cosine (exact matches)

MAX_HISTORY_TURNS = 8
MAX_TURN_CHARS = 2000

SYSTEM_PROMPT = """You are Tek, a local AI assistant that answers questions about the user's own files.
Answer using ONLY the provided file excerpts. Rules:
- Cite sources inline as [1], [2] matching the numbered excerpts you used.
- If the excerpts don't contain the answer, say so plainly — never invent file contents.
- Be concise and direct. Use the user's terminology."""

REWRITE_PROMPT = """Rewrite the follow-up message as ONE standalone search query that fully captures what the user is asking, resolving pronouns and references using the conversation. Keep the user's key terms. Output ONLY the query text, nothing else.

Conversation:
{history}

Follow-up: {question}

Standalone query:"""


def retrieve(
    store: Store,
    embedder: FastEmbedEmbedder,
    query: str,
    k: int = TOP_K,
    reranker: Reranker | None = None,
) -> list[dict]:
    """Hybrid retrieval. Every hit carries a cosine `score`; ordering comes
    from rerank probability when available, RRF fusion otherwise."""
    vector = embedder.embed_query(query)
    vec_hits = store.search(vector, k=CANDIDATE_K)
    fts_hits = store.fts_search(query, vector, k=CANDIDATE_K)

    pool: dict[tuple[str, int], dict] = {}
    for rank, hit in enumerate(vec_hits):
        key = (hit["path"], hit["chunk_index"])
        entry = pool.setdefault(key, {**hit, "rrf": 0.0, "fts_rank": None})
        entry["rrf"] += 1.0 / (RRF_K + rank + 1)
    for rank, hit in enumerate(fts_hits):
        key = (hit["path"], hit["chunk_index"])
        entry = pool.setdefault(key, {**hit, "rrf": 0.0, "fts_rank": None})
        entry["rrf"] += 1.0 / (RRF_K + rank + 1)
        if entry["fts_rank"] is None:
            entry["fts_rank"] = rank

    candidates = sorted(pool.values(), key=lambda c: c["rrf"], reverse=True)
    # Confidence floor: semantic hits need a real cosine score; top exact-match
    # FTS hits are trusted even when the embedding missed them.
    candidates = [
        c
        for c in candidates
        if c["score"] >= MIN_SCORE
        or (c["fts_rank"] is not None and c["fts_rank"] < FTS_TRUST_RANK)
    ][:RERANK_POOL]

    if reranker is not None and candidates:
        probs = reranker.rerank(query, [c["text"] for c in candidates])
        if probs is not None:
            for c, p in zip(candidates, probs):
                c["rerank"] = round(p, 4)
            candidates.sort(key=lambda c: c["rerank"], reverse=True)
            # The CE floor is the no-answer signal: unrelated pairs score
            # orders of magnitude below it, true matches well above.
            candidates = [c for c in candidates if c["rerank"] >= MIN_RERANK]

    hits: list[dict] = []
    per_file: dict[str, int] = {}
    for c in candidates:
        if per_file.get(c["path"], 0) >= MAX_PER_FILE:
            continue
        per_file[c["path"]] = per_file.get(c["path"], 0) + 1
        hits.append(c)
        if len(hits) >= k:
            break

    for i, hit in enumerate(hits):
        hit["ref"] = i + 1
        hit["name"] = hit.get("name") or Path(hit["path"]).name
    return hits


def _build_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit['ref']}] {hit['path']}\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def _clean_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    cleaned = [
        {"role": t["role"], "content": str(t.get("content", ""))[:MAX_TURN_CHARS]}
        for t in history
        if t.get("role") in ("user", "assistant") and str(t.get("content", "")).strip()
    ]
    return cleaned[-MAX_HISTORY_TURNS:]


async def _standalone_query(
    llm_model: str, question: str, history: list[dict]
) -> str:
    """Condense a follow-up into a retrieval query using the conversation.
    Falls back to the raw question on any failure — never blocks the answer."""
    transcript = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history)
    prompt = REWRITE_PROMPT.format(history=transcript, question=question)
    try:
        rewritten = await asyncio.wait_for(
            ollama.generate(llm_model, prompt, temperature=0.0), timeout=12.0
        )
        rewritten = rewritten.strip().strip('"').splitlines()[0].strip()
        if 0 < len(rewritten) <= 400:
            log.info("query rewritten for retrieval: %r", rewritten)
            return rewritten
    except Exception as exc:  # noqa: BLE001
        log.warning("query rewrite failed (%s); using raw question", exc)
    return question


async def answer_stream(
    store: Store,
    embedder: FastEmbedEmbedder,
    llm_model: str,
    question: str,
    history: list[dict] | None = None,
    k: int = TOP_K,
    reranker: Reranker | None = None,
) -> AsyncIterator[dict]:
    turns = _clean_history(history)
    llm = await ollama.status()
    llm_ready = llm["available"] and llm_model in llm["models"]

    retrieval_query = question
    if turns and llm_ready:
        retrieval_query = await _standalone_query(llm_model, question, turns)

    try:
        # Embedding + rerank are sync/CPU-bound — keep the event loop responsive.
        hits = await asyncio.to_thread(
            retrieve, store, embedder, retrieval_query, k, reranker
        )
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "error": f"retrieval failed: {exc}"}
        return

    yield {
        "type": "citations",
        "citations": [
            {
                "ref": h["ref"],
                "path": h["path"],
                "name": h["name"],
                "score": h["score"],
                "preview": h["text"][:240],
            }
            for h in hits
        ],
    }

    if not hits:
        yield {
            "type": "fallback",
            "reason": "no-results",
            "text": "I couldn't find anything in your indexed files matching that. "
            "Try different wording, or check that the relevant folder is indexed in Settings.",
        }
        yield {"type": "done"}
        return

    if not llm_ready:
        # Extractive fallback: no LLM, so present the best passages directly.
        reason = "ollama-offline" if not llm["available"] else "model-missing"
        yield {"type": "fallback", "reason": reason, "text": ""}
        yield {"type": "done"}
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *turns,
        {
            "role": "user",
            "content": f"File excerpts:\n\n{_build_context(hits)}\n\nQuestion: {question}",
        },
    ]
    try:
        async for delta in ollama.chat_stream(llm_model, messages):
            yield {"type": "token", "text": delta}
        yield {"type": "done"}
    except Exception as exc:  # noqa: BLE001
        log.exception("generation failed")
        yield {"type": "error", "error": str(exc)}
