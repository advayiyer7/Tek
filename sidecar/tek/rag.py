"""Retrieval-augmented chat: retrieve top-k chunks, ground the LLM, stream.

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
from .store import Store

log = logging.getLogger(__name__)

TOP_K = 8
MIN_SCORE = 0.30  # cosine-similarity floor below which a hit is noise

SYSTEM_PROMPT = """You are Tek, a local AI assistant that answers questions about the user's own files.
Answer using ONLY the provided file excerpts. Rules:
- Cite sources inline as [1], [2] matching the numbered excerpts you used.
- If the excerpts don't contain the answer, say so plainly — never invent file contents.
- Be concise and direct. Use the user's terminology."""


def retrieve(
    store: Store, embedder: FastEmbedEmbedder, query: str, k: int = TOP_K
) -> list[dict]:
    vector = embedder.embed_query(query)
    hits = [h for h in store.search(vector, k=k) if h["score"] >= MIN_SCORE]
    for i, hit in enumerate(hits):
        hit["ref"] = i + 1
        hit["name"] = Path(hit["path"]).name
    return hits


def _build_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit['ref']}] {hit['path']}\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


async def answer_stream(
    store: Store,
    embedder: FastEmbedEmbedder,
    llm_model: str,
    question: str,
    k: int = TOP_K,
) -> AsyncIterator[dict]:
    try:
        # Embedding is sync/CPU-bound — keep the event loop responsive.
        hits = await asyncio.to_thread(retrieve, store, embedder, question, k)
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

    llm = await ollama.status()
    if not llm["available"] or llm_model not in llm["models"]:
        # Extractive fallback: no LLM, so present the best passages directly.
        reason = "ollama-offline" if not llm["available"] else "model-missing"
        yield {"type": "fallback", "reason": reason, "text": ""}
        yield {"type": "done"}
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
