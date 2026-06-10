"""Ollama client: detection, model listing, and streaming chat.

Tek degrades gracefully without Ollama — callers must check availability and
fall back to retrieval-only behavior.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .config import OLLAMA_URL

log = logging.getLogger(__name__)


async def status() -> dict:
    """Probe the local Ollama server; never raises."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            version = (await client.get(f"{OLLAMA_URL}/api/version")).json()
            tags = (await client.get(f"{OLLAMA_URL}/api/tags")).json()
        models = [m["name"] for m in tags.get("models", [])]
        return {"available": True, "version": version.get("version", "?"), "models": models}
    except Exception:  # noqa: BLE001
        return {"available": False, "version": None, "models": []}


async def chat_stream(
    model: str, messages: list[dict], temperature: float = 0.2
) -> AsyncIterator[str]:
    """Yield response text deltas from Ollama's /api/chat."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {body}")
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama error: {data['error']}")
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if data.get("done"):
                    return


async def generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    """Non-streaming completion for short structured tasks (rename/tag)."""
    parts: list[str] = []
    async for delta in chat_stream(model, [{"role": "user", "content": prompt}], temperature):
        parts.append(delta)
    return "".join(parts)
