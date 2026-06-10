"""Paragraph-aware chunking with sentence-clean overlap.

Targets ~1200 chars (~300 tokens) per chunk: small enough for precise
retrieval, large enough to carry context into the LLM prompt. Overlap is cut
at a sentence (or word) boundary so embeddings never see torn fragments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 1200
MAX_CHARS = 1800
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40  # drop fragments too small to mean anything

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]?\s")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def _clean_tail(text: str, target: int) -> str:
    """The last ~target chars of text, starting at a sentence boundary when
    one exists in the window, else at a word boundary."""
    if len(text) <= target:
        return text
    window = text[-(target * 2):]
    matches = list(_SENTENCE_END.finditer(window))
    for m in matches:
        if len(window) - m.end() <= target:
            return window[m.end():].strip()
    tail = text[-target:]
    space = tail.find(" ")
    return tail[space + 1:].strip() if 0 <= space < len(tail) - 1 else tail.strip()


def chunk_text(text: str) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []

    # Split on blank lines first (paragraphs / code blocks), then pack
    # paragraphs into chunks near the target size.
    paragraphs = [p for p in (part.strip() for part in text.split("\n\n")) if p]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= MAX_CHARS:
            pieces.append(para)
        else:
            pieces.extend(_split_long(para))

    chunks: list[Chunk] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > TARGET_CHARS:
            chunks.append(Chunk(index=len(chunks), text=current))
            overlap = _clean_tail(current, OVERLAP_CHARS) if OVERLAP_CHARS else ""
            current = f"{overlap}\n\n{piece}" if overlap else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if len(current.strip()) >= MIN_CHUNK_CHARS or not chunks:
        chunks.append(Chunk(index=len(chunks), text=current))
    return [c for c in chunks if len(c.text.strip()) >= MIN_CHUNK_CHARS]


def _split_long(para: str) -> list[str]:
    """Split an oversized paragraph on sentences, then lines, then hard-wrap."""
    out: list[str] = []
    current = ""
    for line in para.split("\n"):
        while len(line) > MAX_CHARS:
            # Prefer a sentence/word boundary near the limit over a hard cut.
            cut = MAX_CHARS
            window = line[:MAX_CHARS]
            boundary = max(
                (m.end() for m in _SENTENCE_END.finditer(window)), default=-1
            )
            if boundary >= MAX_CHARS // 2:
                cut = boundary
            else:
                space = window.rfind(" ")
                if space >= MAX_CHARS // 2:
                    cut = space
            out.append(line[:cut].strip())
            line = line[cut:].strip()
        if current and len(current) + len(line) + 1 > MAX_CHARS:
            out.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return out
