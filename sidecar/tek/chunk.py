"""Paragraph-aware chunking with overlap.

Targets ~1200 chars (~300 tokens) per chunk: small enough for precise
retrieval, large enough to carry context into the LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_CHARS = 1200
MAX_CHARS = 1800
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40  # drop fragments too small to mean anything


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


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
            current = current[-OVERLAP_CHARS:] + "\n\n" + piece if OVERLAP_CHARS else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if len(current.strip()) >= MIN_CHUNK_CHARS or not chunks:
        chunks.append(Chunk(index=len(chunks), text=current))
    return [c for c in chunks if len(c.text.strip()) >= MIN_CHUNK_CHARS]


def _split_long(para: str) -> list[str]:
    """Split an oversized paragraph on line breaks, then hard-wrap."""
    out: list[str] = []
    current = ""
    for line in para.split("\n"):
        while len(line) > MAX_CHARS:
            out.append(line[:MAX_CHARS])
            line = line[MAX_CHARS - OVERLAP_CHARS :]
        if current and len(current) + len(line) + 1 > MAX_CHARS:
            out.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return out
