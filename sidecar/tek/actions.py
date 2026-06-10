"""File-action planning. The sidecar only ever PROPOSES operations.

Execution happens in the Electron main process after the user reviews the
preview and explicitly confirms — the safety contract lives there. Plans are
lists of {kind, src, dest?, reason} operations.

- dedupe: content-hash scan, works without any LLM
- organize: rule-based (by type or by modified date), works without any LLM
- rename: LLM-suggested descriptive names from file content (needs Ollama)
- summarize: LLM summary of one file (needs Ollama)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from . import ollama
from .extract import extract_text
from .scanner import SKIP_DIRS

log = logging.getLogger(__name__)

HASH_CHUNK = 1 << 20
MAX_SCAN_FILES = 50_000

CATEGORY_BY_EXT = {
    "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt", ".tex"},
    "Spreadsheets": {".xlsx", ".xls", ".csv", ".tsv", ".ods"},
    "Presentations": {".pptx", ".ppt", ".key", ".odp"},
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".heic", ".bmp", ".tiff"},
    "Audio": {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"},
    "Video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Code": {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".cs", ".go", ".rs",
             ".rb", ".php", ".sh", ".ps1", ".sql", ".html", ".css", ".json", ".yaml", ".yml"},
    "Installers": {".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".appimage"},
}


def _walk_files(folder: str) -> list[Path]:
    root = Path(folder)
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if len(out) >= MAX_SCAN_FILES:
            break
        if any(part in SKIP_DIRS or part.startswith(".") for part in p.parts[len(root.parts):]):
            continue
        if p.is_file():
            out.append(p)
    return out


def _file_hash(path: Path) -> str | None:
    h = hashlib.blake2b(digest_size=16)
    try:
        with path.open("rb") as f:
            while chunk := f.read(HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def dedupe_scan(folder: str) -> dict:
    """Group exact duplicates by (size, then content hash). Proposes trashing
    all but the oldest copy of each group — to the recycle bin, never a hard
    delete."""
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in _walk_files(folder):
        try:
            by_size[p.stat().st_size].append(p)
        except OSError:
            continue

    groups: list[dict] = []
    operations: list[dict] = []
    for size, paths in by_size.items():
        if len(paths) < 2 or size == 0:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for p in paths:
            digest = _file_hash(p)
            if digest:
                by_hash[digest].append(p)
        for digest, dupes in by_hash.items():
            if len(dupes) < 2:
                continue
            dupes.sort(key=lambda p: p.stat().st_mtime)
            keep, extras = dupes[0], dupes[1:]
            groups.append(
                {
                    "hash": digest,
                    "size": size,
                    "keep": str(keep),
                    "duplicates": [str(p) for p in extras],
                }
            )
            for extra in extras:
                operations.append(
                    {
                        "kind": "trash",
                        "src": str(extra),
                        "reason": f"exact duplicate of {keep.name} (kept oldest copy)",
                    }
                )
    wasted = sum(g["size"] * len(g["duplicates"]) for g in groups)
    return {"groups": groups, "operations": operations, "wastedBytes": wasted}


def organize_plan(folder: str, strategy: str) -> dict:
    """Rule-based tidy-up of ONE folder level (non-recursive on purpose:
    reorganizing a whole tree is too invasive to propose automatically)."""
    root = Path(folder)
    operations: list[dict] = []
    if not root.is_dir():
        return {"operations": [], "error": "folder does not exist"}

    from datetime import datetime, timezone

    for p in sorted(root.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if strategy == "by-type":
            ext = p.suffix.lower()
            category = next(
                (cat for cat, exts in CATEGORY_BY_EXT.items() if ext in exts), "Other"
            )
            dest_dir = root / category
        else:  # by-date
            stamp = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            dest_dir = root / f"{stamp.year:04d}-{stamp.month:02d}"
        if p.parent == dest_dir:
            continue
        operations.append(
            {
                "kind": "move",
                "src": str(p),
                "dest": str(dest_dir / p.name),
                "reason": f"{strategy}: {dest_dir.name}/",
            }
        )
    return {"operations": operations}


_RENAME_PROMPT = """Suggest a clear, descriptive filename for this file based on its content.
Rules: lowercase-kebab-case, max 6 words, keep the extension {ext}, no dates unless central to the content.
Respond with ONLY a JSON object: {{"name": "your-suggestion{ext}"}}

Current name: {name}
Content excerpt:
{excerpt}"""


async def rename_plan(paths: list[str], llm_model: str) -> dict:
    """LLM-suggested descriptive renames. Requires Ollama."""
    operations: list[dict] = []
    errors: list[str] = []
    for raw in paths[:50]:
        p = Path(raw)
        if not p.is_file():
            errors.append(f"not a file: {raw}")
            continue
        text = extract_text(raw)
        if not text:
            errors.append(f"no readable text: {p.name}")
            continue
        prompt = _RENAME_PROMPT.format(ext=p.suffix.lower(), name=p.name, excerpt=text[:2500])
        try:
            reply = await ollama.generate(llm_model, prompt)
            match = re.search(r"\{.*\}", reply, re.DOTALL)
            suggestion = json.loads(match.group(0))["name"] if match else None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p.name}: {exc}")
            continue
        if not suggestion:
            errors.append(f"{p.name}: no usable suggestion")
            continue
        safe = re.sub(r"[^\w\-. ]", "", suggestion).strip()
        if not safe.lower().endswith(p.suffix.lower()):
            safe += p.suffix.lower()
        if safe and safe != p.name:
            operations.append(
                {
                    "kind": "rename",
                    "src": str(p),
                    "dest": str(p.with_name(safe)),
                    "reason": "AI-suggested descriptive name",
                }
            )
    return {"operations": operations, "errors": errors}


async def summarize(path: str, llm_model: str) -> dict:
    text = extract_text(path)
    if not text:
        return {"error": "couldn't extract readable text from this file"}
    prompt = (
        "Summarize this document in 3-6 tight bullet points. Lead with what it IS, "
        f"then what matters in it.\n\nFilename: {Path(path).name}\n\n{text[:8000]}"
    )
    try:
        summary = await ollama.generate(llm_model, prompt)
        return {"summary": summary.strip()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
