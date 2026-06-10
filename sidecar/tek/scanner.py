"""File discovery with aggressive filtering.

Cost scales with embeddable text, not disk size: only allowlisted text-bearing
extensions are indexed, size-capped, with noisy directories skipped entirely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Text-bearing formats Tek knows how to extract. Everything else is skipped.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log",
    ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env.example",
    ".html", ".htm", ".xml",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".scala", ".lua", ".r", ".jl", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".sql",
}
RICH_EXTENSIONS = {".pdf", ".docx"}
INDEXABLE_EXTENSIONS = TEXT_EXTENSIONS | RICH_EXTENSIONS

SKIP_DIRS = {
    "node_modules", ".git", ".svn", ".hg", "__pycache__", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "out", "target",
    ".next", ".nuxt", ".cache", ".gradle", ".idea", ".vs", "$RECYCLE.BIN",
    "System Volume Information", "AppData", ".tek",
}

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB — bigger than any honest document
MAX_PDF_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class FileEntry:
    path: str
    mtime_ns: int
    size: int


def is_indexable(path: Path, size: int) -> bool:
    ext = path.suffix.lower()
    if ext not in INDEXABLE_EXTENSIONS:
        return False
    limit = MAX_PDF_BYTES if ext == ".pdf" else MAX_FILE_BYTES
    return 0 < size <= limit


def scan_folders(folders: list[str]) -> Iterator[FileEntry]:
    """Yield indexable files under the given roots, skipping noisy dirs."""
    seen: set[str] = set()
    for root in folders:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for name in filenames:
                full = Path(dirpath) / name
                key = str(full).lower()
                if key in seen:
                    continue
                try:
                    stat = full.stat()
                except OSError:
                    continue
                if is_indexable(full, stat.st_size):
                    seen.add(key)
                    yield FileEntry(path=str(full), mtime_ns=stat.st_mtime_ns, size=stat.st_size)
