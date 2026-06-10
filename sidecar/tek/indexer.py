"""Incremental indexing pipeline: scan -> diff -> extract -> chunk -> embed -> store.

Runs in a worker thread (embedding is CPU-bound); progress is polled by the UI
via /index/status. A watchfiles-based watcher feeds changed paths back through
the same per-file pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import chunk_text
from .config import Config
from .embed import FastEmbedEmbedder
from .extract import extract_text
from .scanner import FileEntry, is_indexable, scan_folders
from .store import Store

log = logging.getLogger(__name__)

EMBED_BATCH_CHUNKS = 128  # chunks pooled across files per embed call
WATCH_FTS_REBUILD_MAX = 50_000  # above this, leave FTS refresh to full runs


def _embed_header(path: str) -> str:
    """Filename + parent folder, prepended to chunk text before embedding
    (never stored): 'recipes/sourdough.md' lets the embedding connect a
    query's topic to the file it lives in."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


@dataclass
class IndexProgress:
    state: str = "idle"  # idle | loading-model | scanning | indexing | done | error
    total_files: int = 0
    processed_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    removed_files: int = 0
    total_chunks: int = 0
    current_path: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "totalFiles": self.total_files,
            "processedFiles": self.processed_files,
            "indexedFiles": self.indexed_files,
            "skippedFiles": self.skipped_files,
            "removedFiles": self.removed_files,
            "totalChunks": self.total_chunks,
            "currentPath": self.current_path,
            "error": self.error,
            "elapsedS": round(
                ((self.finished_at or time.monotonic()) - self.started_at), 1
            )
            if self.started_at
            else 0.0,
        }


@dataclass
class Indexer:
    config: Config
    embedder: FastEmbedEmbedder
    store: Store
    progress: IndexProgress = field(default_factory=IndexProgress)
    _thread: threading.Thread | None = None
    _watcher_thread: threading.Thread | None = None
    _stop_watch: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_full_index(self) -> bool:
        """Kick a full (incremental) index of all configured folders."""
        with self._lock:
            if self.running:
                return False
            self.progress = IndexProgress(state="scanning", started_at=time.monotonic())
            self._thread = threading.Thread(target=self._run_full, daemon=True)
            self._thread.start()
            return True

    def _run_full(self) -> None:
        prog = self.progress
        try:
            prog.state = "loading-model"
            self.embedder.ensure_loaded()

            prog.state = "scanning"
            entries = list(scan_folders(self.config.settings.folders))
            prog.total_files = len(entries)

            known = self.store.known_files()
            on_disk = {e.path for e in entries}

            # Files that vanished (or fell outside the folder set) get purged.
            stale = [p for p in known if p not in on_disk]
            if stale:
                self.store.remove_files(stale)
                prog.removed_files = len(stale)

            prog.state = "indexing"
            pending: list[tuple[FileEntry, list[str]]] = []
            pending_chunks = 0

            def flush() -> None:
                nonlocal pending, pending_chunks
                if not pending:
                    return
                # One embed call across files: far better ONNX batch
                # utilization than per-file calls on small documents.
                to_embed = [
                    f"{_embed_header(e.path)}\n{t}" for e, texts in pending for t in texts
                ]
                vectors = self.embedder.embed_passages(to_embed)
                offset = 0
                for e, texts in pending:
                    vecs = vectors[offset : offset + len(texts)]
                    offset += len(texts)
                    self.store.replace_file(e.path, e.mtime_ns, e.size, texts, vecs)
                    prog.indexed_files += 1
                    prog.total_chunks += len(texts)
                    prog.processed_files += 1
                pending = []
                pending_chunks = 0

            for entry in entries:
                prog.current_path = entry.path
                if known.get(entry.path) == (entry.mtime_ns, entry.size):
                    prog.skipped_files += 1
                    prog.processed_files += 1
                    continue
                text = extract_text(entry.path)
                chunks = chunk_text(text) if text else []
                if not chunks:
                    # Unreadable/empty: record it so we don't retry every run.
                    self.store.replace_file(entry.path, entry.mtime_ns, entry.size, [], [])
                    prog.skipped_files += 1
                    prog.processed_files += 1
                    continue
                pending.append((entry, [c.text for c in chunks]))
                pending_chunks += len(chunks)
                if pending_chunks >= EMBED_BATCH_CHUNKS:
                    flush()
            flush()

            self.store.ensure_indexes()
            prog.state = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("indexing failed")
            prog.state = "error"
            prog.error = str(exc)
        finally:
            prog.current_path = ""
            prog.finished_at = time.monotonic()

    def _index_one(self, entry: FileEntry, prog: IndexProgress) -> None:
        text = extract_text(entry.path)
        if not text:
            # Unreadable/empty: record it so we don't retry every run.
            self.store.replace_file(entry.path, entry.mtime_ns, entry.size, [], [])
            prog.skipped_files += 1
            return
        chunks = chunk_text(text)
        if not chunks:
            self.store.replace_file(entry.path, entry.mtime_ns, entry.size, [], [])
            prog.skipped_files += 1
            return
        texts = [c.text for c in chunks]
        header = _embed_header(entry.path)
        vectors = self.embedder.embed_passages([f"{header}\n{t}" for t in texts])
        self.store.replace_file(entry.path, entry.mtime_ns, entry.size, texts, vectors)
        prog.indexed_files += 1
        prog.total_chunks += len(texts)

    # -- watcher -----------------------------------------------------------

    def start_watcher(self) -> None:
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._stop_watch.clear()
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()

    def stop_watcher(self) -> None:
        self._stop_watch.set()

    def _watch_loop(self) -> None:
        from watchfiles import watch

        folders = [f for f in self.config.settings.folders if Path(f).is_dir()]
        if not folders:
            return
        log.info("watching %d folder(s) for changes", len(folders))
        try:
            for changes in watch(
                *folders, stop_event=self._stop_watch, debounce=1600, step=500
            ):
                if not self.config.settings.watch_enabled:
                    continue
                if self.running:
                    continue  # full index in flight; it will pick changes up
                self._apply_changes({path for _, path in changes})
        except Exception:  # noqa: BLE001
            log.exception("watcher stopped unexpectedly")

    def _apply_changes(self, paths: set[str]) -> None:
        removed: list[str] = []
        reindexed = 0
        for raw in paths:
            p = Path(raw)
            try:
                stat = p.stat()
                exists = p.is_file()
            except OSError:
                exists = False
                stat = None
            if exists and stat and is_indexable(p, stat.st_size):
                entry = FileEntry(path=str(p), mtime_ns=stat.st_mtime_ns, size=stat.st_size)
                try:
                    self._index_one(entry, self.progress)
                    reindexed += 1
                    log.info("re-indexed %s", p)
                except Exception:  # noqa: BLE001
                    log.exception("failed to re-index %s", p)
            elif not exists:
                removed.append(str(p))
        if removed:
            self.store.remove_files(removed)
            log.info("removed %d deleted file(s) from index", len(removed))
        if reindexed and self.store.stats()["chunks"] <= WATCH_FTS_REBUILD_MAX:
            # Keep BM25 fresh for live edits; on huge libraries vector search
            # covers new rows until the next full index run.
            self.store.ensure_indexes()
