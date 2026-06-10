"""LanceDB-backed vector store: chunks table + files manifest table.

The files table is the incremental-indexing ledger: (path, mtime_ns, size)
decides whether a file needs re-embedding. Chunk vectors live in the chunks
table keyed by path so a changed file's chunks can be atomically replaced.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import lancedb
import pyarrow as pa

log = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class Store:
    def __init__(self, db_dir: Path, dim: int) -> None:
        self.db = lancedb.connect(str(db_dir))
        self.dim = dim
        self._lock = threading.Lock()
        self._chunks = self._open_or_create(
            CHUNKS_TABLE,
            pa.schema(
                [
                    pa.field("path", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            ),
        )
        self._files = self._open_or_create(
            FILES_TABLE,
            pa.schema(
                [
                    pa.field("path", pa.string()),
                    pa.field("mtime_ns", pa.int64()),
                    pa.field("size", pa.int64()),
                    pa.field("chunk_count", pa.int32()),
                ]
            ),
        )

    def _open_or_create(self, name: str, schema: pa.Schema):
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    # -- manifest ----------------------------------------------------------

    def known_files(self) -> dict[str, tuple[int, int]]:
        """path -> (mtime_ns, size) for everything currently indexed."""
        rows = self._files.search().select(["path", "mtime_ns", "size"]).limit(10_000_000).to_list()
        return {r["path"]: (r["mtime_ns"], r["size"]) for r in rows}

    def stats(self) -> dict:
        return {"files": self._files.count_rows(), "chunks": self._chunks.count_rows()}

    # -- writes ------------------------------------------------------------

    def replace_file(
        self,
        path: str,
        mtime_ns: int,
        size: int,
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        with self._lock:
            quoted = _sql_quote(path)
            self._chunks.delete(f"path = {quoted}")
            self._files.delete(f"path = {quoted}")
            if texts:
                self._chunks.add(
                    [
                        {"path": path, "chunk_index": i, "text": t, "vector": v}
                        for i, (t, v) in enumerate(zip(texts, vectors))
                    ]
                )
            self._files.add(
                [{"path": path, "mtime_ns": mtime_ns, "size": size, "chunk_count": len(texts)}]
            )

    def remove_files(self, paths: list[str]) -> None:
        if not paths:
            return
        with self._lock:
            predicate = "path IN (" + ", ".join(_sql_quote(p) for p in paths) + ")"
            self._chunks.delete(predicate)
            self._files.delete(predicate)

    def maybe_create_ann_index(self) -> None:
        """Build an ANN index once the library is big enough to need one.

        Below ~20k chunks LanceDB's brute-force scan is already a few ms, and
        IVF-PQ needs enough rows to train well — so only index past that.
        Idempotent: replace=False makes re-calls cheap no-ops.
        """
        try:
            count = self._chunks.count_rows()
            if count < 20_000:
                return
            with self._lock:
                self._chunks.create_index(
                    metric="cosine", vector_column_name="vector", replace=False
                )
            log.info("ANN index ensured for %d chunks", count)
        except Exception as exc:  # noqa: BLE001 — index is an optimization, never fatal
            if "already exist" not in str(exc).lower():
                log.warning("ANN index creation skipped: %s", exc)

    # -- search ------------------------------------------------------------

    def search(self, vector: list[float], k: int = 8) -> list[dict]:
        rows = (
            self._chunks.search(vector)
            .metric("cosine")
            .limit(k)
            .select(["path", "chunk_index", "text", "_distance"])
            .to_list()
        )
        results = []
        for row in rows:
            # LanceDB reports cosine *distance* (0 = identical, 2 = opposite).
            score = max(0.0, 1.0 - row.get("_distance", 1.0))
            results.append(
                {
                    "path": row["path"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "score": round(score, 4),
                }
            )
        return results
