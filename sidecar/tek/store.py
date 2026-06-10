"""LanceDB-backed vector store: chunks table + files manifest table.

The files table is the incremental-indexing ledger: (path, mtime_ns, size)
decides whether a file needs re-embedding. Chunk vectors live in the chunks
table keyed by path so a changed file's chunks can be atomically replaced.

Search is hybrid: ANN/brute-force vector search plus a native (lance) BM25
full-text index over chunk text + filename. Fusion happens in rag.py.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

log = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"

# Bump when the chunks schema changes; mismatched tables are dropped and the
# next index run rebuilds them from the source files (cheap, local).
EXPECTED_CHUNK_COLUMNS = {"path", "name", "chunk_index", "text", "vector"}


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fts_sanitize(query: str) -> str:
    """Reduce a user query to plain terms the FTS match parser always accepts."""
    return re.sub(r"[^\w\s]", " ", query).strip()


class Store:
    def __init__(self, db_dir: Path, dim: int) -> None:
        self.db = lancedb.connect(str(db_dir))
        self.dim = dim
        self._lock = threading.Lock()
        self._fts_ready = False
        self._chunks = self._open_or_create(
            CHUNKS_TABLE,
            pa.schema(
                [
                    pa.field("path", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            ),
            expected_columns=EXPECTED_CHUNK_COLUMNS,
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
        try:
            # An FTS index built in a previous run persists with the table.
            self._fts_ready = any(
                "FTS" in str(getattr(idx, "index_type", "")).upper()
                for idx in self._chunks.list_indices()
            )
        except Exception:  # noqa: BLE001
            self._fts_ready = False

    def _open_or_create(
        self, name: str, schema: pa.Schema, expected_columns: set[str] | None = None
    ):
        if name in self.db.table_names():
            table = self.db.open_table(name)
            if expected_columns and set(table.schema.names) != expected_columns:
                # Index format upgrade: drop and let the next index run rebuild.
                log.warning("table %r has an old schema; rebuilding (reindex needed)", name)
                self.db.drop_table(name)
                if FILES_TABLE in self.db.table_names():
                    self.db.drop_table(FILES_TABLE)
            else:
                return table
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
                name = Path(path).name
                self._chunks.add(
                    [
                        {
                            "path": path,
                            "name": name,
                            "chunk_index": i,
                            "text": t,
                            "vector": v,
                        }
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

    def ensure_indexes(self) -> None:
        """(Re)build search indexes after an index run.

        - FTS (native lance BM25) over text + filename: always, rebuild is
          seconds even at 100k chunks and keeps the index in sync with writes.
        - ANN (IVF-PQ) for vectors: only past ~20k chunks — below that
          LanceDB's brute-force scan is already a few ms, and IVF-PQ needs
          enough rows to train well.
        Both are optimizations: failure must never break indexing.
        """
        try:
            if self._chunks.count_rows() == 0:
                return
            with self._lock:
                # Native FTS indexes are single-column; one per searched
                # column, then both are queried together via fts_columns.
                self._chunks.create_fts_index("text", use_tantivy=False, replace=True)
                self._chunks.create_fts_index("name", use_tantivy=False, replace=True)
            self._fts_ready = True
            log.info("FTS indexes rebuilt")
        except Exception as exc:  # noqa: BLE001
            log.warning("FTS index creation skipped: %s", exc)

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

    # Back-compat alias (older callers).
    maybe_create_ann_index = ensure_indexes

    # -- search ------------------------------------------------------------

    def search(self, vector: list[float], k: int = 8) -> list[dict]:
        """Vector search. Returns hits with a cosine-similarity `score`."""
        rows = (
            self._chunks.search(vector)
            .metric("cosine")
            .limit(k)
            .select(["path", "name", "chunk_index", "text", "_distance"])
            .to_list()
        )
        results = []
        for row in rows:
            # LanceDB reports cosine *distance* (0 = identical, 2 = opposite).
            score = max(0.0, 1.0 - row.get("_distance", 1.0))
            results.append(
                {
                    "path": row["path"],
                    "name": row["name"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "score": round(score, 4),
                }
            )
        return results

    def fts_search(self, query: str, query_vector: list[float], k: int = 8) -> list[dict]:
        """BM25 full-text search over chunk text + filename.

        Cosine similarity against `query_vector` is computed locally for each
        hit so FTS-only results carry the same `score` semantics as vector
        hits. Returns [] when the FTS index doesn't exist yet (pre-first-index)
        — hybrid retrieval degrades to vector-only.
        """
        if not self._fts_ready:
            return []
        terms = _fts_sanitize(query)
        if not terms:
            return []
        try:
            rows = (
                self._chunks.search(terms, query_type="fts", fts_columns=["text", "name"])
                .limit(k)
                .select(["path", "name", "chunk_index", "text", "vector"])
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001 — FTS is an enhancement, never fatal
            log.warning("FTS search failed (falling back to vector-only): %s", exc)
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        results = []
        for row in rows:
            v = np.asarray(row["vector"], dtype=np.float32)
            vn = float(np.linalg.norm(v)) or 1.0
            cosine = float(np.dot(q, v) / (qn * vn))
            results.append(
                {
                    "path": row["path"],
                    "name": row["name"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "score": round(max(0.0, cosine), 4),
                }
            )
        return results

    def mark_fts_stale_ok(self) -> None:
        """Called after small incremental writes: lance FTS doesn't see brand-new
        rows until reindexed, which is acceptable between watcher batches —
        vector search still covers them immediately."""
        # Intentionally a no-op; kept as an explicit decision marker.
