"""Embedding interface with a fastembed (ONNX) implementation.

fastembed over sentence-transformers: no multi-GB torch dependency, fast CPU
inference, and the bge-small model is ~130MB — right for a shipped desktop
app. Swappable behind the Embedder protocol.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol, Sequence

log = logging.getLogger(__name__)

# bge models want an instruction prefix on *queries only* (not passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int
    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Lazy-loading wrapper around fastembed.TextEmbedding.

    The first call downloads the ONNX model to cache_dir (one-time, ~130MB);
    is_ready/loading let the API report that state to the UI.
    """

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dim = 384  # bge-small / MiniLM class; verified against output below
        self._model = None
        self._lock = threading.Lock()
        self.loading = False
        self.load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self.loading = True
            try:
                from fastembed import TextEmbedding

                log.info("loading embedding model %s", self.model_name)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=self.cache_dir
                )
                probe = next(iter(self._model.embed(["dimension probe"])))
                self.dim = len(probe)
                self.load_error = None
                log.info("embedding model ready (dim=%d)", self.dim)
            except Exception as exc:  # noqa: BLE001
                self.load_error = str(exc)
                log.error("embedding model failed to load: %s", exc)
                raise
            finally:
                self.loading = False

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.ensure_loaded()
        assert self._model is not None
        return [vec.tolist() for vec in self._model.embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        self.ensure_loaded()
        assert self._model is not None
        prefixed = BGE_QUERY_PREFIX + text if "bge" in self.model_name.lower() else text
        return next(iter(self._model.embed([prefixed]))).tolist()
