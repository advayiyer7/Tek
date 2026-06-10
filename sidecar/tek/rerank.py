"""Cross-encoder reranking via fastembed's TextCrossEncoder (ONNX, local).

A cross-encoder reads (query, passage) pairs jointly, so it is far more
precise than bi-encoder cosine similarity — used to re-order the hybrid
candidate pool before answering. ms-marco MiniLM-L-6 is ~80MB, downloads once
to the same models cache, and scores a 16-candidate pool in ~100-200ms on CPU.

Strictly an enhancement: if the model can't load (offline first run, low
disk), retrieval silently continues with fusion ordering alone.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Sequence

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Lazy-loading, thread-safe wrapper; never raises out of rerank()."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                log.info("loading reranker %s", self.model_name)
                self._model = TextCrossEncoder(
                    model_name=self.model_name, cache_dir=self.cache_dir
                )
                log.info("reranker ready")
                return True
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                log.warning("reranker unavailable (%s); using fusion ranking only", exc)
                return False

    def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None:
        """Relevance probability (sigmoid of the CE logit) per passage, or
        None when the model is unavailable."""
        if not passages or not self._ensure_loaded():
            return None
        assert self._model is not None
        try:
            logits = list(self._model.rerank(query, list(passages), batch_size=8))
            return [1.0 / (1.0 + math.exp(-x)) for x in logits]
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed (%s); using fusion ranking", exc)
            return None
