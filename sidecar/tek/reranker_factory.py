"""Select the reranker backend from settings.

Both backends share the same interface (lazy `rerank()` returning sigmoid
probabilities or None, plus `model_name`/`is_ready`), so the rest of the
pipeline is backend-agnostic. The off-the-shelf path stays the default; the
trained ONNX model is opt-in via `rerank_backend="tek-onnx"` so it can be
A/B-tested without removing the old path.
"""

from __future__ import annotations

import logging

from .config import Settings
from .rerank import Reranker

log = logging.getLogger(__name__)


def build_reranker(settings: Settings, cache_dir: str):
    if settings.rerank_backend == "tek-onnx" and settings.rerank_onnx_dir:
        from .onnx_reranker import OnnxReranker

        log.info("reranker backend: tek-onnx (%s)", settings.rerank_onnx_dir)
        return OnnxReranker(settings.rerank_onnx_dir)
    log.info("reranker backend: fastembed (%s)", settings.rerank_model)
    return Reranker(settings.rerank_model, cache_dir)
