"""Trained cross-encoder reranker served via ONNX Runtime (int8, local CPU).

Drop-in alternative to the fastembed `Reranker`: same lazy-loading, thread-safe,
fail-open contract and the same output semantics (sigmoid of the CE logit, one
probability per passage). The model + tokenizer are produced by the
/reranker-training subproject and shipped as a quantized ONNX file.

Serve-time deps are deliberately minimal: onnxruntime (already used for
embeddings) + tokenizers (small Rust wheel). No torch, no transformers.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

MAX_LEN = 512


class OnnxReranker:
    """Loads <model_dir>/model.onnx (or model_int8.onnx) + tokenizer.json."""

    def __init__(self, model_dir: str, model_file: str = "model_int8.onnx") -> None:
        self.model_dir = Path(model_dir)
        self.model_file = model_file
        self.model_name = f"tek-onnx:{self.model_dir.name}/{model_file}"
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._failed = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._session is not None:
                return True
            if self._failed:
                return False
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                onnx_path = self.model_dir / self.model_file
                tok_path = self.model_dir / "tokenizer.json"
                log.info("loading onnx reranker %s", onnx_path)
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 0  # let ORT pick; CPU provider
                self._session = ort.InferenceSession(
                    str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self._input_names = {i.name for i in self._session.get_inputs()}
                tok = Tokenizer.from_file(str(tok_path))
                tok.enable_truncation(max_length=MAX_LEN)
                tok.enable_padding()
                self._tokenizer = tok
                log.info("onnx reranker ready (inputs=%s)", sorted(self._input_names))
                return True
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                log.warning("onnx reranker unavailable (%s); using fusion ranking only", exc)
                return False

    def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None:
        if not passages or not self._ensure_loaded():
            return None
        assert self._session is not None and self._tokenizer is not None
        try:
            import numpy as np

            encs = self._tokenizer.encode_batch([(query, p) for p in passages])
            ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.asarray([e.type_ids for e in encs], dtype=np.int64)
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            logits = self._session.run(None, feeds)[0].reshape(-1)
            return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
        except Exception as exc:  # noqa: BLE001
            log.warning("onnx rerank failed (%s); using fusion ranking", exc)
            return None
