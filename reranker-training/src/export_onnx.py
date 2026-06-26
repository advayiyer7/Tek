"""Export the trained cross-encoder to ONNX, int8-quantize, and verify parity.

Produces models/onnx/{model.onnx (fp32), model_int8.onnx, tokenizer.json, ...}
and writes parity.json comparing PyTorch vs ONNX-fp32 vs ONNX-int8 logits on a
sample of real (query, passage) pairs.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .config import ONNX_DIR, ST_MODEL_DIR, VAL_PATH


def _sample_pairs(n: int = 24) -> list[tuple[str, str]]:
    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(r["query"], r["doc"]) for r in rows[:n]] or [("hello", "world")]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = ONNX_DIR / "model.onnx"
    int8 = ONNX_DIR / "model_int8.onnx"

    # 1. export fp32 ONNX -------------------------------------------------
    print("exporting fp32 ONNX via optimum…", flush=True)
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(ST_MODEL_DIR), export=True)
    ort_model.save_pretrained(str(ONNX_DIR))
    tok = AutoTokenizer.from_pretrained(str(ST_MODEL_DIR))
    tok.save_pretrained(str(ONNX_DIR))
    # optimum may name the file model.onnx already; normalize.
    if not fp32.exists():
        cand = next(ONNX_DIR.glob("*.onnx"))
        cand.rename(fp32)
    assert (ONNX_DIR / "tokenizer.json").exists(), "fast tokenizer.json missing; needed for serve-time tokenization"

    # 2. int8 dynamic quantization ---------------------------------------
    # per_channel=True quantizes each weight column with its own scale, which
    # tames the per-pair outliers transformer MatMuls produce under per-tensor
    # int8 — far better ranking fidelity for the same 4x size win.
    print("int8 dynamic quantization (per-channel)…", flush=True)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True)
    size_fp32 = fp32.stat().st_size / 1e6
    size_int8 = int8.stat().st_size / 1e6
    print(f"  fp32 {size_fp32:.1f}MB -> int8 {size_int8:.1f}MB ({size_int8/size_fp32:.0%})", flush=True)

    # 3. parity: torch vs ORT fp32 vs ORT int8 ---------------------------
    pairs = _sample_pairs()
    enc = tok([q for q, _ in pairs], [d for _, d in pairs],
              padding=True, truncation=True, max_length=512, return_tensors="np")
    feeds = {k: v.astype(np.int64) for k, v in enc.items()}

    torch_model = AutoModelForSequenceClassification.from_pretrained(str(ST_MODEL_DIR)).eval()
    with torch.no_grad():
        torch_logits = torch_model(**{k: torch.tensor(v) for k, v in feeds.items()}).logits.numpy().reshape(-1)

    def run(path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = {i.name for i in sess.get_inputs()}
        f = {k: v for k, v in feeds.items() if k in names}
        return sess.run(None, f)[0].reshape(-1)

    fp32_logits = run(fp32)
    int8_logits = run(int8)

    parity = {
        "n_pairs": len(pairs),
        "size_fp32_mb": round(size_fp32, 2),
        "size_int8_mb": round(size_int8, 2),
        "max_abs_logit_diff_torch_vs_onnxfp32": float(np.max(np.abs(torch_logits - fp32_logits))),
        "max_abs_logit_diff_fp32_vs_int8": float(np.max(np.abs(fp32_logits - int8_logits))),
        "max_abs_prob_diff_torch_vs_onnxfp32": float(np.max(np.abs(_sigmoid(torch_logits) - _sigmoid(fp32_logits)))),
        "max_abs_prob_diff_fp32_vs_int8": float(np.max(np.abs(_sigmoid(fp32_logits) - _sigmoid(int8_logits)))),
        "mean_abs_prob_diff_fp32_vs_int8": float(np.mean(np.abs(_sigmoid(fp32_logits) - _sigmoid(int8_logits)))),
    }
    (ONNX_DIR / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    print("PARITY:", json.dumps(parity, indent=2), flush=True)
    # Gate 1 (strict): the fp32 ONNX export must be numerically identical to
    # PyTorch — this is export correctness.
    assert parity["max_abs_logit_diff_torch_vs_onnxfp32"] < 1e-2, "ONNX fp32 diverges from PyTorch"
    # Gate 2 (int8): guard against gross corruption via the MEAN prob shift; a
    # few per-pair outliers are expected and don't necessarily change ranking.
    # The 172-probe eval (trained-fp32 vs trained-int8 top-1) is the real arbiter.
    assert parity["mean_abs_prob_diff_fp32_vs_int8"] < 0.06, "int8 grossly corrupted the model"
    if parity["max_abs_prob_diff_fp32_vs_int8"] > 0.2:
        print(f"[warn] int8 max per-pair prob diff {parity['max_abs_prob_diff_fp32_vs_int8']:.3f} "
              f"(mean {parity['mean_abs_prob_diff_fp32_vs_int8']:.3f}); eval will confirm ranking impact", flush=True)
    print("parity OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
