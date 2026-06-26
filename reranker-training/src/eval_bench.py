"""Phase 4 deliverable: evaluate every reranker variant on the UNTOUCHED
172-probe benchmark and emit one comparison table.

Variants: none | off-the-shelf (fastembed ms-marco) | trained fp32 ONNX |
trained int8 ONNX. Metrics: top-1 (headline), MRR@10, NDCG@10, recall@5,
negative-rejection, and CPU rerank latency p50/p99 (rerank-only + end-to-end).

The benchmark corpus + probes are rebuilt verbatim from eval_stress.py /
eval_retrieval.py — this file never edits them. The MIN_RERANK no-answer floor
for the trained model is calibrated on the VAL split (never the benchmark).
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .config import (CALIB_PATH, EMBED_MODEL, EVAL_MODELS, OFFTHESHELF_RERANK,
                     ONNX_DIR, RESULTS_PATH, VAL_PATH, add_sidecar_to_path)

add_sidecar_to_path()

import eval_retrieval as er  # noqa: E402
import eval_stress as es  # noqa: E402
import tek.rag as rag  # noqa: E402
from tek.config import Config  # noqa: E402
from tek.embed import FastEmbedEmbedder  # noqa: E402
from tek.indexer import Indexer  # noqa: E402
from tek.onnx_reranker import OnnxReranker  # noqa: E402
from tek.rerank import Reranker  # noqa: E402
from tek.store import Store  # noqa: E402


def pctl(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))]


class Timed:
    """Wraps a reranker to record per-call rerank-only latency."""
    def __init__(self, inner) -> None:
        self.inner = inner
        self.model_name = getattr(inner, "model_name", "?")
        self.last_ms = 0.0

    def warmup(self) -> bool:
        r = self.inner.rerank("warmup query", ["warmup passage one", "warmup passage two"])
        return r is not None

    def rerank(self, query, passages):
        t = time.perf_counter()
        r = self.inner.rerank(query, passages)
        self.last_ms = (time.perf_counter() - t) * 1000
        return r


def build_benchmark_corpus(corpus_dir: Path) -> list[tuple[str, tuple[str, ...], str]]:
    """Rebuild the exact non-scale benchmark corpus + the full probe list."""
    rng = random.Random(7)  # identical seed to eval_stress.main
    corpus: dict[str, str] = {}
    corpus.update(er.CORPUS)
    corpus.update(es.CLUSTER_CORPUS)
    corpus.update(es.EDGE_TEXT_FILES)
    corpus.update(es.build_long_docs(rng))
    corpus.update(es.build_filler(110, (1, 3), rng))
    for rel, content in corpus.items():
        f = corpus_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    (corpus_dir / "edge").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "edge/empty.txt").write_bytes(b"")
    (corpus_dir / "edge/whitespace.txt").write_text("   \n\n  \t \n", encoding="utf-8")
    (corpus_dir / "edge/binaryish.txt").write_bytes(b"MZ\x00\x01\x02" + bytes(range(256)) * 16)
    needle = ("the quick brown fox guards the perimeter and the EMERGENCY-OVERRIDE-CODE 9931 "
              "sits exactly here in the middle of an unbroken line ")
    (corpus_dir / "edge/one_long_line.txt").write_text(
        ("lorem ipsum dolor sit amet consectetur " * 600) + needle
        + ("adipiscing elit sed do eiusmod tempor " * 600), encoding="utf-8")

    probes = [(q, (rel,), "legacy") for q, rel in er.PROBES.items()]
    probes += list(es.PROBES)
    probes.append(("emergency override code", ("edge/one_long_line.txt",), "needle"))
    return probes


def calibrate_floor(reranker: Timed) -> float:
    """No-answer floor = 3rd percentile of val POSITIVE scores (keeps ~97% val
    recall). Uses only the val split — never the benchmark."""
    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [r for r in rows if r["label"] == 1]
    scores = []
    for r in pos:
        s = reranker.inner.rerank(r["query"], [r["doc"]])
        if s:
            scores.append(s[0])
    p = float(os.environ.get("TEK_FLOOR_PCTL", "5"))  # val-recall criterion (~95%); see floor_sweep
    floor = max(1e-3, pctl(scores, p)) if scores else 0.02
    CALIB_PATH.write_text(json.dumps(
        {"model": reranker.model_name, "n_val_pos": len(scores), "percentile": p,
         "floor": floor, "pos_score_p50": pctl(scores, 50)}, indent=2), encoding="utf-8")
    return floor


def run_variant(name, store, embedder, probes, reranker, floor) -> dict:
    rag.MIN_RERANK = floor
    cats = defaultdict(lambda: defaultdict(list))
    e2e, rr_ms = [], []
    for query, accept_rels, cat in probes:
        accept = {str(store_corpus_dir / r) for r in accept_rels}
        if reranker:
            reranker.last_ms = 0.0
        t0 = time.perf_counter()
        hits = rag.retrieve(store, embedder, query, k=10, reranker=reranker)
        e2e.append((time.perf_counter() - t0) * 1000)
        if reranker:
            rr_ms.append(reranker.last_ms)
        paths = [h["path"] for h in hits]
        seen, rank = [], 0
        for p in paths:
            if p not in seen:
                seen.append(p)
        rank = next((i + 1 for i, p in enumerate(seen) if p in accept), 0)
        cats[cat]["top1"].append(1.0 if paths and paths[0] in accept else 0.0)
        cats[cat]["r5"].append(1.0 if 0 < rank <= 5 else 0.0)
        cats[cat]["mrr"].append(1.0 / rank if rank else 0.0)
        cats[cat]["ndcg"].append(1.0 / math.log2(rank + 1) if 0 < rank <= 10 else 0.0)

    # negatives: must return empty
    neg = []
    for query, _ in es.NEGATIVES:
        if reranker:
            reranker.last_ms = 0.0
        t0 = time.perf_counter()
        hits = rag.retrieve(store, embedder, query, k=10, reranker=reranker)
        e2e.append((time.perf_counter() - t0) * 1000)
        if reranker:
            rr_ms.append(reranker.last_ms)
        neg.append(1.0 if not hits else 0.0)

    def agg(key):
        vals = [v for c in cats.values() for v in c[key]]
        return sum(vals) / len(vals) if vals else 0.0

    per_cat = {c: round(sum(cats[c]["top1"]) / len(cats[c]["top1"]), 4) for c in cats}
    return {
        "variant": name, "floor": round(floor, 5),
        "top1": round(agg("top1"), 4), "recall5": round(agg("r5"), 4),
        "mrr10": round(agg("mrr"), 4), "ndcg10": round(agg("ndcg"), 4),
        "neg_reject": round(sum(neg) / len(neg), 4),
        "rerank_p50_ms": round(pctl(rr_ms, 50), 1), "rerank_p99_ms": round(pctl(rr_ms, 99), 1),
        "e2e_p50_ms": round(pctl(e2e, 50), 1), "e2e_p99_ms": round(pctl(e2e, 99), 1),
        "per_category_top1": per_cat,
        "positives_n": sum(len(c["top1"]) for c in cats.values()),
    }


store_corpus_dir: Path  # set in main (used by run_variant scoring)


def main() -> int:
    global store_corpus_dir
    work = Path(tempfile.mkdtemp(prefix="tek-rerank-eval-"))
    store_corpus_dir = work / "corpus"
    try:
        probes = build_benchmark_corpus(store_corpus_dir)
        config = Config(work / "data")
        config.update(folders=[str(store_corpus_dir)])
        embedder = FastEmbedEmbedder(EMBED_MODEL, str(EVAL_MODELS))
        embedder.ensure_loaded()
        store = Store(config.db_dir, dim=embedder.dim)
        indexer = Indexer(config=config, embedder=embedder, store=store)
        indexer.start_full_index()
        while indexer.running:
            time.sleep(0.25)
        assert indexer.progress.state == "done", f"index failed: {indexer.progress.error}"
        print(f"indexed benchmark: {store.stats()} ; probes={len(probes)} + {len(es.NEGATIVES)} negatives", flush=True)

        offthe = Timed(Reranker(OFFTHESHELF_RERANK, str(EVAL_MODELS)))
        fp32 = Timed(OnnxReranker(str(ONNX_DIR), "model.onnx"))
        int8 = Timed(OnnxReranker(str(ONNX_DIR), "model_int8.onnx"))
        for r in (offthe, fp32, int8):
            print(f"warmup {r.model_name}: {'ok' if r.warmup() else 'UNAVAILABLE'}", flush=True)

        int8_floor = calibrate_floor(int8)
        print(f"calibrated trained-model floor (val p{os.environ.get('TEK_FLOOR_PCTL','10')}): "
              f"{int8_floor:.4f}", flush=True)

        variants = [
            run_variant("none", store, embedder, probes, None, rag.MIN_RERANK),
            run_variant("offtheshelf", store, embedder, probes, offthe, 0.02),
            run_variant("trained-fp32", store, embedder, probes, fp32, int8_floor),
            run_variant("trained-int8", store, embedder, probes, int8, int8_floor),
        ]

        # ---- table ----
        hdr = f"{'variant':<14}{'top1':>8}{'mrr10':>8}{'ndcg10':>8}{'recall5':>9}{'neg':>7}{'rr_p50':>8}{'rr_p99':>8}{'e2e_p50':>9}"
        print("\n" + "=" * len(hdr))
        print("172-PROBE BENCHMARK — RERANKER COMPARISON")
        print("=" * len(hdr))
        print(hdr)
        for v in variants:
            print(f"{v['variant']:<14}{v['top1']:>8.3f}{v['mrr10']:>8.3f}{v['ndcg10']:>8.3f}"
                  f"{v['recall5']:>9.3f}{v['neg_reject']:>7.2f}{v['rerank_p50_ms']:>8.0f}"
                  f"{v['rerank_p99_ms']:>8.0f}{v['e2e_p50_ms']:>9.0f}")

        base = next(v for v in variants if v["variant"] == "offtheshelf")["top1"]
        mine = next(v for v in variants if v["variant"] == "trained-int8")["top1"]
        lift = mine - base
        print(f"\ntop-1 lift (int8 trained vs off-the-shelf): {lift:+.3f} "
              f"({base:.1%} -> {mine:.1%})")

        results = {"variants": variants, "baseline_top1": base, "trained_int8_top1": mine,
                   "top1_lift": round(lift, 4)}
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nresults -> {RESULTS_PATH}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
