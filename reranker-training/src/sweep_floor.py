"""Pick the no-answer floor for the trained int8 model by sweeping it over the
benchmark from a SINGLE inference pass.

The floor only filters candidates below it; the model's scores don't change. So
we run the reranker once (floor=0), capture each probe's top candidate score +
whether it's an accept doc, then compute top-1 and negative-rejection across a
grid of floors analytically. Picks the floor that maximizes neg-reject while
keeping top-1 above the off-the-shelf baseline.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .config import EMBED_MODEL, EVAL_MODELS, ONNX_DIR, RUNS, add_sidecar_to_path
from .eval_bench import build_benchmark_corpus

add_sidecar_to_path()

import eval_stress as es  # noqa: E402
import tek.rag as rag  # noqa: E402
from tek.config import Config  # noqa: E402
from tek.embed import FastEmbedEmbedder  # noqa: E402
from tek.indexer import Indexer  # noqa: E402
from tek.onnx_reranker import OnnxReranker  # noqa: E402
from tek.store import Store  # noqa: E402

BASELINE_TOP1 = 0.8243
GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.34]


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="tek-floorsweep-"))
    cdir = work / "corpus"
    try:
        probes = build_benchmark_corpus(cdir)
        config = Config(work / "data")
        config.update(folders=[str(cdir)])
        emb = FastEmbedEmbedder(EMBED_MODEL, str(EVAL_MODELS))
        emb.ensure_loaded()
        store = Store(config.db_dir, dim=emb.dim)
        idx = Indexer(config=config, embedder=emb, store=store)
        idx.start_full_index()
        while idx.running:
            time.sleep(0.25)
        rr = OnnxReranker(str(ONNX_DIR), "model_int8.onnx")
        rr.rerank("warm", ["up"])
        rag.MIN_RERANK = 0.0  # keep everything; we sweep the floor offline

        # one pass: capture top candidate score + accept for each positive,
        # and top score for each negative
        pos_recs, neg_scores = [], []
        for q, accept_rels, _ in probes:
            accept = {str(cdir / r) for r in accept_rels}
            hits = rag.retrieve(store, emb, q, k=10, reranker=rr)
            if hits:
                pos_recs.append((hits[0].get("rerank", 0.0), hits[0]["path"] in accept))
            else:
                pos_recs.append((0.0, False))
        for q, _ in es.NEGATIVES:
            hits = rag.retrieve(store, emb, q, k=10, reranker=rr)
            neg_scores.append(hits[0].get("rerank", 0.0) if hits else 0.0)

        print(f"\n{'floor':>7}{'top1':>9}{'neg_reject':>12}   note")
        rows = []
        best = None
        for f in GRID:
            top1 = sum(1 for s, ok in pos_recs if ok and s >= f) / len(pos_recs)
            negr = sum(1 for s in neg_scores if s < f) / len(neg_scores)
            note = "beats baseline top1" if top1 > BASELINE_TOP1 else ""
            print(f"{f:>7.2f}{top1:>9.3f}{negr:>12.3f}   {note}")
            rows.append({"floor": f, "top1": round(top1, 4), "neg_reject": round(negr, 4)})
            # prefer max neg_reject among floors that keep a clear top1 win (>=0.87)
            if top1 >= 0.87 and (best is None or negr > best["neg_reject"]):
                best = rows[-1]
        print(f"\nrecommended operating point (top1>=0.87, max neg-reject): {best}")
        (RUNS / "floor_sweep.json").write_text(
            json.dumps({"baseline_top1": BASELINE_TOP1, "grid": rows, "recommended": best}, indent=2),
            encoding="utf-8")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
