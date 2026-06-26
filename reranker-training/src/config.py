"""Shared configuration: paths, seeds, and the sidecar import shim.

Everything the subproject produces (corpus, datasets, models, run logs) lives
under reranker-training/ so the whole thing is self-contained and disposable.
Reproducibility: one SEED drives corpus generation, query sampling, negative
mining order, the train/val split, and training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # reranker-training/
REPO = ROOT.parent                                      # Tek/
SIDECAR = REPO / "sidecar"                               # the Tek python service

DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"                             # generated training docs on disk
ANCHORS_PATH = DATA / "anchors.jsonl"                    # (path, fact, value) probe anchors
QUERIES_PATH = DATA / "queries.jsonl"                    # generated queries per anchor
PAIRS_PATH = DATA / "pairs.jsonl"                        # mined (query, doc, label) triples
TRAIN_PATH = DATA / "train.jsonl"
VAL_PATH = DATA / "val.jsonl"
STATS_PATH = DATA / "dataset_stats.json"
CALIB_PATH = DATA / "calibration.json"                  # MIN_RERANK floor from val (never benchmark)

MODELS = ROOT / "models"
ST_MODEL_DIR = MODELS / "cross-encoder-tek"             # trained sentence-transformers model
ONNX_DIR = MODELS / "onnx"                               # fp32 + int8 ONNX export
EVAL_MODELS = SIDECAR / ".eval_models"                   # shared embed/rerank cache (reused)

RUNS = ROOT / "runs"
RESULTS_PATH = RUNS / "phase4_results.json"

# --- knobs ----------------------------------------------------------------
SEED = 1234
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # base for fine-tuning (see README)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                   # must match Tek serving
OFFTHESHELF_RERANK = "Xenova/ms-marco-MiniLM-L-6-v2"    # the incumbent we benchmark against

# corpus scale (the "thorough" ~1500-doc setting)
TARGET_DOCS = 1500
MAX_ANCHORS = 2200                                       # bounds Ollama query-gen time
NEG_PER_POS = 4                                          # hard negatives mined per positive
VAL_FRACTION = 0.15                                      # document-level holdout

for _d in (DATA, CORPUS_DIR, MODELS, RUNS):
    _d.mkdir(parents=True, exist_ok=True)


def add_sidecar_to_path() -> None:
    """Make `import tek` and `import eval_stress` resolve against the real service."""
    p = str(SIDECAR)
    if p not in sys.path:
        sys.path.insert(0, p)
