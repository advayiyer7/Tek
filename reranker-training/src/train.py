"""Fine-tune a cross-encoder on the mined pairs (binary cross-entropy on the CE
logit) and checkpoint the best model on the held-out val split.

Base model: cross-encoder/ms-marco-MiniLM-L-6-v2 (see README for why this over
bge-reranker-base — same 6-layer/~22M-param latency envelope as the incumbent,
so we isolate the gain from *training* rather than a bigger model).

W&B logging defaults to OFFLINE (no key needed, fully reproducible); set
WANDB_API_KEY to sync. Everything is seeded.
"""

from __future__ import annotations

import json
import os
import sys

from .config import BASE_MODEL, RUNS, SEED, ST_MODEL_DIR, TRAIN_PATH, VAL_PATH

# --- W&B + determinism env (must be set before heavy imports) -------------
os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
os.environ.setdefault("WANDB_PROJECT", "tek-reranker")
os.environ.setdefault("WANDB_DIR", str(RUNS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EPOCHS = float(os.environ.get("TEK_EPOCHS", "2"))
BATCH = int(os.environ.get("TEK_BATCH", "32"))
LR = float(os.environ.get("TEK_LR", "2e-5"))
# Token lengths in this data are p50=93, p99=107 (only rare long-doc/needle
# chunks reach 512). Capping at 160 truncates just those tails and keeps every
# batch cheap on CPU; group_by_length removes padding waste from the variance.
MAX_LEN = int(os.environ.get("TEK_MAXLEN", "160"))
MAX_STEPS = int(os.environ.get("TEK_MAXSTEPS", "0"))  # >0: throughput smoke only


def _load(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _cap_negs(rows: list[dict], cap: int) -> list[dict]:
    """Keep at most `cap` negatives per query (all positives kept)."""
    if not cap:
        return rows
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        if r["label"] == 1:
            out.append(r)
            continue
        c = seen.get(r["query"], 0)
        if c < cap:
            out.append(r)
            seen[r["query"]] = c + 1
    return out


def main() -> int:
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.evaluation import \
        CrossEncoderClassificationEvaluator
    from sentence_transformers.cross_encoder.losses import (
        BinaryCrossEntropyLoss, MultipleNegativesRankingLoss)
    from transformers import set_seed

    # Loss choice. BCE (the original spec) only cares which side of 0.5 each
    # pair lands on, so it COMPRESSES the pretrained model's wide logit range
    # and destroys ranking resolution (measured: base relevant/irrelevant gap
    # ~21 logits -> ~2 after BCE). MNRL (softmax over positive + hard/in-batch
    # negatives, scale=10) optimizes relative ordering directly and preserves
    # those gaps — the right objective for a reranker. Default to MNRL.
    LOSS = os.environ.get("TEK_LOSS", "mnrl")
    N_HARD = int(os.environ.get("TEK_HARDNEG", "3"))

    set_seed(SEED)
    torch.manual_seed(SEED)

    import random as _random
    from .config import DATA

    train_rows, val_rows = _load(TRAIN_PATH), _load(VAL_PATH)
    train_rows = _cap_negs(train_rows, int(os.environ.get("TEK_NEG_CAP", "0")))

    # Mix in general MS-MARCO pairs (anti-forgetting) if requested + available.
    general_path = DATA / "general.jsonl"
    if os.environ.get("TEK_MIX") == "1" and general_path.exists():
        general = _load(general_path)
        train_rows = train_rows + general
        _random.Random(SEED).shuffle(train_rows)
        print(f"mixed in {len(general)} general (MS-MARCO) pairs", flush=True)

    print(f"train={len(train_rows)} val={len(val_rows)} "
          f"(train pos={sum(r['label'] for r in train_rows)})", flush=True)

    def pairs_ds(rows):
        return Dataset.from_dict({
            "query": [r["query"] for r in rows],
            "passage": [r["doc"] for r in rows],
            "label": [float(r["label"]) for r in rows],
        })

    def grouped_ds(rows, n_neg):
        """One row per query: (query, positive, negative_1..n_neg) for MNRL."""
        from collections import defaultdict
        pos: dict[str, str] = {}
        negs: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            (pos.setdefault(r["query"], r["doc"]) if r["label"] == 1
             else negs[r["query"]].append(r["doc"]))
        cols: dict[str, list[str]] = {"query": [], "positive": []}
        for i in range(n_neg):
            cols[f"negative_{i+1}"] = []
        for q, p in pos.items():
            ng = negs.get(q, [])
            if not ng:
                continue
            padded = [ng[j % len(ng)] for j in range(n_neg)]  # cycle to fixed width
            cols["query"].append(q)
            cols["positive"].append(p)
            for i in range(n_neg):
                cols[f"negative_{i+1}"].append(padded[i])
        return Dataset.from_dict(cols)

    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LEN)

    # Anti-forgetting: freeze the embeddings + the bottom FREEZE encoder layers
    # so a gentle fine-tune adapts the upper layers to Tek's domain without
    # destroying the base reranker's general (MS-MARCO) ranking ability.
    freeze = int(os.environ.get("TEK_FREEZE", "0"))
    if freeze:
        base = model.model.base_model  # underlying BERT-style encoder
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:freeze]:
            for p in layer.parameters():
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.model.parameters())
        print(f"froze embeddings + {freeze} layers; trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M", flush=True)

    if LOSS == "mnrl":
        train_ds = grouped_ds(train_rows, N_HARD)
        val_ds = grouped_ds(val_rows, N_HARD)
        loss = MultipleNegativesRankingLoss(model)
        group_by_len = False  # multi-column rows; length grouping doesn't apply
        print(f"loss=MNRL (ranking); grouped rows train={len(train_ds)} val={len(val_ds)} "
              f"hard_neg={N_HARD}", flush=True)
    else:
        train_ds, val_ds = pairs_ds(train_rows), pairs_ds(val_rows)
        loss = BinaryCrossEntropyLoss(model)
        group_by_len = True
        print("loss=BCE (pairwise)", flush=True)

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=[[r["query"], r["doc"]] for r in val_rows],
        labels=[int(r["label"]) for r in val_rows],
        name="val",
    )

    steps = max(1, (len(train_ds) // BATCH)) * max(1, int(EPOCHS))
    eval_steps = max(50, steps // 4)
    args = CrossEncoderTrainingArguments(
        output_dir=str(RUNS / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        warmup_ratio=0.1,
        group_by_length=group_by_len,  # batch similar-length pairs → minimal padding on CPU
        eval_strategy="no" if MAX_STEPS else "steps",
        eval_steps=eval_steps,
        save_strategy="no" if MAX_STEPS else "steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not MAX_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        max_steps=MAX_STEPS or -1,
        seed=SEED,
        dataloader_num_workers=0,
        report_to=[] if MAX_STEPS else ["wandb"],
        run_name="tek-rerank-minilm",
        fp16=False,
        bf16=False,
    )

    trainer = CrossEncoderTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        loss=loss, evaluator=evaluator,
    )
    trainer.train()

    ST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ST_MODEL_DIR))

    final = evaluator(model)
    print("final val metrics:", json.dumps({k: round(float(v), 4) for k, v in final.items()}), flush=True)
    (ST_MODEL_DIR / "val_metrics.json").write_text(json.dumps(final, indent=2, default=float), encoding="utf-8")
    print(f"saved trained model -> {ST_MODEL_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
