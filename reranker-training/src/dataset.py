"""Assemble mined pairs into train/val, prove zero benchmark overlap, print stats.

Split is DOCUMENT-LEVEL: each gold doc (and therefore all of its queries and
positives) lands entirely in train or val. So a val query and its gold passage
are never seen during training — a stricter "val disjoint from train" than a
random row split would give.
"""

from __future__ import annotations

import json
import random
import string
import sys
from collections import Counter

from .config import (CORPUS_DIR, PAIRS_PATH, SEED, STATS_PATH, TRAIN_PATH,
                     VAL_FRACTION, VAL_PATH)
from .leakage import assert_disjoint, benchmark_queries, normalize_query


def _load_corpus_docs() -> dict[str, str]:
    return {p.relative_to(CORPUS_DIR).as_posix(): p.read_text(encoding="utf-8")
            for p in CORPUS_DIR.rglob("*") if p.is_file()}


TYPO_FRACTION = 0.30  # share of train queries that also get a typo'd variant


def _typo_word(rng: random.Random, w: str) -> str:
    if len(w) < 4:
        return w
    i = rng.randint(0, len(w) - 2)
    op = rng.random()
    if op < 0.25:                       # transpose adjacent
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op < 0.5:                        # drop a char
        return w[:i] + w[i + 1:]
    if op < 0.75:                       # duplicate a char
        return w[:i] + w[i] + w[i:]
    return w[:i] + rng.choice(string.ascii_lowercase) + w[i + 1:]  # substitute


def _typo_query(rng: random.Random, q: str) -> str:
    words = q.split()
    if not words:
        return q
    n = max(1, len(words) // 4)
    for i in rng.sample(range(len(words)), min(n, len(words))):
        if len(words[i]) >= 4:
            words[i] = _typo_word(rng, words[i])
    return " ".join(words)


def augment_typos(train: list[dict]) -> list[dict]:
    """Add typo'd query variants for a fraction of train queries so the reranker
    keeps the base model's misspelling robustness (the benchmark's `typo`
    category) and doesn't over-fit to exact-token matching. Train-only; val
    stays clean. Seeded for reproducibility."""
    rng = random.Random(SEED + 1)
    queries = sorted({r["query"] for r in train})
    chosen = set(rng.sample(queries, int(len(queries) * TYPO_FRACTION)))
    by_q: dict[str, list[dict]] = {}
    for r in train:
        if r["query"] in chosen:
            by_q.setdefault(r["query"], []).append(r)
    extra = []
    for q, rows in by_q.items():
        tq = _typo_query(rng, q)
        if tq == q:
            continue
        for r in rows:
            extra.append({**r, "query": tq, "style": r["style"] + "+typo"})
    return train + extra


def _split_paths(paths: list[str]) -> set[str]:
    rng = random.Random(SEED)
    shuffled = sorted(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    return set(shuffled[:n_val])  # val paths


def main() -> int:
    pairs = [json.loads(l) for l in PAIRS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pairs:
        print("no pairs found; run mine.py first", file=sys.stderr)
        return 1

    # Drop any generated query that coincidentally collides with a benchmark
    # query (e.g. a terse keyword like "winter tires"). Keeps the zero-overlap
    # guarantee without aborting the run; these are rare.
    bench_q = benchmark_queries()
    before = len(pairs)
    pairs = [p for p in pairs if normalize_query(p["query"]) not in bench_q]
    dropped_q = before - len(pairs)
    if dropped_q:
        print(f"dropped {dropped_q} pair(s) whose query collided with the benchmark", flush=True)

    gold_paths = sorted({p["path"] for p in pairs})
    val_paths = _split_paths(gold_paths)
    train = [p for p in pairs if p["path"] not in val_paths]
    val_all = [p for p in pairs if p["path"] in val_paths]

    # Typo-augment train (keeps the base model's misspelling robustness), then
    # drop any augmented query that happens to collide with the benchmark.
    train = augment_typos(train)
    train = [r for r in train if normalize_query(r["query"]) not in bench_q]

    # Document-level split keeps each doc on one side, but a generic/duplicate
    # query string (the LLM occasionally omits the distinguishing entity, and
    # some titles recur across docs) can map to docs on both sides. Drop those
    # val rows so val queries are strictly unseen during training.
    train_q = {p["query"] for p in train}
    val = [p for p in val_all if p["query"] not in train_q]
    dropped_val = len(val_all) - len(val)
    if dropped_val:
        print(f"dropped {dropped_val} val row(s) whose query also occurs in train", flush=True)

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with VAL_PATH.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    # ---- zero-overlap proof vs the benchmark (raises on any leak) ------
    all_queries = [p["query"] for p in (train + val)]
    proof = assert_disjoint(_load_corpus_docs(), all_queries)

    # ---- cross-split query disjointness (now guaranteed by the filter) -
    leaked = {p["query"] for p in train} & {p["query"] for p in val}
    assert not leaked, f"val queries seen in train: {list(leaked)[:5]}"

    def summarize(rows: list[dict]) -> dict:
        pos = sum(r["label"] for r in rows)
        return {
            "rows": len(rows), "positives": pos, "negatives": len(rows) - pos,
            "unique_queries": len({r["query"] for r in rows}),
            "unique_docs": len({r["path"] for r in rows}),
            "by_topic": dict(sorted(Counter(r["topic"] for r in rows).items())),
            "by_style": dict(sorted(Counter(r["style"] for r in rows).items())),
        }

    stats = {
        "seed": SEED, "val_fraction": VAL_FRACTION,
        "dropped_benchmark_query_collisions": dropped_q,
        "dropped_val_train_query_collisions": dropped_val,
        "gold_docs_total": len(gold_paths), "gold_docs_val": len(val_paths),
        "train": summarize(train), "val": summarize(val),
        "leakage_proof": proof,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- human-readable report ----------------------------------------
    print("=" * 70)
    print("DATASET STATS")
    print("=" * 70)
    print(json.dumps({k: stats[k] for k in ("gold_docs_total", "gold_docs_val", "train", "val")}, indent=2))
    print("\n" + "=" * 70)
    print("ZERO-OVERLAP PROOF vs the 172-probe benchmark")
    print("=" * 70)
    print(json.dumps(proof, indent=2))

    print("\n" + "=" * 70)
    print("EXAMPLE TRIPLES (query | +positive | -hard negative)")
    print("=" * 70)
    rng = random.Random(SEED)
    by_q: dict[str, dict] = {}
    for r in train:
        by_q.setdefault(r["query"], {"pos": None, "neg": None, "topic": r["topic"], "style": r["style"]})
        by_q[r["query"]]["pos" if r["label"] else "neg"] = r["doc"]
    examples = [q for q, v in by_q.items() if v["pos"] and v["neg"]]
    for q in rng.sample(examples, min(6, len(examples))):
        v = by_q[q]
        print(f"\n[{v['topic']}/{v['style']}] Q: {q}")
        print(f"   +  {v['pos'][:130].strip()}")
        print(f"   -  {v['neg'][:130].strip()}")
    print(f"\nwrote {TRAIN_PATH.name} ({len(train)}) and {VAL_PATH.name} ({len(val)}); stats -> {STATS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
