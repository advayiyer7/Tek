"""The no-leakage guard — the single most important file in this subproject.

It reconstructs the *exact* benchmark surface (every document the 172-probe
harness indexes, and every query it issues, from BOTH eval_stress.py and
eval_retrieval.py) and proves that the training/val data shares:
  - no document content  (verbatim or substring, either direction)
  - no query string       (after normalization)
  - none of the benchmark's distinctive answer tokens (IDs, codes, serials)

A violation raises immediately. The whole reranker result is invalid if any
benchmark probe or target file bleeds into training, so this fails loud.
"""

from __future__ import annotations

import random
import re

from .config import add_sidecar_to_path

add_sidecar_to_path()

import eval_retrieval as er  # noqa: E402
import eval_stress as es  # noqa: E402

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\b[A-Za-z0-9][\w.\-]{3,}\b")


def _is_distinctive(tok: str) -> bool:
    """A genuine identifier (the literal answer a keyword/needle probe hinges on)
    — an ID, serial, code, IP or port — as opposed to generic vocabulary that
    legitimately recurs (net-30, sprint50, p95, 2024). The test: must contain a
    digit, and be either (a) mixed-case-alnum like POL/U2723QE/C02XK1ABCDE, or
    (b) have a long (>=4) digit run, or (c) have >=2 digit groups (dotted IPs)."""
    core = tok.strip(".-_")
    if len(core) < 5 or not any(c.isdigit() for c in core):
        return False
    if re.fullmatch(r"\d{4}", core):           # bare year
        return False
    groups = re.findall(r"\d+", core)
    longest = max((len(g) for g in groups), default=0)
    has_alpha = any(c.isalpha() for c in core)
    if any(c.isupper() for c in tok) and has_alpha:
        return True
    return longest >= 4 or len(groups) >= 2


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def normalize_query(q: str) -> str:
    return _WS.sub(" ", q.strip().lower()).rstrip("?.! ")


def benchmark_docs() -> dict[str, str]:
    """Every file the 172-probe harness writes to disk and indexes — including
    both the base and scale filler sets, so training avoids all of it."""
    docs: dict[str, str] = {}
    docs.update(er.CORPUS)
    docs.update(es.CLUSTER_CORPUS)
    docs.update(es.EDGE_TEXT_FILES)
    # Long needle docs + filler are procedurally built from rng(7) in main().
    rng = random.Random(7)
    docs.update(es.build_long_docs(rng))
    docs.update(es.build_filler(110, (1, 3), rng))      # base run
    docs.update(es.build_filler(1500, (95, 115), rng))  # scale run
    # The one-long-line edge file with the EMERGENCY-OVERRIDE needle.
    docs["edge/one_long_line.txt"] = "emergency override code 9931 one long line needle"
    return docs


def benchmark_queries() -> set[str]:
    """Every query string the harness issues, from both eval modules."""
    qs: set[str] = set()
    qs.update(er.PROBES.keys())
    qs.update(er.NEGATIVE_PROBES)
    qs.update(q for q, _, _ in es.PROBES)
    qs.update(q for q, _ in es.NEGATIVES)
    qs.update(q for q, _ in es.MULTIHOP)
    qs.add("emergency override code")  # appended inside es.main()
    return {normalize_query(q) for q in qs}


def benchmark_distinctive_tokens() -> set[str]:
    toks: set[str] = set()
    for text in benchmark_docs().values():
        for m in _TOKEN.finditer(text):
            if _is_distinctive(m.group(0)):
                toks.add(m.group(0).lower())
    return toks


def assert_disjoint(train_docs: dict[str, str], queries: list[str]) -> dict:
    """Prove zero overlap. Returns a proof report; raises AssertionError on any
    violation. `train_docs` is {relpath: content}; `queries` is every generated
    train+val query."""
    bench_docs = benchmark_docs()
    bench_norm_docs = {normalize_text(t) for t in bench_docs.values()}
    bench_queries = benchmark_queries()
    bench_tokens = benchmark_distinctive_tokens()

    report: dict = {}

    # 1. exact document-content overlap (normalized) ----------------------
    train_norm = {rel: normalize_text(t) for rel, t in train_docs.items()}
    exact_doc_hits = [rel for rel, t in train_norm.items() if t in bench_norm_docs]
    assert not exact_doc_hits, f"LEAK: training docs duplicate benchmark content: {exact_doc_hits[:5]}"

    # 2. filename namespace overlap --------------------------------------
    fname_hits = sorted(set(train_docs) & set(bench_docs))
    assert not fname_hits, f"LEAK: training reuses benchmark file paths: {fname_hits[:5]}"

    # 3. substring containment either direction (catches partial reuse) ---
    #    Compare against benchmark hand-authored docs (skip giant filler/long
    #    docs whose generic sentences would false-positive; those are covered
    #    by exact + token checks). Hand-authored = the fact-bearing targets.
    hand = {**er.CORPUS, **es.CLUSTER_CORPUS, **es.EDGE_TEXT_FILES}
    hand_norm = [normalize_text(t) for t in hand.values() if len(t) > 40]
    contain_hits = []
    for rel, t in train_norm.items():
        for bt in hand_norm:
            if bt in t or t in bt:
                contain_hits.append(rel)
                break
    assert not contain_hits, f"LEAK: training docs contain benchmark text: {contain_hits[:5]}"

    # 4. query overlap ----------------------------------------------------
    norm_q = [normalize_query(q) for q in queries]
    q_hits = sorted({q for q in norm_q if q in bench_queries})
    assert not q_hits, f"LEAK: training queries appear in benchmark: {q_hits[:5]}"

    # 5. distinctive answer-token overlap --------------------------------
    train_blob = normalize_text(" ".join(train_docs.values()))
    train_tok_set = set(re.findall(r"[\w.\-]+", train_blob))
    token_hits = sorted(bench_tokens & train_tok_set)
    assert not token_hits, f"LEAK: benchmark answer tokens present in training corpus: {token_hits[:10]}"

    report.update(
        benchmark_docs=len(bench_docs),
        benchmark_queries=len(bench_queries),
        benchmark_distinctive_tokens=len(bench_tokens),
        train_docs=len(train_docs),
        train_queries=len(queries),
        exact_doc_overlap=0,
        filename_overlap=0,
        substring_overlap=0,
        query_overlap=0,
        distinctive_token_overlap=0,
        verdict="ZERO OVERLAP VERIFIED",
    )
    return report


if __name__ == "__main__":
    # Smoke: print the size of the benchmark surface we must avoid.
    print("benchmark docs:", len(benchmark_docs()))
    print("benchmark queries:", len(benchmark_queries()))
    print("benchmark distinctive tokens:", len(benchmark_distinctive_tokens()))
