"""Retrieval correctness check: builds a ~20-file corpus with distinct facts,
indexes it through the real pipeline, and asserts the right file is retrieved
top-1 for fact-specific queries.

Run:  .venv/Scripts/python eval_retrieval.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from tek.chunk import chunk_text
from tek.config import Config
from tek.embed import FastEmbedEmbedder
from tek.indexer import Indexer
from tek.rag import retrieve
from tek.scanner import scan_folders
from tek.store import Store

CORPUS: dict[str, str] = {
    "recipes/carbonara.md": "# Carbonara\nClassic Roman pasta: guanciale, eggs, pecorino romano, black pepper. Never add cream. Toss off-heat so the eggs don't scramble.",
    "recipes/sourdough.md": "# Sourdough starter\nFeed the starter daily with equal parts flour and water. It should double within 6 hours when healthy and smell pleasantly sour.",
    "recipes/curry.txt": "Thai green curry needs coconut milk, green curry paste, fish sauce, palm sugar, thai basil and bamboo shoots. Simmer gently, never boil.",
    "finance/tax_notes_2025.md": "Estimated quarterly tax payments are due April 15, June 16, September 15, and January 15. Keep 30% of freelance income aside for taxes.",
    "finance/budget.txt": "Monthly budget: rent 1800, groceries 450, utilities 180, transit 95, savings target 1000. Review subscriptions every quarter.",
    "finance/investments.md": "Portfolio allocation: 70% total-market index funds, 20% international, 10% bonds. Rebalance annually each January.",
    "work/standup_notes.md": "Sprint 42 standup: auth refactor is blocked on the SSO vendor, payments retry logic shipped, search latency regression traced to cache misses.",
    "work/onboarding.md": "New hires need: laptop from IT, VPN access, repo permissions, and the staging database credentials from the platform team.",
    "work/api_design.md": "The webhook API uses HMAC-SHA256 signatures with a per-tenant secret. Retries use exponential backoff capped at 24 hours.",
    "personal/travel_japan.md": "Japan trip plan: Tokyo 4 nights, Kyoto 3 nights, Osaka 2 nights. Get the JR rail pass before arrival. Cherry blossom season peaks early April.",
    "personal/garden.txt": "Tomatoes go in after the last frost. Basil and marigolds are good companion plants. Water deeply twice a week rather than lightly every day.",
    "personal/books.md": "Reading list: Project Hail Mary, The Idea Factory, Working in Public, The Making of the Atomic Bomb, A Pattern Language.",
    "tech/docker_cheatsheet.md": "docker compose up -d to start detached, docker system prune -af to reclaim disk space, docker logs -f to tail a container.",
    "tech/git_tips.md": "git rebase -i squashes commits before a PR. git bisect finds the commit that broke a test. git reflog recovers lost commits.",
    "tech/keyboard.txt": "Custom keyboard build: Gateron Brown switches, GMK keycaps, gasket-mounted plate, lubed stabilizers to fix rattle.",
    "health/workout.md": "Push pull legs split: bench and overhead press Monday, deadlifts and rows Wednesday, squats and lunges Friday. Deload every sixth week.",
    "health/sleep_notes.txt": "Sleep hygiene: no caffeine after 2pm, screens off an hour before bed, bedroom at 18 degrees, consistent wake time even on weekends.",
    "projects/portfolio_site.md": "Portfolio site stack: Astro with Tailwind, deployed on Cloudflare Pages, dark mode by default, blog posts written in MDX.",
    "projects/tek_ideas.md": "Tek roadmap ideas: OCR for scanned PDFs, image embeddings with CLIP, a quick-switcher palette, scheduled re-index, export citations.",
    "notes/wifi.txt": "Home router admin is at 192.168.1.1. Guest network is on VLAN 20 with client isolation. The NAS reserves 192.168.1.40.",
}

# query -> file that must be the top hit
PROBES: dict[str, str] = {
    "how do I keep my sourdough starter healthy": "recipes/sourdough.md",
    "what ingredients go in thai green curry": "recipes/curry.txt",
    "when are my quarterly estimated taxes due": "finance/tax_notes_2025.md",
    "how is the webhook API authenticated": "work/api_design.md",
    "what was blocking the auth refactor": "work/standup_notes.md",
    "how many nights are we staying in Kyoto": "personal/travel_japan.md",
    "command to reclaim docker disk space": "tech/docker_cheatsheet.md",
    "which switches did I use in my keyboard build": "tech/keyboard.txt",
    "what temperature should the bedroom be for sleep": "health/sleep_notes.txt",
    "what is my router admin address": "notes/wifi.txt",
    "what days do I do deadlifts": "health/workout.md",
    "how should I rebalance my portfolio": "finance/investments.md",
}


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="tek-eval-"))
    corpus_dir = work / "corpus"
    data_dir = work / "data"
    try:
        for rel, content in CORPUS.items():
            f = corpus_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")

        config = Config(data_dir)
        config.update(folders=[str(corpus_dir)])
        embedder = FastEmbedEmbedder(config.settings.embed_model, str(config.models_dir))
        store = Store(config.db_dir, dim=embedder.dim)
        indexer = Indexer(config=config, embedder=embedder, store=store)

        t0 = time.perf_counter()
        indexer.start_full_index()
        while indexer.running:
            time.sleep(0.2)
        assert indexer.progress.state == "done", f"index failed: {indexer.progress.error}"
        stats = store.stats()
        print(
            f"indexed {stats['files']} files / {stats['chunks']} chunks "
            f"in {time.perf_counter() - t0:.1f}s (includes model load)"
        )
        scanned = len(list(scan_folders([str(corpus_dir)])))
        assert stats["files"] == len(CORPUS) == scanned, "file count mismatch"

        # Incremental check: re-run must skip everything unchanged.
        indexer.start_full_index()
        while indexer.running:
            time.sleep(0.1)
        assert indexer.progress.indexed_files == 0, "incremental re-index re-embedded files"
        print("incremental re-index: all unchanged files skipped [OK]")

        # Chunker sanity on a long doc.
        long_doc = "\n\n".join(f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 12 for i in range(40))
        chunks = chunk_text(long_doc)
        assert len(chunks) > 3 and all(len(c.text) <= 2200 for c in chunks)

        passed = 0
        t0 = time.perf_counter()
        for query, expected_rel in PROBES.items():
            expected = str(corpus_dir / expected_rel)
            hits = retrieve(store, embedder, query, k=5)
            top = hits[0]["path"] if hits else "(no hits)"
            ok = top == expected
            passed += ok
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {query!r:55s} -> {Path(top).name} (score {hits[0]['score'] if hits else 0})")
        avg_ms = (time.perf_counter() - t0) / len(PROBES) * 1000
        print(f"\n{passed}/{len(PROBES)} probes correct top-1 · avg query {avg_ms:.0f}ms")
        return 0 if passed == len(PROBES) else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
