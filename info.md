# Tek — portfolio source material

Raw facts for the website page. Everything below is accurate to the shipped
v0.2.0 release; numbers come from the committed eval harness and CI logs.

- **Repo:** https://github.com/advayiyer7/Tek
- **Download:** https://github.com/advayiyer7/Tek/releases/tag/v0.2.0
- **Author:** Advay Iyer

---

## One-liner

Tek is a downloadable, cross-platform desktop app: a **local-first AI agent
for your files**. It indexes your folders on-device, lets you semantically
search and chat with everything you have, and performs file actions
(organize, rename, dedupe, summarize) — always behind a preview you
explicitly approve. Nothing ever leaves your machine.

## The pitch (longer)

Cloud AI assistants can't see your files, and the ones that can want to
upload them. Tek's bet is that modern small models are good enough to run the
entire stack locally: a 130MB embedding model, an 80MB reranker, and an
optional 2GB chat LLM give you semantic search and grounded Q&A over your
personal documents with zero network calls on your content. Privacy isn't a
setting — it's the architecture.

---

## What it does (feature list)

- **Index folders you choose**: txt/md/code (30+ extensions), PDF, DOCX.
  Aggressive filtering — media, binaries, archives, `node_modules`-style
  noise are skipped, so cost scales with embeddable text, not disk size.
- **Incremental by design**: only changed files (mtime+size) are re-embedded;
  a file watcher keeps the index live as you edit/add/delete files.
- **Hybrid semantic search**: meaning-based queries ("that note about my
  lease") and exact-keyword queries ("192.168.1.40", "HMAC-SHA256") both rank
  the right file first. Results have similarity scores, open/reveal actions.
- **Chat with your files**: streamed answers grounded in retrieved passages,
  with clickable [1]-style citations to the source files. Multi-turn —
  follow-ups like "and where are the winter tires?" resolve against the
  conversation.
- **Honest no-answer**: when nothing in your files matches, Tek says so
  instead of hallucinating — the reranker's confidence floor doubles as a
  no-answer signal (verified by negative probes in the eval).
- **File actions, safely**: find duplicates (content hash), organize a folder
  by type or date, AI-rename files from their content, summarize. Every
  mutation shows a preview plan first; deletes go to the recycle bin, never
  hard-delete.
- **Degrades gracefully**: without Ollama, search still works fully and chat
  shows the best matching passages instead of synthesized prose.

## How a user uses it

1. Download the installer for their OS (exe / dmg / AppImage) — **no
   prerequisites**, every build bundles its own Python runtime.
2. First launch: the app sets up its engine once (~1 min) and downloads the
   embedding + reranker models (~210MB, one time).
3. Library page → pick folders → index. Progress UI shows files/chunks.
4. Search page: type anything, meaning- or keyword-based.
5. Chat page: ask questions, get cited answers. For synthesized prose,
   install [Ollama](https://ollama.com) and `ollama pull llama3.2:3b` —
   otherwise extractive answers.
6. Actions page: dedupe / organize / AI-rename with preview + confirm.

Caveats to mention: builds are currently unsigned (SmartScreen / Gatekeeper
warning), and the macOS build is Apple Silicon only.

---

## Stack

| Layer | Technology |
|---|---|
| Desktop shell | Electron 42 + electron-vite 5 (Vite 7) |
| UI | React 19, TypeScript 5.9, Tailwind CSS 4 |
| Engine (sidecar) | Python 3.12, FastAPI + uvicorn on `127.0.0.1` (dynamic port) |
| Embeddings | `BAAI/bge-small-en-v1.5`, int8 ONNX via fastembed (~130MB, CPU) |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder, ONNX via fastembed (~80MB) |
| Vector + full-text store | LanceDB (embedded, on-disk; IVF-PQ ANN past 20k chunks; native BM25 FTS) |
| Local LLM | Ollama (default `llama3.2:3b`), strictly optional |
| Extraction | pypdf, python-docx, encoding-sniffing plain-text reader |
| File watching | watchfiles (Rust-backed) |
| Packaging | electron-builder (NSIS / DMG / AppImage) + python-build-standalone (bundled CPython 3.12) |
| Testing | Custom retrieval eval harness; Playwright `_electron` E2E drivers; GitHub Actions CI on Ubuntu + macOS |

## Architecture

Three processes, strict trust boundaries:

1. **Electron main** — window lifecycle, spawns/health-polls/kills the Python
   sidecar, routes all IPC, pumps chat streams. It is the *only* code allowed
   to mutate the filesystem, and only after explicit user confirmation in the
   UI ("the action execution gate"). Deletes → recycle bin.
2. **Renderer (React)** — fully sandboxed; never touches network or
   filesystem. Five pages: Chat, Search, Library, Actions, Settings. All
   traffic flows renderer → IPC → main → local HTTP.
3. **Python sidecar (FastAPI)** — bound to `127.0.0.1` on a dynamic port,
   owned by main. Pipeline: scanner → extractors → chunker → embedder →
   LanceDB. Handles retrieval, RAG streaming via Ollama, the file watcher,
   and action *planning* — it can propose moves/renames but can never
   execute them.

The split means a compromised renderer can't read files, and the sidecar
can't destroy anything: the safety contract lives in exactly one place.

## The retrieval pipeline (the interesting part)

Indexing: paragraph-aware chunking (~1200 chars, sentence-clean overlap);
each chunk is embedded with a `parentFolder/filename` context header (header
is never stored — it just connects topic queries to the file they live in);
embeddings are batched across files (128 chunks/call) for ONNX throughput.

Query time, each stage degrading gracefully if unavailable:

1. **Vector search** (bge-small cosine) — semantic recall
2. **BM25 full-text search** over chunk text *and* filenames — exact-keyword
   recall (IPs, API names, error codes that embeddings miss)
3. **Reciprocal-rank fusion** of both candidate lists
4. **Cross-encoder rerank** of the fused pool — reads (query, passage) pairs
   jointly for precision at the top; its relevance probability has a floor
   that doubles as the honest "nothing matches" signal
5. **Per-file cap** so one document can't crowd out the whole context

Multi-turn chat additionally rewrites follow-up questions into standalone
retrieval queries using the local LLM ("and where are the winter tires?" →
"where are the winter tires stored").

**Measured results** (committed eval harness, 20-file corpus, 21 probes):
19/19 probes return the correct file top-1 — including exact-keyword probes
and zero-keyword paraphrases ("is it okay to put cream in carbonara" →
carbonara.md) — and 2/2 adversarial negative probes correctly return
*nothing*. ~190ms per query on CPU including the reranker. Same scores
reproduced on Windows, macOS, and Linux in CI.

**Stress eval** (`sidecar/eval_stress.py`, deliberately adversarial — built
to make the pipeline drop points): 148 positive probes across 10 categories
over a 172-file corpus with confusable clusters, near-duplicate versions,
buried needles, typos and zero-keyword paraphrases, plus 24 unanswerable
negatives and 6 two-hop queries.

- **82.4% top-1** (122/148), recall@5 84.5%, MRR@10 0.83
- Perfect categories: exact-keyword 16/16, temporal 8/8, legacy 19/19;
  confusable clusters 92.9% — but paraphrase 50% and typo'd queries 46.7%
- **87.5% of unanswerable queries correctly return nothing** (12/12
  unrelated, 9/12 topically-adjacent traps)
- Scale-invariant: across three corpus sizes — 214, 17,689, and 24,090
  chunks (the last large enough that the IVF-PQ ANN index actually engages) —
  top-1 holds at 82.4% / 82.4% / 81.8% with the same failures. Query latency
  is governed by the cross-encoder, not the index: it *falls* from ~1.9s to
  ~1.0s p50 as the corpus grows (the rerank pool stays fixed at 16 while
  per-query fixed costs amortize)
- Failure anatomy: 22 of 26 misses are the *no-answer floor misfiring* — the
  right file was already ranked top-1 by hybrid search, but the ms-marco
  cross-encoder scores first-person/typo'd phrasings near zero ("who is my
  landlord" → 0.018 against the passage containing "landlord Marta Chen"),
  so the floor returns "nothing found". Only 4 misses ranked a wrong file
  first. Ablation without the reranker: recall@5 jumps to 98.0% and p50 falls
  to 134ms, but negative rejection collapses to 0/24 — the floor is a
  measured precision/recall trade, not a free win.

## How it was built (process / engineering story)

- **Eval-first retrieval.** Before building UI, a correctness harness
  (`sidecar/eval_retrieval.py`) indexes a fixture corpus through the real
  pipeline and asserts top-1 retrieval on fact-specific queries. Model and
  store choices (bge-small + LanceDB) were locked only after 12/12, and every
  retrieval change since had to keep the suite green. When hybrid search and
  reranking were added, the suite was extended with the failure modes they
  target (exact-keyword, paraphrase, negative probes) — 21 probes now.
- **Decision log.** Every architectural choice is recorded in `PROJECT.md`
  with the date and the why — e.g. fastembed/ONNX over sentence-transformers
  (no 2.5GB torch dependency in a public download), sidecar-plans /
  main-executes (safety contract in one place), schema changes rebuild the
  index instead of migrating (it's a cache of local files).
- **Real E2E testing.** Playwright drives the actual Electron app — not a
  test build — through index → search → chat flows (`scripts/drive-*.mjs`),
  including the *installed* app post-NSIS to verify the packaged first-run
  bootstrap. Findings included a packaging path bug only reproducible in the
  installed build.
- **Cross-platform CI.** Every push runs the retrieval eval, typecheck, a GUI
  smoke test (real window under Xvfb on Linux), and installer builds on
  Ubuntu and macOS runners. Release artifacts are the exact bytes that
  passed CI.
- **Shipping the Python problem.** v0 required system Python; v0.2 bundles
  python-build-standalone (CPython 3.12, ~44MB) into each installer and
  bootstraps a venv in user data on first run — interpreter bundled, wheels
  downloaded once. That kept the installer at ~135MB instead of ~400MB.

## Numbers worth quoting

- 82.4% top-1 on a 148-probe adversarial stress eval (confusables,
  near-duplicates, typos, buried needles); 87.5% of unanswerable queries
  correctly return nothing
- Accuracy is scale-invariant in the tested range: 214 → 24,090 chunks
  (113×, past the ANN threshold), top-1 holds within ~1 point
- 19/19 retrieval probes top-1 on the original correctness gate, 2/2
  negatives empty, ~190ms/query (CPU, small corpus; ~1.3s/query with a full
  rerank pool on a mid-range CPU)
- 100% local: 130MB embedder + 80MB reranker + optional 2GB LLM
- Installer sizes: ~135MB (Windows), ~146MB (macOS arm64), ~238MB (AppImage)
- 3 platforms tested in CI with identical eval scores
- Zero runtime prerequisites for end users

## Honest limitations / roadmap

Unsigned builds (SmartScreen/Gatekeeper warnings); macOS is arm64-only;
English-tuned embedding model; no OCR for scanned PDFs yet. From the stress
eval: typo'd and first-person-phrased queries can trip the no-answer floor
into a false "nothing found" (the right file was retrieved, then the
cross-encoder under-scored it); first-index throughput is ~6 chunks/s
end-to-end on a mid-range CPU (a 1,100-file library takes ~45 min), dominated
by per-file store writes. Roadmap: code signing, guided Ollama first-run
(auto-pull with progress), cloud-model opt-in with keys in the OS keychain,
CLIP image embeddings, OCR, query spell-correction + calibrated no-answer
floor, batched store writes.

## Assets for the page

- Screenshots of Search (hybrid results with scores), Chat (cited answer +
  follow-up), and the degraded-mode banner exist from the E2E runs; fresh
  ones can be captured by running `node scripts/drive-chat.mjs` (requires
  `npm i --no-save playwright`).
- Suggested demo flow for a GIF: index a notes folder → search an exact IP
  address → ask "what mileage was the last oil change done at?" → follow up
  with "and where are the winter tires?" → show the citation click opening
  the file.
