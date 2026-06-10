# Tek — Project Plan

A downloadable, cross-platform (macOS / Windows / Linux) desktop app: a
**local-first AI agent for your files**. Indexes files on-device, supports
semantic search + chat (RAG), and performs file actions on request. Free public
download, hosted on my portfolio.

## Non-negotiable principles

- **Local-first / private.** Local embedding model + local LLM (Ollama) by
  default. Cloud models are opt-in only. File contents never leave the machine
  unless the user explicitly enables cloud.
- **Cross-platform from day one** (macOS, Windows, Linux).
- **Cost scales with embeddable text, not disk size.** Aggressively filter what
  gets indexed — skip media, binaries, archives, VM images.
- **Incremental indexing.** Initial index is a one-time background job; after
  that only re-embed changed files (mtime/size dedupe) via a file watcher.
- **Safe by default.** Never execute a mutating file action (move/rename/delete)
  without a preview + explicit user confirmation. Deletes go to the recycle
  bin, never hard-delete.
- **Secrets** live in the OS keychain (Electron safeStorage) — never in the
  renderer, never in source, never committed.
- **Shippable quality.** Portfolio flagship + public download.

## Architecture

- **Electron main:** window management, sidecar lifecycle (spawn / health-poll /
  kill), IPC, chat-stream pump, folder picker, and the **action execution
  gate** — the only code that mutates the filesystem, post-confirmation.
- **Renderer (React, sandboxed):** Chat / Search / Library / Actions / Settings.
  Never touches network or fs directly.
- **Python sidecar (FastAPI on `127.0.0.1`, dynamic port):** scanner →
  extractors (txt/md/code, pypdf, python-docx) → chunker → fastembed (bge-small
  int8 ONNX) → LanceDB. Retrieval, RAG streaming via Ollama, watchfiles
  watcher, and action *planning* (never execution).

## Status

- [x] Scaffold + sidecar wiring (round-trip proven, commit `d93ff8c`)
- [x] Retrieval core — `sidecar/eval_retrieval.py`: 12/12 probes top-1 correct
  on a 20-file corpus, ~19ms/query, incremental re-index verified.
  **Locked: bge-small-en-v1.5 (fastembed/ONNX) + LanceDB.**
- [x] Chat UI + grounded streamed answers with citations; graceful degraded
  mode without Ollama (extractive best-passages)
- [x] Scaled ingestion: type/size/dir filters, PDF + DOCX, progress UI, folder
  picker, watchfiles incremental re-indexing
- [x] Performance: int8-quantized ONNX embeddings; ANN (IVF-PQ) index built
  automatically past 20k chunks; LanceDB is disk-backed by design
- [x] File actions behind mandatory preview+confirm: dedupe (hash), organize
  (by type/date), AI rename + summarize (Ollama)
- [x] electron-builder config (NSIS / dmg / AppImage) + first-run venv
  bootstrap on packaged installs

## Remaining / future

- [ ] Real-data validation pass (point at a big folder; tune chunking, scores)
- [ ] App icon + signed installers; produce + test a real .exe/.dmg/.AppImage
- [ ] Bundle a Python runtime (drop the system-Python requirement) — or ship
  inference fully in-sidecar via llama.cpp
- [ ] Guided Ollama first-run (auto-pull model with progress)
- [ ] Cloud provider opt-in (keys in OS keychain via safeStorage)
- [ ] Flourishes: confidence scores, image embeddings (CLIP), OCR for scans
- [ ] Demo script + portfolio page

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-06-09 | electron-vite 5 (Vite 7); plugin-react pinned ^5.2 | plugin-react 6 needs Vite 8, electron-vite caps at 7 |
| 2026-06-09 | Sidecar transport: HTTP (FastAPI) on 127.0.0.1, dynamic port | Streaming, concurrency, curl-testable |
| 2026-06-09 | Renderer sandboxed; all traffic renderer → IPC → main → HTTP | Smallest attack surface |
| 2026-06-09 | **fastembed (ONNX) over sentence-transformers** | No ~2.5GB torch dep; int8 bge-small is ~130MB and fast on CPU — right for a public download |
| 2026-06-09 | **bge-small-en-v1.5 + LanceDB locked** after eval: 12/12 top-1, ~19ms | Proven correct on fixture corpus |
| 2026-06-09 | Sidecar plans actions; only Electron main executes, post-confirm; deletes → recycle bin | Safety contract in one place |
| 2026-06-09 | No-Ollama degraded mode: search + extractive answers + hash/rule actions | App is useful with zero extra installs |
| 2026-06-10 | Packaged builds create the Python venv in userData on first run (needs system Python) | Honest v0 packaging; bundled runtime later |
