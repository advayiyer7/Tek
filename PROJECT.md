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
  that only re-embed changed files (mtime/hash dedupe) via a file watcher.
- **Safe by default.** Never execute a mutating file action (move/rename/delete)
  without a preview + explicit user confirmation.
- **Secrets** live in the OS keychain (Electron safeStorage/keytar) — never in
  the renderer, never in source, never committed.
- **Shippable quality.** Portfolio flagship + public download: error handling,
  edge cases, and UX polish matter.

## Architecture

- **Electron main:** window management, Python-sidecar lifecycle, IPC, keychain
  access, file-watcher triggers.
- **Renderer (React):** chat UI, settings, folder picker, indexing-progress UI.
  Sandboxed; never touches the network or filesystem directly.
- **Python sidecar (FastAPI on `127.0.0.1`, dynamic port):** ingestion pipeline
  (extract → chunk → embed → store in LanceDB), retrieval (top-k ANN), and
  generation. Consumed only by the main process.

## Phase checklist

- [x] **Phase 1 — Scaffold + sidecar wiring.**
  Electron + React + Vite + Tailwind shell; spawn the Python sidecar; round-trip
  a test message renderer → main → sidecar → back.
  *Checkpoint: app launches, status pill goes green, echo round-trip works.*
- [ ] **Phase 2 — Retrieval core on a small folder (CLI/script, no UI).**
  Extract text (.txt/.md/code first), chunk, embed locally, store in LanceDB,
  query top-k. Prove retrieval correctness on ~20 files. **Lock the embedding
  model + vector store here.**
- [ ] **Phase 3 — Chat UI + grounded answers (first demo).**
  Query → retrieve top-k → LLM (Ollama) → stream the answer with citations to
  source files.
- [ ] **Phase 4 — Scale ingestion.**
  File-type filter, PDF + richer formats, batched embedding with progress UI,
  folder picker, file watcher with incremental re-indexing.
- [ ] **Phase 5 — Performance hardening.**
  int8 quantization, RAM/index tuning, disk-backed index if needed,
  query-latency checks — against real data only.
- [ ] **Phase 6 — Agentic file actions.**
  Tool loop for rename/move/dedupe/tag/summarize, each with mandatory preview +
  confirmation before touching the filesystem.
- [ ] **Phase 7 — Flourishes + packaging.**
  Optional: confidence scoring, local/cloud routing, image embeddings. Then
  electron-builder installers (.dmg / NSIS / AppImage), first-run permissions,
  README, demo script.

**Workflow:** one phase at a time; each phase is a working vertical slice ending
in a verifiable checkpoint and a git commit. Pause for confirmation between
phases.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-06-09 | electron-vite 5 (Vite 7) as build tooling | First-class Electron main/preload/renderer bundling + HMR; plugin-react 6 needs Vite 8 so pinned plugin-react ^5.2 |
| 2026-06-09 | Sidecar transport: **HTTP (FastAPI/uvicorn) on 127.0.0.1, dynamic port** over stdio | Streaming, concurrent requests, easy to test with curl; port picked by main at runtime to avoid collisions |
| 2026-06-09 | Renderer fully sandboxed; all sidecar traffic goes renderer → IPC → main → HTTP | Renderer never needs network/fs access — smaller attack surface |
| 2026-06-09 | Sidecar venv at `sidecar/.venv`, created by `npm run sidecar:setup` | Dev-friendly; packaging strategy (bundled runtime) decided in Phase 7 |
