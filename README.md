# Tek

**A local-first AI agent for your files.**

Tek indexes your files on-device, lets you semantically search and chat across
everything, and can take file actions (organize, rename, dedupe, summarize) —
always behind a preview you explicitly approve.

**Privacy is the core feature.** Text extraction, embeddings, the vector index,
and LLM inference all run on your machine. Tek makes no network calls with your
file contents. Cloud models will be strictly opt-in.

## What works without anything extra

- **Index** folders you choose: `.txt`/`.md`/code/PDF/DOCX, filtered aggressively
  (media, binaries, archives skipped). Incremental — only changed files are
  re-embedded, and a file watcher keeps the index live.
- **Hybrid semantic search** across everything: meaning-based (embeddings) and
  exact-keyword (BM25) retrieval fused together, then re-ordered by a local
  cross-encoder reranker — so both "that note about my lease" and
  "192.168.1.40" find the right file. Similarity scores, open/reveal.
- **Find duplicates** (content-hash) and **organize folders** (by type or date).
- Every mutating action shows a preview first; deletions go to the recycle bin.

## What lights up with [Ollama](https://ollama.com) (free, local)

- **Chat with your files** — streamed answers grounded in retrieved passages,
  with clickable citations and multi-turn follow-ups ("and where are the
  winter tires?"). Without Ollama you still get the best matching passages,
  just not synthesized prose.
- **AI rename** (descriptive names from content) and **summaries**.

```sh
ollama pull llama3.2:3b   # any chat model works; pick it in Settings
```

## Stack

| Layer | Choice |
|---|---|
| Shell | Electron + React + Vite + Tailwind ([electron-vite](https://electron-vite.org)) |
| Engine | Python sidecar (FastAPI) on `127.0.0.1`, spawned & owned by the main process |
| Embeddings | `bge-small-en-v1.5` (int8 ONNX via fastembed, ~130MB, fully local) |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX, ~80MB, fully local) |
| Vector + FTS store | LanceDB (embedded, on-disk; native BM25 full-text index) |
| LLM | Ollama (local, optional) |

The renderer is fully sandboxed and never touches the network or filesystem —
everything flows renderer → IPC → main → local HTTP → sidecar. File mutations
execute only in the main process, only after explicit confirmation.

## Development

Prerequisites: Node 20+, Python 3.10+. (End users need neither — installers
bundle a standalone Python runtime and set everything up on first launch.)

```sh
npm install            # JS deps (downloads Electron)
npm run sidecar:setup  # creates sidecar/.venv with Python deps
npm run dev            # launch with HMR
```

The first index downloads the embedding model (one time, ~130MB). Useful
scripts: `npm run typecheck`, `npm run build`, `npm run sidecar:eval`
(retrieval-correctness check: 12 probe queries over a 20-file corpus),
`npm run dist:win|mac|linux` (installers via electron-builder).

## License

TBD
