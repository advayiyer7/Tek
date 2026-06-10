# Tek

**A local-first AI agent for your files.**

Tek indexes your files on-device, lets you semantically search and chat across
everything, and can take file actions (organize, rename, dedupe, tag, summarize)
on request — always with a preview and your explicit confirmation.

**Privacy is the core feature.** By default, nothing leaves your machine: local
embeddings, local vector store (LanceDB), and local inference via Ollama. Cloud
models are strictly opt-in.

> **Status: Phase 1** — Electron shell + Python sidecar wiring. See
> [PROJECT.md](PROJECT.md) for the full build plan.

## Stack

- **Shell:** Electron + React + Vite + Tailwind (via [electron-vite](https://electron-vite.org))
- **Sidecar:** Python (FastAPI) spawned by the Electron main process, bound to
  `127.0.0.1` on a dynamic port — owns extraction, chunking, embeddings, vector
  search, and LLM calls
- **Vector store:** LanceDB &nbsp;·&nbsp; **Embeddings:** local sentence-transformers-class model &nbsp;·&nbsp; **LLM:** Ollama (local, default)

## Development

Prerequisites: Node 20+, Python 3.10+.

```sh
npm install            # JS dependencies (downloads Electron)
npm run sidecar:setup  # creates sidecar/.venv and installs Python deps
npm run dev            # launches the app with HMR
```

The app window opens, spawns the Python sidecar, and shows a live status pill.
Type a message in the console to round-trip it: renderer → Electron main → Python
sidecar → back, with latency.

Other scripts:

```sh
npm run typecheck      # TypeScript checks (main/preload + renderer)
npm run build          # production bundle to out/
```

## License

TBD
