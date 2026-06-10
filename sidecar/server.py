"""Tek sidecar — local Python service owned by the Electron main process.

Owns ingestion (extract -> chunk -> embed -> LanceDB), retrieval, RAG chat via
Ollama, and file-action PLANNING (execution happens in Electron main after
explicit user confirmation).

Binds to 127.0.0.1 only; port and data dir are passed by the main process.
File contents never leave this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tek import SIDECAR_VERSION
from tek import actions as actions_mod
from tek import ollama as ollama_mod
from tek import rag
from tek.config import Config
from tek.embed import FastEmbedEmbedder
from tek.indexer import Indexer
from tek.rerank import Reranker
from tek.store import Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tek.server")

_started_at = time.monotonic()

app = FastAPI(title="Tek Sidecar", version=SIDECAR_VERSION)

# Populated in main() before uvicorn starts.
config: Config
embedder: FastEmbedEmbedder
store: Store
indexer: Indexer
reranker: Reranker


def _active_reranker() -> Reranker | None:
    return reranker if config.settings.rerank_enabled else None


# ---------------------------------------------------------------- models ----


class SettingsUpdate(BaseModel):
    folders: list[str] | None = None
    llm_model: str | None = None
    watch_enabled: bool | None = None
    rerank_enabled: bool | None = None


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    k: int = Field(default=8, ge=1, le=20)


class DedupeRequest(BaseModel):
    folder: str


class OrganizeRequest(BaseModel):
    folder: str
    strategy: str = Field(default="by-type", pattern="^(by-type|by-date)$")


class RenameRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=50)


class SummarizeRequest(BaseModel):
    path: str


# -------------------------------------------------------------- endpoints ---


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "tek-sidecar",
        "version": SIDECAR_VERSION,
        "python": platform.python_version(),
        "uptimeS": round(time.monotonic() - _started_at, 1),
        "embedModel": {
            "name": embedder.model_name,
            "ready": embedder.is_ready,
            "loading": embedder.loading,
            "error": embedder.load_error,
        },
        "reranker": {
            "name": reranker.model_name,
            "ready": reranker.is_ready,
            "enabled": config.settings.rerank_enabled,
        },
        "index": store.stats(),
    }


@app.get("/settings")
def get_settings() -> dict:
    return config.settings.model_dump()


@app.put("/settings")
def put_settings(update: SettingsUpdate) -> dict:
    changes = {k: v for k, v in update.model_dump().items() if v is not None}
    if "folders" in changes:
        bad = [f for f in changes["folders"] if not Path(f).is_dir()]
        if bad:
            raise HTTPException(400, detail=f"not a directory: {bad[0]}")
        changes["folders"] = [str(Path(f).resolve()) for f in changes["folders"]]
    settings = config.update(**changes)
    if "folders" in changes:
        # Watcher set changes with the folder list; restart it.
        indexer.stop_watcher()
        if settings.folders:
            indexer.start_watcher()
    return settings.model_dump()


@app.post("/index/start")
def index_start() -> dict:
    if not config.settings.folders:
        raise HTTPException(400, detail="no folders configured")
    started = indexer.start_full_index()
    if started and config.settings.watch_enabled:
        indexer.start_watcher()
    return {"started": started, "alreadyRunning": not started}


@app.get("/index/status")
def index_status() -> dict:
    snap = indexer.progress.snapshot()
    snap["stats"] = store.stats()
    snap["running"] = indexer.running
    return snap


@app.post("/search")
def search(req: SearchRequest) -> dict:
    # Sync endpoint → FastAPI runs it in a worker thread, so lazy model load
    # (first run downloads ~130MB) doesn't block the event loop.
    started = time.perf_counter()
    hits = rag.retrieve(store, embedder, req.query, k=req.k, reranker=_active_reranker())
    return {
        "results": [
            {
                "path": h["path"],
                "name": h["name"],
                "chunkIndex": h["chunk_index"],
                "text": h["text"],
                "score": h["score"],
            }
            for h in hits
        ],
        "tookMs": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def stream():
        async for event in rag.answer_stream(
            store,
            embedder,
            config.settings.llm_model,
            req.question,
            history=[t.model_dump() for t in req.history],
            k=req.k,
            reranker=_active_reranker(),
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/ollama/status")
async def ollama_status() -> dict:
    result = await ollama_mod.status()
    result["configuredModel"] = config.settings.llm_model
    return result


@app.post("/actions/dedupe")
async def dedupe(req: DedupeRequest) -> dict:
    if not Path(req.folder).is_dir():
        raise HTTPException(400, detail="folder does not exist")
    return await asyncio.to_thread(actions_mod.dedupe_scan, req.folder)


@app.post("/actions/organize")
async def organize(req: OrganizeRequest) -> dict:
    return await asyncio.to_thread(actions_mod.organize_plan, req.folder, req.strategy)


@app.post("/actions/rename")
async def rename(req: RenameRequest) -> dict:
    llm = await ollama_mod.status()
    if not llm["available"]:
        raise HTTPException(409, detail="AI rename needs Ollama running")
    return await actions_mod.rename_plan(req.paths, config.settings.llm_model)


@app.post("/actions/summarize")
async def summarize(req: SummarizeRequest) -> dict:
    if not Path(req.path).is_file():
        raise HTTPException(400, detail="file does not exist")
    llm = await ollama_mod.status()
    if not llm["available"]:
        raise HTTPException(409, detail="summaries need Ollama running")
    return await actions_mod.summarize(req.path, config.settings.llm_model)


# ------------------------------------------------------------------ main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek Python sidecar")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", required=True, help="writable dir for settings + index")
    args = parser.parse_args()

    global config, embedder, store, indexer, reranker
    config = Config(Path(args.data_dir))
    embedder = FastEmbedEmbedder(config.settings.embed_model, str(config.models_dir))
    store = Store(config.db_dir, dim=embedder.dim)
    indexer = Indexer(config=config, embedder=embedder, store=store)
    reranker = Reranker(config.settings.rerank_model, str(config.models_dir))
    if config.settings.folders and config.settings.watch_enabled:
        indexer.start_watcher()

    print(f"TEK_SIDECAR_STARTING host={args.host} port={args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
