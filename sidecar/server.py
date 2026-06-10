"""Tek sidecar — local Python service owned by the Electron main process.

Phase 1: a minimal FastAPI app proving the renderer -> main -> sidecar
round-trip. Later phases add ingestion (extract/chunk/embed/store), retrieval,
and generation here.

Binds to 127.0.0.1 only; the port is chosen by the Electron main process and
passed via --port. Never expose this service on a public interface.
"""

import argparse
import platform
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

SIDECAR_VERSION = "0.1.0"
_started_at = time.monotonic()

app = FastAPI(title="Tek Sidecar", version=SIDECAR_VERSION)


class EchoRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "tek-sidecar",
        "version": SIDECAR_VERSION,
        "python": platform.python_version(),
        "uptime_s": round(time.monotonic() - _started_at, 1),
    }


@app.post("/echo")
def echo(req: EchoRequest) -> dict:
    return {
        "reply": f"sidecar received: {req.message!r}",
        "length": len(req.message),
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek Python sidecar")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"TEK_SIDECAR_STARTING host={args.host} port={args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
