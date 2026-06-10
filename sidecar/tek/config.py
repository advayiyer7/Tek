"""Sidecar configuration: data directory layout and persisted settings.

The Electron main process passes --data-dir (a folder under the app's
userData). Everything Tek persists — settings, the LanceDB store, model
caches — lives there. File contents never leave the machine.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434"


class Settings(BaseModel):
    """User-facing settings persisted to settings.json in the data dir."""

    folders: list[str] = Field(default_factory=list)
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    watch_enabled: bool = True
    # Cloud is strictly opt-in and unused unless explicitly enabled. Keys are
    # never stored here — they live in the OS keychain via Electron.
    cloud_enabled: bool = False


class Config:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_dir = data_dir / "lancedb"
        self.models_dir = data_dir / "models"
        self.settings_path = data_dir / "settings.json"
        for d in (self.data_dir, self.db_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.settings = self._load()

    def _load(self) -> Settings:
        if self.settings_path.exists():
            try:
                return Settings.model_validate_json(self.settings_path.read_text("utf-8"))
            except Exception:
                # Corrupt settings should never brick the app; fall back to
                # defaults but keep the broken file for inspection.
                self.settings_path.rename(self.settings_path.with_suffix(".json.bad"))
        return Settings()

    def update(self, **changes: object) -> Settings:
        with self._lock:
            self.settings = self.settings.model_copy(update=changes)
            self.settings_path.write_text(self.settings.model_dump_json(indent=2), "utf-8")
            return self.settings
