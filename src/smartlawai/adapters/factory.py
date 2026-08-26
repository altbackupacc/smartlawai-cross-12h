"""Backend factory: SMARTLAW_BACKEND=local|gcloud."""
from __future__ import annotations

import os

from .base import BackendInterface


def get_backend(name: str | None = None) -> BackendInterface:
    name = (name or os.environ.get("SMARTLAW_BACKEND", "local")).lower()
    if name == "gcloud":
        from .gcloud import GCloudBackend
        return GCloudBackend()
    if name == "local":
        from .local import LocalBackend
        return LocalBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'local' or 'gcloud'.")
