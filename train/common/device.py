"""Device resolution and dtype selection. Every training/inference script uses this
instead of hardcoding cuda:0 or assuming bf16 (CLAUDE.md #2)."""
from __future__ import annotations

import torch


def resolve_device(arg: str | None = None) -> torch.device:
    if arg and arg not in ("auto", ""):
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def dtype_for_device() -> torch.dtype:
    """bf16 where the GPU supports it (Ampere+, e.g. L4/A100); fp16 otherwise
    (e.g. Turing T4). Never assume -- always probe."""
    return torch.bfloat16 if supports_bf16() else torch.float16
