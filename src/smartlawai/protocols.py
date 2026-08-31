"""Stub encoder/generator/verifier seams: M2/M3/M5 swap in real implementations
conforming to these Protocols without changing pipeline.py's call sites.

I1 enforcement deliberately lives outside these Protocols -- Encoder/Reranker/
Generator/Verifier are pure ML-shaped calls with no scope awareness. The actual
boundary is BackendInterface.fetch_chunks_scoped(scope), called by pipeline.py's
retrieve stage before any Protocol implementation ever sees a candidate chunk."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from smartlawai.adapters.base import Chunk, RetrievedChunk


@runtime_checkable
class Encoder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[RetrievedChunk]: ...


@dataclass
class Claim:
    text: str
    passage_ids: list[str]
    citations: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    claims: list[Claim]
    unanswerable_aspects: list[str]
    model_id: str


@runtime_checkable
class Generator(Protocol):
    def generate(self, question: str, passages: list[RetrievedChunk]) -> GenerationResult: ...


@dataclass
class VerificationResult:
    status: str  # "ok" | "unavailable" -- I2: never a silently substituted score
    score: float | None
    reason: str = ""


@runtime_checkable
class Verifier(Protocol):
    def verify(self, claim: Claim, passages: list[RetrievedChunk]) -> VerificationResult: ...
