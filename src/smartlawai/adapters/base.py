"""Storage + retrieval contract. The portable core depends ONLY on this module.
Concrete backends: local.py (DuckDB + filesystem) and gcloud.py (Postgres + GCS).
No cloud SDKs are imported here, so swapping backends never touches business logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import faiss
import numpy as np

from smartlawai.scope import Scope


# --------------------------------------------------------------------------- #
# Data transfer objects
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    doc_id: str
    filename: str
    doc_type: str
    language: str
    source: str
    storage_uri: str
    num_pages: int = 0
    ocr_applied: bool = False
    raw_text: str = ""
    session_id: Optional[str] = None
    owner_id: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    char_start: int
    char_end: int
    parent_chunk_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    bm25_indexed: bool = False
    owner_id: str = ""


@dataclass
class Clause:
    clause_id: str
    doc_id: str
    chunk_id: Optional[str]
    clause_type: str
    clause_text: str
    span_start: int
    span_end: int
    risk_tier: str
    risk_score: float
    statute_ref: Optional[str] = None


@dataclass
class Summary:
    summary_id: str
    doc_id: str
    section_label: str
    level: str  # SECTION | FINAL
    summary_text: str
    model: str
    faithfulness_score: Optional[float] = None


@dataclass
class AuditEvent:
    event_id: str
    session_id: str
    event_type: str
    endpoint: str
    query_text: str = ""
    response_text: str = ""
    faithfulness_score: Optional[float] = None
    pii_redacted: bool = False


@dataclass
class IngestionRecord:
    ingest_id: str
    doc_id: str
    stage_path: str
    ocr_engine: str
    ocr_status: str
    lang_detected: str
    preprocess_status: str = "PENDING"
    error_msg: Optional[str] = None


@dataclass
class EvalResult:
    eval_id: str
    task: str
    system_name: str
    metric: str
    value: float
    split: str = "test"
    n_examples: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class BackendInterface(ABC):
    # ---------- relational / table ops ----------
    @abstractmethod
    def upload_doc(self, local_path: str, doc: Document) -> str:
        """Persist the physical file to blob storage and insert a document row.
        Returns the storage URI."""

    @abstractmethod
    def store_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def store_clauses(self, clauses: list[Clause]) -> None: ...

    @abstractmethod
    def store_summaries(self, summaries: list[Summary]) -> None: ...

    @abstractmethod
    def store_audit(self, event: AuditEvent) -> None: ...

    @abstractmethod
    def store_ingestion(self, rec: IngestionRecord) -> None: ...

    @abstractmethod
    def store_eval(self, results: list[EvalResult]) -> None: ...

    @abstractmethod
    def mark_preprocessed(self, doc_id: str, status: str) -> None: ...

    @abstractmethod
    def fetch_document_text(self, doc_id: str) -> str: ...

    @abstractmethod
    def fetch_chunks(self, doc_id: str) -> list[Chunk]: ...

    @abstractmethod
    def fetch_all_chunks(self) -> list[Chunk]: ...

    @abstractmethod
    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]: ...

    def fetch_chunks_scoped(self, scope: Scope) -> list[Chunk]:
        """The ONLY retrieval path serving code may call (I1). Concrete, not
        abstract: base default raises so each backend opts in explicitly without
        being forced to implement it just to remain instantiable."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fetch_chunks_scoped (I1)")

    @abstractmethod
    def list_doc_ids(self) -> list[str]: ...

    @abstractmethod
    def update_chunk_embeddings(self, pairs: list[tuple[str, list[float]]]) -> None:
        """pairs = [(chunk_id, embedding_vector), ...]"""

    @abstractmethod
    def fetch_annotations(self, split: str) -> list[dict]:
        """For benchmark export (Phase 8)."""

    # ---------- FAISS index lifecycle ----------
    @abstractmethod
    def _write_index_bytes(self, key: str, data: bytes) -> None:
        """Backend-specific: persist raw bytes (disk / GCS)."""

    @abstractmethod
    def _read_index_bytes(self, key: str) -> bytes:
        """Backend-specific: fetch raw bytes."""

    @abstractmethod
    def faiss_index_exists(self, key: str) -> bool: ...

    def save_faiss_index(self, index: faiss.Index, key: str) -> None:
        """Serialize in-memory index -> bytes -> persistent storage.
        Identical across all backends; only _write_index_bytes differs."""
        data = faiss.serialize_index(index).tobytes()
        self._write_index_bytes(key, data)
        self._load_faiss_cached.cache_clear()

    def load_faiss_index(self, key: str) -> faiss.Index:
        """Fetch bytes -> deserialize -> return an IN-MEMORY index.
        Cached per process so we pay the storage round-trip only once."""
        return self._load_faiss_cached(key)

    @lru_cache(maxsize=8)
    def _load_faiss_cached(self, key: str) -> faiss.Index:
        raw = self._read_index_bytes(key)
        arr = np.frombuffer(raw, dtype="uint8")
        return faiss.deserialize_index(arr)

    # ---------- helpers usable by all backends ----------
    @staticmethod
    def new_flat_ip_index(dim: int) -> faiss.Index:
        """Exact cosine/inner-product index. Ideal for <=5K-doc benchmarks."""
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
