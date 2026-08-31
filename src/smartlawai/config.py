"""Every threshold, model id, and path lives here. No magic numbers elsewhere
(CLAUDE.md §4). M0 migrates only what the skeleton itself needs; the constants
still scattered through core/*.py are migrated as those files are rewritten in
M2/M3/M5, not swept wholesale here."""
from __future__ import annotations

import os

LOCAL_DATA_DIR: str = os.environ.get("SMARTLAW_LOCAL_DIR", "./.smartlaw_local")

RETRIEVE_TOP_K: int = 20
RERANK_TOP_K: int = 5
RERANK_FLOOR: float = -10.0  # migrated verbatim from core/rag.py; recalibrating the
                              # value against the labelled OOS set is M5's job (PLAN.md M5.6)
FAITHFULNESS_THRESHOLD: float = 0.70  # M0 stub verifier's own line only

STUB_ENCODER_MODEL_ID: str = "stub-encoder-v0"
STUB_RERANKER_MODEL_ID: str = "stub-reranker-v0"
STUB_GENERATOR_MODEL_ID: str = "stub-generator-v0"
STUB_VERIFIER_MODEL_ID: str = "stub-verifier-v0"

DEFAULT_OWNER_ID: str = "anon"
