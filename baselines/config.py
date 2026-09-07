"""Every M7 baseline model id, endpoint default, and cutoff date lives here.
No magic numbers outside this module (CLAUDE.md #4), mirroring
src/smartlawai/config.py's own shape."""
from __future__ import annotations

import os
from datetime import date

# ----- Model identities -----
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"   # matches gcloud/deploy.sh

# UNVERIFIED -- confirm the exact HF repo id and license/gating status for
# SaulLM-7B-Instruct before finalizing gcloud/deploy_saullm.sh. This is a
# placeholder, not a confirmed value.
SAULLM_MODEL_ID = "Equall/Saul-7B-Instruct-v1"

# ----- SaulLM endpoint (OpenAI-compatible vLLM, same shape as Mistral's) -----
SAULLM_BASE_URL = os.environ.get("SAULLM_BASE_URL", "http://localhost:11435/v1")
SAULLM_API_KEY = os.environ.get("SAULLM_API_KEY", "not-needed")

# ----- Frontier API (Anthropic) -----
FRONTIER_PROVIDER = os.environ.get("FRONTIER_PROVIDER", "anthropic")
FRONTIER_MODEL_ID = os.environ.get("FRONTIER_MODEL", "")   # required at call time, no silent default

# ----- Retrieval sizing (kept local to baselines/, not imported from
# smartlawai.config, since M2 may later recalibrate that module's values for
# the *served* pipeline without those changes silently reaching baselines) -----
RAG_RETRIEVE_TOP_K = 20
RAG_RERANK_TOP_K = 5
BM25_TOP_K = 20

# ----- T1 contamination check: model training-cutoff dates -----
# Human-verified from each provider's own model card / release announcement --
# never invented. Left empty until sourced and cited (e.g. in the commit that
# populates each entry). baselines.contamination.split_by_cutoff() raises
# KeyError for any model_id missing here rather than silently skipping the
# split -- an unconfigured cutoff is a gap to fill deliberately.
MODEL_CUTOFF_DATES: dict[str, date] = {
    # MISTRAL_MODEL_ID: date(...),   # source: <provider announcement/model card URL>
    # SAULLM_MODEL_ID: date(...),    # source: <provider announcement/model card URL>
    # FRONTIER_MODEL_ID: date(...),  # source: <provider announcement/model card URL>
}
