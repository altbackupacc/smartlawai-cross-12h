#!/usr/bin/env bash
# Smoke test script for Module 3 (Structured Generation) against a live
# Cloud Run GPU or Vertex AI vLLM endpoint.
#
# Prereqs:
#   gcloud auth login
#   export MISTRAL_BASE_URL="https://<mistral-vllm-url>/v1"
#   export MISTRAL_API_KEY="<token-if-authenticated>"
#
# Usage:
#   ./gcloud/smoke_test_m3.sh
set -euo pipefail

BASE_URL="${MISTRAL_BASE_URL:-http://localhost:8000/v1}"
MODEL="${MISTRAL_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
API_KEY="${MISTRAL_API_KEY:-not-needed}"
DEVICE="${DEVICE:-cuda}"

echo "Running M3 Structured Generation smoke test against: ${BASE_URL}"
python scripts/smoke_test_m3.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --api-key "${API_KEY}" \
  --device "${DEVICE}"
