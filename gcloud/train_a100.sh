#!/usr/bin/env bash
# Submit an M8b Arm L (Qwen2.5-14B) QLoRA training job to Vertex AI Custom Jobs on a
# single A100 — the heavy tier only (CLAUDE.md #2). Do NOT use this for Arms S/C/M or
# M8a encoders; those run on L4 via ./gcloud/train_l4.sh — A100 is materially more
# expensive per hour and reserved for genuine headroom/speed need.
#
# DRAFT-ONLY: this is written for a human to review and run — Claude Code does not
# execute this itself (CLAUDE.md #2, OPS.md #9).
#
# Prereqs (one-time, human-run):
#   gcloud auth login
#   gcloud auth application-default login
#   gcloud config set project <PROJECT_ID>
#
# Usage:
#   DATA=data/qlora/smoke_test.jsonl SEED=42 ./gcloud/train_a100.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-central1}"
REPO="smartlaw"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/train:latest"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
GCS_BUCKET="${GCS_BUCKET:?set GCS_BUCKET}"

ENTRYPOINT="${ENTRYPOINT:-train.qlora.train_qlora}"
ARM="${ARM:-L}"
SEED="${SEED:?set SEED}"
DATA="${DATA:-}"
JOB_NAME="smartlaw-qlora-${ARM}-seed${SEED}-$(date +%Y%m%dT%H%M%S)"

if [[ "$ARM" != "L" ]]; then
  echo "Refusing: A100 is reserved for Arm L (CLAUDE.md #2). Use ./gcloud/train_l4.sh for Arm ${ARM}." >&2
  exit 1
fi

ARGV=("$ENTRYPOINT")
[[ -n "$DATA" ]] && ARGV+=("$DATA")
ARGV+=("--seed=${SEED}" "--arm=${ARM}" "--device=cuda" "--gcs-bucket=${GCS_BUCKET}" "--run-id=${JOB_NAME}")
# One-off flag overrides not worth hardcoding here, e.g.
# EXTRA_ARGS="--save-steps=1 --epochs=1" (space-separated -- this is a bash array):
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGV+=($EXTRA_ARGS)
fi

# See train_l4.sh for why this uses --config instead of --worker-pool-spec/--env-vars:
# the latter flag doesn't exist, and env vars are needed for HF_TOKEN specifically.
CONFIG_FILE="$(mktemp)"
trap 'rm -f "$CONFIG_FILE"' EXIT

{
  echo "workerPoolSpecs:"
  echo "  - machineSpec:"
  echo "      machineType: a2-highgpu-1g"
  echo "      acceleratorType: NVIDIA_TESLA_A100"
  echo "      acceleratorCount: 1"
  echo "    replicaCount: 1"
  echo "    containerSpec:"
  echo "      imageUri: \"${IMAGE}\""
  echo "      args:"
  for a in "${ARGV[@]}"; do
    printf '        - "%s"\n' "$a"
  done
  echo "      env:"
  echo "        - name: HF_TOKEN"
  echo "          value: \"${HF_TOKEN}\""
  echo "        - name: GCS_BUCKET"
  echo "          value: \"${GCS_BUCKET}\""
  echo "        - name: RUN_ID"
  echo "          value: \"${JOB_NAME}\""
} > "$CONFIG_FILE"

gcloud ai custom-jobs create \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --config="$CONFIG_FILE"

echo "Submitted: ${JOB_NAME}"
echo "Track it:  gcloud ai custom-jobs list --region=${REGION} --filter=displayName=${JOB_NAME}"
echo "Vertex AI Custom Jobs self-terminate on completion or failure — no manual"
echo "teardown needed for this job (see gcloud/teardown.sh for what DOES need one)."
