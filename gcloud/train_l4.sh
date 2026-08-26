#!/usr/bin/env bash
# Submit an M8a encoder or M8b QLoRA (Arm S/C/M) training job to Vertex AI Custom
# Jobs on a single L4 GPU. DRAFT-ONLY: this is written for a human to review and run
# — Claude Code does not execute this itself (CLAUDE.md #2, OPS.md #9).
#
# Prereqs (one-time, human-run):
#   gcloud auth login
#   gcloud auth application-default login
#   gcloud config set project <PROJECT_ID>
#
# Usage:
#   TASK=encoder ENTRYPOINT=train.encoders.doc_cls SEED=42 ./gcloud/train_l4.sh
#   TASK=qlora   ENTRYPOINT=train.qlora.train_qlora ARM=C SEED=42 ./gcloud/train_l4.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-central1}"
REPO="smartlaw"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/train:latest"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
GCS_BUCKET="${GCS_BUCKET:?set GCS_BUCKET}"

TASK="${TASK:?set TASK (encoder|qlora)}"
ENTRYPOINT="${ENTRYPOINT:?set ENTRYPOINT, e.g. train.qlora.train_qlora}"
ARM="${ARM:-}"
SEED="${SEED:?set SEED}"
JOB_NAME="smartlaw-${TASK}-${ARM:-na}-seed${SEED}-$(date +%Y%m%dT%H%M%S)"

if [[ "$ARM" == "L" ]]; then
  echo "Refusing: Arm L is A100-only (CLAUDE.md #2). Use ./gcloud/train_a100.sh instead." >&2
  exit 1
fi

gcloud ai custom-jobs create \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE}" \
  --args="${ENTRYPOINT},--seed=${SEED},--arm=${ARM},--device=cuda,--gcs-bucket=${GCS_BUCKET},--run-id=${JOB_NAME}" \
  --env-vars="HF_TOKEN=${HF_TOKEN},GCS_BUCKET=${GCS_BUCKET},RUN_ID=${JOB_NAME}"
# Dockerfile.train's ENTRYPOINT is ["python", "-m"], so --args becomes
# `python -m ${ENTRYPOINT} --seed=... --arm=... --device=cuda ...` — no dispatcher needed.

echo "Submitted: ${JOB_NAME}"
echo "Track it:  gcloud ai custom-jobs list --region=${REGION} --filter=displayName=${JOB_NAME}"
echo "Vertex AI Custom Jobs self-terminate on completion or failure — no manual"
echo "teardown needed for this job (see gcloud/teardown.sh for what DOES need one)."
