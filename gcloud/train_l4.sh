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
#   TASK=encoder ENTRYPOINT=train.encoders.finetune_inlegalbert \
#     DATA=data/encoders/doc_cls.jsonl ENCODER_TASK=seq SEED=42 ./gcloud/train_l4.sh
#   TASK=qlora   ENTRYPOINT=train.qlora.train_qlora \
#     DATA=data/qlora/smoke_test.jsonl ARM=C SEED=42 ./gcloud/train_l4.sh
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
DATA="${DATA:-}"
ENCODER_TASK="${ENCODER_TASK:-seq}"   # only used when TASK=encoder: seq|token
JOB_NAME="smartlaw-${TASK}-${ARM:-na}-seed${SEED}-$(date +%Y%m%dT%H%M%S)"

if [[ "$ARM" == "L" ]]; then
  echo "Refusing: Arm L is A100-only (CLAUDE.md #2). Use ./gcloud/train_a100.sh instead." >&2
  exit 1
fi

# train_qlora.py and finetune_inlegalbert.py take different flags (--arm vs --task) --
# build the arg list per task rather than assuming one shape fits both.
if [[ "$TASK" == "qlora" ]]; then
  ARGS="${ENTRYPOINT}"
  [[ -n "$DATA" ]] && ARGS="${ARGS},${DATA}"
  ARGS="${ARGS},--seed=${SEED},--arm=${ARM},--device=cuda,--gcs-bucket=${GCS_BUCKET},--run-id=${JOB_NAME}"
elif [[ "$TASK" == "encoder" ]]; then
  : "${DATA:?set DATA for encoder jobs, e.g. DATA=data/encoders/doc_cls.jsonl}"
  ARGS="${ENTRYPOINT},${DATA},--task=${ENCODER_TASK},--seed=${SEED},--device=cuda,--gcs-bucket=${GCS_BUCKET},--run-id=${JOB_NAME}"
else
  echo "Unknown TASK: ${TASK} (expected encoder|qlora)" >&2
  exit 1
fi

gcloud ai custom-jobs create \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE}" \
  --args="${ARGS}" \
  --env-vars="HF_TOKEN=${HF_TOKEN},GCS_BUCKET=${GCS_BUCKET},RUN_ID=${JOB_NAME}"
# Dockerfile.train's ENTRYPOINT is ["python", "-m"], so --args becomes
# `python -m ${ENTRYPOINT} [DATA] --seed=... --device=cuda ...` — no dispatcher needed.

echo "Submitted: ${JOB_NAME}"
echo "Track it:  gcloud ai custom-jobs list --region=${REGION} --filter=displayName=${JOB_NAME}"
echo "Vertex AI Custom Jobs self-terminate on completion or failure — no manual"
echo "teardown needed for this job (see gcloud/teardown.sh for what DOES need one)."
