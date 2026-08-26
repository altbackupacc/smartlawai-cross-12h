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
ARGV=("$ENTRYPOINT")
if [[ "$TASK" == "qlora" ]]; then
  [[ -n "$DATA" ]] && ARGV+=("$DATA")
  ARGV+=("--seed=${SEED}" "--arm=${ARM}" "--device=cuda" "--gcs-bucket=${GCS_BUCKET}" "--run-id=${JOB_NAME}")
elif [[ "$TASK" == "encoder" ]]; then
  : "${DATA:?set DATA for encoder jobs, e.g. DATA=data/encoders/doc_cls.jsonl}"
  ARGV+=("$DATA" "--task=${ENCODER_TASK}" "--seed=${SEED}" "--device=cuda" "--gcs-bucket=${GCS_BUCKET}" "--run-id=${JOB_NAME}")
else
  echo "Unknown TASK: ${TASK} (expected encoder|qlora)" >&2
  exit 1
fi
# One-off flag overrides not worth hardcoding here, e.g.
# EXTRA_ARGS="--save-steps=1 --epochs=1" for a smoke test (space-separated -- this is
# a bash array, not the comma-joined string the old --worker-pool-spec form used):
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGV+=($EXTRA_ARGS)
fi

# gcloud's --worker-pool-spec flag shorthand has no env-vars key -- HF_TOKEN/GCS_BUCKET/
# RUN_ID need to reach the container's environment (checkpointing.py's HF Hub push
# relies on HF_TOKEN being set, not just passed as a CLI arg), so the job is submitted
# via a generated --config YAML instead, which supports containerSpec.env directly.
CONFIG_FILE="$(mktemp)"
trap 'rm -f "$CONFIG_FILE"' EXIT

{
  echo "workerPoolSpecs:"
  echo "  - machineSpec:"
  echo "      machineType: g2-standard-8"
  echo "      acceleratorType: NVIDIA_L4"
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
