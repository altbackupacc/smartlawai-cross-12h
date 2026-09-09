#!/usr/bin/env bash
# Submit an M2 retrieval benchmark or dense indexing job to Vertex AI Custom
# Jobs on a single GCP L4 (default) or A100 GPU. DRAFT-ONLY: written for a human
# to review and run (CLAUDE.md #2, OPS.md #9).
#
# Prereqs (one-time, human-run):
#   gcloud auth login
#   gcloud auth application-default login
#   gcloud config set project <PROJECT_ID>
#
# Usage:
#   # Run on default L4 GPU:
#   ./gcloud/run_m2_retrieval.sh
#
#   # Run on A100 GPU:
#   ACCEL=A100 ./gcloud/run_m2_retrieval.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-smartlawai-1}"
REGION="${REGION:-us-central1}"
REPO="smartlaw"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/train:latest"
GCS_BUCKET="${GCS_BUCKET:-smartlawai-1-checkpoints}"
HF_TOKEN="${HF_TOKEN:-not_needed}"

ACCEL="${ACCEL:-A100}"
JOB_NAME="smartlaw-m2-retrieval-${ACCEL,,}-$(date +%Y%m%dT%H%M%S)"

if [[ "$ACCEL" == "A100" ]]; then
  MACHINE_TYPE="a2-highgpu-1g"
  ACCEL_TYPE="NVIDIA_TESLA_A100"
  ACCEL_COUNT=1
elif [[ "$ACCEL" == "L4" ]]; then
  MACHINE_TYPE="g2-standard-8"
  ACCEL_TYPE="NVIDIA_L4"
  ACCEL_COUNT=1
else
  echo "Unknown ACCEL: ${ACCEL}. Expected L4 or A100." >&2
  exit 1
fi

echo "============================================================"
echo "SmartLawAI M2 Retrieval Job (GCP ${ACCEL})"
echo "Machine: ${MACHINE_TYPE} with 1x ${ACCEL_TYPE}"
echo "Region:  ${REGION}"
echo "Project: ${PROJECT}"
echo "Job:     ${JOB_NAME}"
echo "============================================================"

CONFIG_FILE="$(mktemp)"
trap 'rm -f "$CONFIG_FILE"' EXIT

cat <<EOF > "$CONFIG_FILE"
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCEL_TYPE}
      acceleratorCount: ${ACCEL_COUNT}
    replicaCount: 1
    containerSpec:
      imageUri: "${IMAGE}"
      args:
        - "scripts.run_retrieval_bench"
        - "--device=cuda"
        - "--batch-size=32"
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
        - name: GCS_BUCKET
          value: "${GCS_BUCKET}"
        - name: RUN_ID
          value: "${JOB_NAME}"
EOF

gcloud ai custom-jobs create \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --config="$CONFIG_FILE"

echo "[+] Successfully submitted: ${JOB_NAME}"
echo "    Monitor progress: gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION}"
