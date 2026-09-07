#!/usr/bin/env bash
# GCloud deploy: SaulLM-7B (vLLM) to Cloud Run GPU, for M7's SaulLM baseline.
# Parallel to deploy.sh (which deploys Mistral) -- same shape, different model.
# DRAFT-ONLY (OPS.md #9, CLAUDE.md #2): this script is written for a human to
# read and run. Claude never invokes `gcloud run deploy` itself.
#
# Prereqs: gcloud auth login; gcloud config set project <PROJECT_ID>
#
# ⚠️ OPEN ITEM before running this: confirm the exact HF repo id and
# license/gating status for SaulLM-7B-Instruct (SAULLM_HF_MODEL below is a
# placeholder pending that check, per baselines/config.py's own note) --
# gated models need HF_TOKEN passed through the same way deploy.sh does for
# Mistral.
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-central1}"
REPO="smartlaw"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
SAULLM_HF_MODEL="${SAULLM_HF_MODEL:-Equall/Saul-7B-Instruct-v1}"  # UNVERIFIED, confirm before running

gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" || true
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker pull vllm/vllm-openai:latest
docker tag vllm/vllm-openai:latest "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/saullm-vllm:latest"
docker push "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/saullm-vllm:latest"
gcloud run deploy saullm-vllm \
  --image "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/saullm-vllm:latest" \
  --region "$REGION" --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
  --no-cpu-throttling --max-instances 1 --port 8000 --allow-unauthenticated \
  --set-env-vars "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  --args="--model=${SAULLM_HF_MODEL},--max-model-len=8192"

SAULLM_URL="$(gcloud run services describe saullm-vllm --region "$REGION" --format='value(status.url)')"

echo "SaulLM endpoint: ${SAULLM_URL}"
echo "Set in .env: SAULLM_BASE_URL=${SAULLM_URL}/v1"
echo "Set in .env: SAULLM_MODEL=${SAULLM_HF_MODEL}"

# Tear down when done -- an L4 Cloud Run service with min-instances unset
# scales to zero when idle, but confirm via gcloud/teardown.sh or the console
# rather than assuming, per CLAUDE.md #2's "tear down instances the moment a
# job finishes."
