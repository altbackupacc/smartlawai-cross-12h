#!/usr/bin/env bash
# GCloud deploy: app to Cloud Run, Mistral (vLLM) to Cloud Run GPU.
# Prereqs: gcloud auth login; gcloud config set project <PROJECT_ID>
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-central1}"
REPO="smartlaw"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com sqladmin.googleapis.com
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" || true
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# 1) Mistral via vLLM on an L4 GPU
docker pull vllm/vllm-openai:latest
docker tag vllm/vllm-openai:latest "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/mistral-vllm:latest"
docker push "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/mistral-vllm:latest"
gcloud run deploy mistral-vllm \
  --image "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/mistral-vllm:latest" \
  --region "$REGION" --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
  --no-cpu-throttling --max-instances 1 --port 8000 --allow-unauthenticated \
  --set-env-vars "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  --args="--model=mistralai/Mistral-7B-Instruct-v0.3,--max-model-len=8192"

MISTRAL_URL="$(gcloud run services describe mistral-vllm --region "$REGION" --format='value(status.url)')"

# 2) The SmartLawAI app (no GPU)
docker build -f docker/Dockerfile.api -t "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/app:latest" .
docker push "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/app:latest"
gcloud run deploy smartlaw-app \
  --image "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/app:latest" \
  --region "$REGION" --cpu 4 --memory 8Gi --port 8000 --allow-unauthenticated \
  --set-env-vars "SMARTLAW_BACKEND=gcloud,MISTRAL_BASE_URL=${MISTRAL_URL}/v1,MISTRAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3,GCS_BUCKET=${GCS_BUCKET},GCP_PROJECT=${PROJECT},PG_HOST=${PG_HOST},PG_DB=${PG_DB},PG_USER=${PG_USER},PG_PASSWORD=${PG_PASSWORD}"

echo "App: $(gcloud run services describe smartlaw-app --region "$REGION" --format='value(status.url)')"
echo "Mistral: ${MISTRAL_URL}"
