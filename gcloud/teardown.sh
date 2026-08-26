#!/usr/bin/env bash
# Tear down GCP resources that do NOT self-terminate. Vertex AI Custom Jobs
# (train_l4.sh, train_a100.sh) already self-terminate on completion or failure and
# do not need this. What DOES need it: Cloud Run services (M3's serving endpoint,
# deployed by gcloud/deploy.sh), which stay up — and billed — until deleted.
#
# DRAFT-ONLY: this is written for a human to review and run — Claude Code does not
# execute this itself, and this script deletes a live, possibly billed resource, so
# it asks for confirmation before doing anything (CLAUDE.md #2, OPS.md #9).
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${REGION:-us-central1}"

echo "Cloud Run services currently deployed in ${REGION}:"
gcloud run services list --project="$PROJECT" --region="$REGION"

read -r -p "Delete mistral-vllm and smartlaw-app? [y/N] " CONFIRM
if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
  gcloud run services delete mistral-vllm --project="$PROJECT" --region="$REGION" --quiet || true
  gcloud run services delete smartlaw-app --project="$PROJECT" --region="$REGION" --quiet || true
  echo "Deleted. Verify nothing GPU-backed remains:"
  gcloud run services list --project="$PROJECT" --region="$REGION"
else
  echo "Aborted — nothing deleted."
fi

echo
echo "Also check for any Custom Job that appears stuck (should be rare — they"
echo "self-terminate):"
echo "  gcloud ai custom-jobs list --region=${REGION} --filter='state=JOB_STATE_RUNNING'"
