#!/usr/bin/env bash
# Convenience wrapper: submit the Arm-S QLoRA smoke test with everything already
# decided, so you don't have to reconstruct the full TASK=/ENTRYPOINT=/DATA=/ARM=/
# SEED=/EXTRA_ARGS= invocation from memory each time.
#
# Assumes gcloud/bootstrap_session.sh has already been sourced this session (so
# GCP_PROJECT, REGION, GCS_BUCKET, HF_TOKEN are set).
#
# Usage: ./gcloud/smoke_test.sh
set -euo pipefail

TASK=qlora \
ENTRYPOINT=train.qlora.train_qlora \
DATA=data/qlora/smoke_test.jsonl \
ARM=S \
SEED="${SEED:-42}" \
EXTRA_ARGS="--save-steps=1 --epochs=1" \
  ./gcloud/train_l4.sh
