# Restore a Cloud Shell session's state after a reconnect -- SOURCE this, don't
# execute it (`source gcloud/bootstrap_session.sh` or `. gcloud/bootstrap_session.sh`),
# since it needs to export variables into your *current* shell, not a subprocess.
#
# Cloud Shell keeps files across reconnects but throws away shell state (env vars,
# cwd, active gcloud project) every time -- this replaces the multi-line block you'd
# otherwise have to retype from memory each time (see OPS.md #9 / the .env pattern
# CLAUDE.md #7 already specifies for secrets).
#
# One-time setup before this works: cp .env.example .env, then fill in GCP_PROJECT,
# GCS_BUCKET, HF_TOKEN (and HF_HUB_ORG if you're not using "smartlawai"). .env is
# gitignored (I7) -- it stays on this Cloud Shell disk only, never pushed anywhere.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This must be SOURCED, not executed -- run:" >&2
  echo "  source gcloud/bootstrap_session.sh" >&2
  echo "Running it directly (./gcloud/bootstrap_session.sh) would export variables" >&2
  echo "into a subshell that exits immediately, silently doing nothing." >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "No .env found in $(pwd)." >&2
  echo "Run: cp .env.example .env   then fill in GCP_PROJECT, GCS_BUCKET, HF_TOKEN." >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${GCP_PROJECT:?GCP_PROJECT missing from .env}"
: "${GCS_BUCKET:?GCS_BUCKET missing from .env}"
: "${HF_TOKEN:?HF_TOKEN missing from .env}"
export REGION="${REGION:-us-central1}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

echo "Session restored:"
echo "  project = ${GCP_PROJECT}"
echo "  region  = ${REGION}"
echo "  bucket  = gs://${GCS_BUCKET}"
echo "  HF_TOKEN set (${#HF_TOKEN} chars)"
