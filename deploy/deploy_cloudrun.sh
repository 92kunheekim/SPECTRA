#!/usr/bin/env bash
# One-command deploy of SPECTRA to Google Cloud Run (scale-to-zero, public HTTPS).
#
# One-time GCP setup:
#   gcloud auth login && gcloud config set project "$GCP_PROJECT"
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
#   gcloud artifacts repositories create spectra \
#       --repository-format=docker --location="${GCP_REGION:-us-central1}"
#
# Then:  GCP_PROJECT=my-proj ./deploy/deploy_cloudrun.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
REPO="${AR_REPO:-spectra}"
SERVICE="${SERVICE:-spectra}"
TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/spectra:${TAG}"

echo "[1/4] build base serving image (linux/amd64 for Cloud Run)"
docker build --platform linux/amd64 -f deploy/Dockerfile -t spectra:latest .

echo "[2/4] bake trained checkpoint into a deployable image"
test -f models/model.pt || { echo "models/model.pt missing - put the trained checkpoint there first"; exit 1; }
docker build --platform linux/amd64 -f deploy/Dockerfile.cloudrun -t "$IMAGE" .

echo "[3/4] push to Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "$IMAGE"

echo "[4/4] deploy to Cloud Run (min-instances 0 => scale to zero)"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --project "$PROJECT" --region "$REGION" \
  --platform managed --allow-unauthenticated \
  --cpu 2 --memory 2Gi --min-instances 0 --max-instances 5 \
  --concurrency 8 --timeout 120 \
  --set-env-vars SPECTRA_MODE=E,SPECTRA_DEVICE=cpu

echo
echo "Live URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)'
