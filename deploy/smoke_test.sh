#!/usr/bin/env bash
# Build the image, mint an UNTRAINED checkpoint, start the API, and verify
# /health and /predict end to end. Requires docker + curl. No GPU, no network
# at request time (ESM weights are baked in).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
IMG="spectra:smoke"; PORT="${PORT:-8000}"; CID=""
cleanup(){ [ -n "$CID" ] && docker rm -f "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[1/5] build image"
docker build -f deploy/Dockerfile -t "$IMG" .

echo "[2/5] mint UNTRAINED checkpoint (plumbing only)"
mkdir -p models
docker run --rm -v "$PWD/models:/models" "$IMG" \
  python -m spectra.inference.dummy_checkpoint --mode E --out /models/model.pt

echo "[3/5] start server"
CID=$(docker run -d -p "${PORT}:8000" \
  -e SPECTRA_CHECKPOINT=/models/model.pt \
  -v "$PWD/models:/models:ro" "$IMG")

echo "[4/5] wait for /health"
for i in $(seq 1 60); do
  if curl -sf "localhost:${PORT}/health" >/dev/null; then break; fi
  sleep 2
  [ "$i" = 60 ] && { echo "health check timed out"; docker logs "$CID"; exit 1; }
done
curl -s "localhost:${PORT}/health"; echo

echo "[5/5] POST /predict with a real binder payload"
RESP=$(curl -s -X POST "localhost:${PORT}/predict" \
  -H 'content-type: application/json' -d @deploy/sample_request.json)
echo "response: $RESP"
echo "$RESP" | grep -q '"probability"' \
  && echo "✅ SMOKE TEST PASSED (deployment path works end to end)" \
  || { echo "❌ SMOKE TEST FAILED"; exit 1; }
