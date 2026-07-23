#!/bin/sh
# SPECTRA API entrypoint. Honors $PORT (Cloud Run / Fly / Render inject it);
# defaults to 8000 for local docker-compose and smoke tests.
set -e
exec uvicorn spectra.inference.api:app --host 0.0.0.0 --port "${PORT:-8000}"
