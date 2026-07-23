# Deployment

SPECTRA serves predictions from the **sequence-based** model (ESM-2 + optional
Rosetta features) — no PDB/graph pipeline at inference, so it is light to deploy.

## 1. Batch CLI
```bash
python -m spectra.inference.predict \
    --checkpoint model.pt --mode E \
    --input_csv pairs.csv --out_csv scored.csv
```
`pairs.csv` columns: `tra_seq, trb_seq, peptide, mhc_seq` (+ optional `feat_*`).

## 2. REST API (FastAPI)
```bash
SPECTRA_CHECKPOINT=model.pt uvicorn spectra.inference.api:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
    -d '{"tra_seq":"CA...","trb_seq":"CAS...","peptide":"GILGFVFTL","mhc_seq":"GSHSMRY..."}'
# -> {"probability": 0.83}
```
Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`.

## 3. Docker
```bash
docker build -f deploy/Dockerfile -t spectra:latest .
docker run -p 8000:8000 -e SPECTRA_CHECKPOINT=/models/model.pt \
    -v $PWD/models:/models:ro spectra:latest
# or: docker compose -f deploy/docker-compose.yml up --build
```
The image installs CPU torch; swap the base image for a CUDA one for GPU serving.

## 4. Export a serving bundle
```bash
python -m spectra.inference.export --checkpoint ckpt.pt --mode E --out serving_bundle.pt
```
Bundles weights + config for the serving loader (portable across environments).
