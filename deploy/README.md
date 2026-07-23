# Deployment (Docker)

A CPU inference image that serves the **sequence-based** SPECTRA model
(ESM-2 + optional Rosetta features) via FastAPI. The structure/graph backends
(`dgl`, `torch_geometric`) are **not** installed — inference doesn't need them,
which keeps the image small.

## Design notes
- **Multi-stage build**: deps compile in a builder stage; the runtime stage
  copies only the venv + baked model cache. Runs as a non-root user.
- **ESM-2 weights are baked in** at build time (`ESM_MODEL`, default
  `facebook/esm2_t6_8M_UR50D`) and served with `TRANSFORMERS_OFFLINE=1` — no
  network at request time.
- **Trained checkpoint is provided at runtime** (volume), not baked, so the
  image stays generic and small.
- `/health` returns 200 even before a checkpoint is mounted (reports
  `no_model_loaded`), so the container is orchestrator-friendly.

## Build & run
```bash
# build
docker build -f deploy/Dockerfile -t spectra:latest .

# run (mount a trained checkpoint)
docker run -p 8000:8000 \
  -e SPECTRA_CHECKPOINT=/models/model.pt \
  -v $PWD/models:/models:ro \
  spectra:latest

curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d @deploy/sample_request.json
```

Or with compose: `docker compose -f deploy/docker-compose.yml up --build`.

## End-to-end smoke test (no trained model needed)
```bash
bash deploy/smoke_test.sh
```
Builds the image, mints an **untrained** checkpoint (via
`spectra.inference.dummy_checkpoint`), starts the API, and verifies
`/health` + `/predict`. This proves the deployment path works before any HPC
training run — predictions from the untrained model are meaningless by design.

## Configuration (env vars)
| Var | Default | Purpose |
|-----|---------|---------|
| `SPECTRA_CHECKPOINT` | — | path to trained weights (mount it) |
| `SPECTRA_MODE` | `E` | ablation mode the checkpoint was trained with |
| `SPECTRA_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SPECTRA_ESM_CKPT` | `facebook/esm2_t6_8M_UR50D` | must match the baked model |

## GPU serving
Swap the base image for `nvidia/cuda:12.2.*-runtime`, install a CUDA torch
wheel, run with `--gpus all` (needs the NVIDIA Container Toolkit), and set
`SPECTRA_DEVICE=cuda`.

## Production checkpoint delivery
For real deployments, don't bake large checkpoints into the image. Pull the
trained `model.pt` from an artifact store / release asset at container start
(init step or entrypoint), or mount it from a model volume / object store.


## Deploy to Google Cloud Run (scale-to-zero public endpoint)

The serving container listens on `$PORT` (Cloud Run injects it) and runs as a
non-root user, so it deploys to Cloud Run, Fly.io, or Render unchanged. Cloud
Run has no persistent volume, so the trained checkpoint is **baked in** via
`Dockerfile.cloudrun`.

**One-time setup**

```bash
gcloud auth login
export GCP_PROJECT=your-project GCP_REGION=us-central1
gcloud config set project "$GCP_PROJECT"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create spectra \
    --repository-format=docker --location="$GCP_REGION"
```

**Deploy (one command)**

```bash
GCP_PROJECT=your-project ./deploy/deploy_cloudrun.sh
```

It builds the base image, bakes `models/model.pt`, pushes to Artifact Registry,
deploys a scale-to-zero revision (`--min-instances 0`), and prints the live
HTTPS URL. Test it:

```bash
URL=$(gcloud run services describe spectra --region "$GCP_REGION" --format='value(status.url)')
curl -s "$URL/health"
curl -s -X POST "$URL/predict" -H 'content-type: application/json' -d @deploy/sample_request.json
```

**Continuous deployment.** `.github/workflows/deploy-cloudrun.yml` does the same
on every `v*` tag via Workload Identity Federation — set repo secrets
`GCP_WIF_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `GCP_PROJECT` (and optional var
`GCP_REGION`), then `git tag v0.1.0 && git push --tags`.

**Serving real predictions.** `models/model.pt` in the repo is the *untrained*
smoke-test checkpoint (meaningless scores). To serve real predictions, replace
it with the trained checkpoint from the cluster (the 15-fold CV winner) before
building — e.g. `scp seadragon:$SCRATCH/spectra/out/.../best.ckpt models/model.pt`.
Because it is a real weight file, track it with **git-lfs** rather than committing
35 MB into git history.
