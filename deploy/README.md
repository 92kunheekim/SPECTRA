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
