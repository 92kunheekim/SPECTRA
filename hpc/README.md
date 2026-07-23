# HPC (seadragon) — containerized train / evaluate

Seadragon does not run Docker (no root on shared nodes); it uses **Apptainer**,
and only allows `apptainer pull` (not build). So we **build with Docker, run
with Apptainer**. The training image is **framework-only** — ESM-2 weights and
data are bind-mounted from scratch, so the image is small and builds with no
network.

## 1. Build & publish the training image (laptop or CI)
```bash
docker build -f deploy/Dockerfile.train -t ghcr.io/<user>/spectra-train:latest .
docker push ghcr.io/<user>/spectra-train:latest
```
(CI publishes this automatically on push to `main` / tags — see `.github/workflows/docker.yml`.)

## 2. Pull to a .sif on a seadragon LOGIN node (has internet)
```bash
apptainer pull spectra-train.sif docker://ghcr.io/<user>/spectra-train:latest
export SPECTRA_SIF=$PWD/spectra-train.sif
```

## 3. Point at data + ESM on scratch
The phase scripts default to `$SCRATCH = /rsrch3/scratch/genomic_med/$USER` and expect:
```
$SCRATCH/spectra/training_rosetta.csv
$SCRATCH/spectra/training_data_split_leakfree.json
$SCRATCH/huggingface_model/facebook/esm2_t6_8M_UR50D/   # the ESM-2 snapshot you already have
```
Override any of these via `SPECTRA_SCRATCH`, `SPECTRA_DATA_DIR`, or
`SPECTRA_ESM_CKPT` if your layout differs. ESM is passed to the model with
`--esm_checkpoint "$ESM_DIR"` (a local dir — no HF download).

## 4. Submit the phased jobs
| Phase | Script | What | Queue |
|-------|--------|------|-------|
| 0 | `phase0_smoke.lsf` | GPU visible + 1 step runs | `gpu-test` |
| 1 | `phase1_ablation_fold0.lsf` | A–H on fold 0 (pick winner) | `egpu` array[1-8] |
| 2 | `phase2_cv_winner.lsf` | winner × 15 folds, held-out test | `egpu` array[1-15] |
| 3 | `phase3_ensemble.lsf` | 5-seed ensemble of winner | `egpu` array[1-5] |

```bash
bsub < hpc/phase0_smoke.lsf
bsub < hpc/phase1_ablation_fold0.lsf
SPECTRA_WINNER=E bsub < hpc/phase2_cv_winner.lsf
SPECTRA_WINNER=E bsub < hpc/phase3_ensemble.lsf
```
`--nv` passes the host GPU into the container; `--bind $SCRATCH:$SCRATCH` mounts
data + ESM. Monitor with `bjobs -l`, `bhist -l`, `nvidia-smi`.

## 5. Aggregate results (login node)
```bash
apptainer exec "$SPECTRA_SIF" python -m spectra.evaluation.aggregate \
    --results_dir "$SCRATCH/spectra/out/phase2_cv" --group_by fold --out_prefix docs/results
apptainer exec "$SPECTRA_SIF" python -m spectra.evaluation.aggregate \
    --results_dir "$SCRATCH/spectra/out/phase1_ablation" --group_by mode --out_prefix docs/ablation
```
Writes `docs/results.md` + `docs/results.png` (mean±std AUROC/AUPRC/AUC0.1/MCC/
per-peptide) — drop straight into the README.

## Alternative: no image build (stock NGC PyTorch)
Skip building/publishing entirely — pull a stock PyTorch container and run the
repo from a bind mount:
```bash
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:24.05-py3
# one-time: install the extra deps into a persistent scratch prefix
apptainer exec --nv pytorch.sif pip install --target "$SCRATCH/pylibs" \
    transformers pytorch-lightning torchmetrics scikit-learn pandas pyyaml matplotlib
# run with the repo + deps on PYTHONPATH
apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind /path/to/SPECTRA:/opt/spectra \
    --env PYTHONPATH=/opt/spectra:$SCRATCH/pylibs pytorch.sif \
    python -m spectra.training.ablation --modes E \
      --data_csv "$SCRATCH/spectra/training_rosetta.csv" \
      --split_json "$SCRATCH/spectra/training_data_split_leakfree.json" --fold 0 \
      --esm_checkpoint "$SCRATCH/huggingface_model/facebook/esm2_t6_8M_UR50D" \
      --devices 1 --out_dir "$SCRATCH/spectra/out/ablation"
```
Trade-off: no build/registry step, but you install deps once into a scratch
prefix (`--target`) and carry it on `PYTHONPATH` instead of baking an image.

> `conda activate spectra` + the `train_*.lsf` templates remain a non-container
> fallback. The Apptainer path above is the reproducible one.
