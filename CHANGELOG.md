# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Added
- Standalone repository scaffold: package layout, configs, CI, tests, docs.
- `spectra.config` for environment-driven, portable paths.
- Flagship EGNN multimodal model migrated to `spectra/models/spectra_model.py`.
- ESM + Rosetta ablation model (modes A-H) + dataset + runner migrated to
  `spectra/models/ablation_model.py`, `spectra/data/dataset_ablation.py`,
  `spectra/training/ablation.py`.
- Granular `spectra/models/*` modules re-export the concrete classes.
- All hardcoded `/rsrch3/...` cluster paths parameterized; experiments/ sanitized.
- Architecture lineage archived under `experiments/`.
- Multi-GPU / multi-node **DDP** support (`spectra/training/distributed.py`).
- **HPC/LSF** templates + job generator under `hpc/` (single-GPU array + multi-GPU DDP).
- **Deployment**: batch CLI, FastAPI service, serving bundle export (`spectra/inference/`),
  Dockerfile + docker-compose under `deploy/`.
- Production-grade serving image: multi-stage, non-root, pinned deps, ESM-2 baked
  in for offline inference, healthcheck; `deploy/smoke_test.sh` end-to-end test;
  `spectra.inference.dummy_checkpoint` for pre-training deployment validation;
  Docker build CI workflow.
- `docs/hpc.md`, `docs/deployment.md`; `serve`/`graph`/`dev` optional-dependency extras.

### Added (train/eval on HPC)
- Leak-free split loader: `--split_json/--fold` with a true **held-out test**
  (`load_datasets_from_split`); runs save `test_predictions.csv`.
- `spectra/evaluation/metrics.py` (AUROC/AUPRC/AUC0.1/MCC/F1/per-peptide),
  `evaluate` CLI, and `aggregate` (mean±std table + figure).
- `deploy/Dockerfile.train` (CUDA 12.1 GPU image, ESM baked, offline) for
  Docker→Apptainer on seadragon.
- `hpc/phase0_smoke` / `phase1_ablation_fold0` / `phase2_cv_winner` /
  `phase3_ensemble` Apptainer LSF scripts + `hpc/README.md`.
- Training image is framework-only; ESM-2 weights bind-mounted from scratch
  (`--esm_checkpoint`) instead of baked. Documented a build-free NGC option.
- CI builds+pushes serving and training images to GHCR; API uses a lifespan handler.

### Fixed
- Rosetta feature loading now accepts bare CSV column names + the `dG_separated/dSASAx100` alias (previously required `feat_*`, so features loaded as zero).
- Quickstart now runs: real `spectra.training.train` entrypoint (full model by
  default), `scripts/run_ablation.sh` / `run_ensemble.sh` match the actual CLI.
- Multi-GPU wired: `make_trainer` drives the ablation runner; `--devices /
  --strategy / --precision / --sync-batchnorm` are live. HPC array template runs
  one ablation mode per task.
- `--config` YAML loading implemented; configs flattened to match CLI flags.

### To do
- Port two-stage Rosetta pretraining onto the flagship model (currently in
  experiments/, tied to the `model_llm` generation).
- Rewire `ensemble_tcr.py` onto the packaged ablation modules.
- Add an end-to-end training smoke test and a tiny example dataset.
