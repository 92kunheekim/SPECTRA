<div align="center">

# SPECTRA

**S**tructure, **P**rotein-language, and **E**nergetics via **C**ross-attention for **TR**-pMHC **A**ffinity

Multimodal deep learning for T-cell receptor–peptide-MHC binding prediction — fusing a protein language model (ESM-2, LoRA-adapted), 3D structure (equivariant / heterogeneous graph neural networks), and Rosetta interface energetics through learned gated cross-attention.

![python](https://img.shields.io/badge/python-3.10-3776ab)
![pytorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![license](https://img.shields.io/badge/license-Yale%20Non--Commercial-blue)
![status](https://img.shields.io/badge/status-research-orange)

</div>

## Overview

Predicting whether a TCR binds a given peptide-MHC is a central problem in immunology and immunotherapy design. Sequence-only models miss the geometry of the interface; structure-only models discard the rich priors in pretrained protein language models. **SPECTRA** unifies three complementary views of a TCR-pMHC complex:

- **Protein language (ESM-2).** Per-residue embeddings of the TCR (α/β) and peptide-MHC chains, with **chain-specific LoRA adapters** for parameter-efficient post-training of the language model.
- **Structure.** The 3D complex as a graph, encoded either with an **E(n)-equivariant GNN** or a **pseudo-heterogeneous graph transformer** that maintains separate message passing for each biological edge type (TCR↔peptide, TCR↔MHC, peptide↔MHC, …).
- **Energetics.** **Rosetta interface descriptors** (shape complementarity, ΔG separated, per-residue interface energy, buried SASA, Lennard-Jones / solvation / electrostatic terms).

These are combined by **bidirectional cross-attention** and a **per-dimension gated fusion** that learns how much to trust each modality per sample — and gracefully falls back when structure or energetics are missing.

> SPECTRA is a research derivative built on **ImmunoStruct** (Krishnaswamy Lab, Yale) and inspired by **STAG-LLM** (Kavraki Lab, Rice). See [`NOTICE`](NOTICE) and [`ARCHITECTURE.md`](ARCHITECTURE.md) for attribution and the design lineage.

## Key features

- ESM-2 sequence backbone with frozen / partial-unfreeze / full-finetune / **LoRA** modes
- Two interchangeable structure backbones (EGNN or hetero graph transformer)
- Optional injection of ESM residue embeddings as graph node features
- Two-stage **Rosetta multi-task pretraining** (ΔG regression + binding classification)
- **Ablation matrix (modes A–H)** isolating the contribution of each component
- **Multi-seed ensembling** with five combination strategies and variance-based uncertainty
- Portable, environment-driven paths — no hardcoded cluster locations

## Installation

```bash
git clone https://github.com/<your-username>/SPECTRA.git
cd SPECTRA
conda env create -f environment.yml && conda activate spectra
pip install -e .
```

Graph backends (`torch-geometric`, `dgl`) are CUDA-specific — see [`docs/setup.md`](docs/setup.md).

## Quickstart

```bash
# Point SPECTRA at your data (nothing is committed to the repo — see data/README.md)
export SPECTRA_DATA_DIR=/path/to/data
export SPECTRA_PDB_DIR=/path/to/pdbs

# Train the full model (mode E). Add --devices 4 --strategy ddp for multi-GPU.
python -m spectra.training.train --data_csv "$SPECTRA_DATA_DIR/training.csv"

# ...or load hyperparameters from a config (CLI flags still override):
python -m spectra.training.train --config configs/model/full_fusion.yaml \
    --data_csv "$SPECTRA_DATA_DIR/training.csv"

# Run the A–H architecture ablation (writes outputs/ablation_results.csv)
bash scripts/run_ablation.sh

# Train a 5-seed ensemble of the full model
bash scripts/run_ensemble.sh
```

## Repository structure

```
SPECTRA/
├── spectra/                 # the Python package
│   ├── config.py            # environment-driven paths (replaces cluster paths)
│   ├── data/                # dataset, leak-free splits, collate
│   ├── models/              # esm_encoder, lora, structure, cross_attention,
│   │                        #   rosetta, fusion, heads, spectra_model
│   ├── training/            # train, ablation, distributed (DDP), pretrain, ensemble
│   ├── inference/           # predict CLI, FastAPI api, serving, export
│   ├── evaluation/          # metrics (AUROC/AUPRC/AUC0.1/MCC/per-peptide), evaluate, aggregate
│   └── utils/               # seed, logging
├── configs/                 # YAML configs (default + per-model)
├── hpc/                     # LSF templates (single-GPU array + multi-GPU DDP) + generator
├── deploy/                  # Dockerfile (serving) + Dockerfile.train (GPU) + compose
├── scripts/                 # run_ablation.sh, run_ensemble.sh
├── tests/                   # import/config/model smoke tests
├── docs/                    # setup & training guides
├── data/                    # (no data committed) — how to obtain it
├── experiments/             # archived lineage (earlier variants, ensembling, HPC jobs)
├── ARCHITECTURE.md          # model design & evolution
├── NOTICE                   # attribution to upstream works
└── LICENSE                  # Yale Non-Commercial
```

## Method at a glance

| Modality | Encoder | Source |
|---|---|---|
| Sequence | ESM-2 + LoRA adapters | `spectra/models/esm_encoder.py`, `lora.py` |
| Structure | EGNN / hetero graph transformer | `spectra/models/structure.py` |
| Energetics | Residual MLP over 12 Rosetta features | `spectra/models/rosetta.py` |
| Fusion | Bidirectional cross-attention + gated fusion | `spectra/models/cross_attention.py`, `fusion.py` |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and how the architecture evolved.

## Training at scale (HPC / multi-GPU)

SPECTRA runs on PyTorch Lightning and scales by configuration:

- **Single-GPU job arrays** for the A–H ablation sweep — one mode per array task
  (`hpc/train_single_gpu_array.lsf`, `hpc/generate_ablation_jobs.sh`) — the
  data-parallel-across-jobs pattern used on the LSF cluster.
- **Multi-GPU DDP** for a single large model (`hpc/train_multi_gpu_ddp.lsf`):
  `--devices 4 --strategy ddp --sync-batchnorm --precision bf16-mixed`, wired
  through `spectra/training/distributed.py` (`--num-nodes > 1` for multi-node).

See [`docs/hpc.md`](docs/hpc.md).

## Deployment

The sequence-based model serves without the structure pipeline:

- **Batch CLI:** `python -m spectra.inference.predict --checkpoint model.pt --input_csv pairs.csv`
- **REST API (FastAPI):** `uvicorn spectra.inference.api:app` → `POST /predict`
- **Docker:** multi-stage CPU image, ESM-2 baked in, non-root, healthcheck — `docker build -f deploy/Dockerfile -t spectra .`
- **Smoke test (no trained model needed):** `bash deploy/smoke_test.sh`
- **Serverless (live):** deployed on **Modal** as a public scale-to-zero endpoint (`modal deploy deploy/modal_app.py`); the same container recipe targets **Google Cloud Run** (`deploy/deploy_cloudrun.sh`).
- **Observability:** `GET /stats` (JSON latency percentiles) and `GET /metrics` (Prometheus text).

See [`deploy/README.md`](deploy/README.md) and [`docs/deployment.md`](docs/deployment.md).

## Performance (deployed inference)

Measured against the live serverless endpoint (Modal, CPU, mode-E, ESM-2 `t6`)
with the standard-library load-tester (`deploy/loadtest/load_test.py`):

| Metric | Value |
|---|---|
| Warm latency **p50** | **524 ms** end-to-end (~240 ms server-side compute) |
| Warm latency **p95** | 605 ms |
| Warm latency **p99** | 659 ms |
| Throughput | **8 req/s** @ concurrency 8 (120 requests, **0 errors**) |
| Cold start (scale-from-zero) | ~9 s |

Warm figures are 30 sequential requests against a hot container. About half of the
524 ms p50 is client↔cloud network round-trip and ~240 ms is model compute
(confirmed by the server-side `/stats` view). The ~9 s cold start is the
scale-to-zero trade-off — the container loads torch + ESM on the first request
after idle, in exchange for **zero cost while idle**. Under burst concurrency Modal
autoscales more containers, so tail latency during a cold fan-out includes their
one-time cold starts.

```bash
# reproduce
python deploy/loadtest/load_test.py https://<workspace>--spectra-tcr-pmhc-fastapi-app.modal.run
curl https://<workspace>--spectra-tcr-pmhc-fastapi-app.modal.run/stats
```

`/stats` is per-container (in-process), so under autoscaling it reflects a single
replica; aggregate across replicas by scraping `/metrics` (Prometheus) centrally.

## Evaluation

Training uses the **leak-free 15-fold split** (`--split_json splits.json --fold k`)
and reports on a **held-out test split** — never on val. Each run saves
`test_predictions.csv`; metrics (`spectra/evaluation/metrics.py`) cover the
imbalance-aware set — **AUROC, AUPRC, AUC0.1** (partial AUC), **MCC, F1**, and
**per-peptide macro-AUROC**. Aggregate across folds into a table + figure:

```bash
python -m spectra.evaluation.aggregate --results_dir outputs/phase2_cv --group_by fold
```

See [`hpc/README.md`](hpc/README.md) for the seadragon (Apptainer) train→evaluate flow.

## Results

All numbers below are on the **held-out test split** of the leak-free 15-fold split — never validation. Mode **E** is the full tri-modal model (4-chain per-chain pooling + bidirectional cross-attention + Rosetta energetics).

### Cross-validated performance (mode E, 15 folds)

Mean ± standard deviation across the 15 CV folds:

| Metric | Score |
|---|---|
| AUROC | **0.825 ± 0.013** |
| AUPRC | 0.644 ± 0.024 |
| AUC0.1 (partial AUC, FPR ≤ 0.1) | 0.731 ± 0.013 |
| MCC | 0.395 ± 0.035 |
| F1 | 0.499 ± 0.028 |
| Per-peptide macro-AUROC | 0.699 ± 0.027 |

AUPRC and the per-peptide macro-AUROC are the honest metrics here: the data are class-imbalanced, and the per-peptide macro average down-weights well-represented epitopes so a few dominant peptides can't inflate the headline. The tight fold-to-fold spread (AUROC σ ≈ 0.013) indicates the split is stable and the result is not driven by a lucky fold.

### Component ablation (modes A–H, fold 0)

Each mode toggles one design axis, so the deltas isolate what each component is worth. Held-out test, fold 0:

| Mode | Chains | Cross-attn | Rosetta | AUROC | AUPRC |
|---|---|:---:|:---:|---|---|
| A | 1 | – | – | 0.734 | 0.498 |
| F | 1 | – | ✓ | 0.740 | 0.507 |
| G | 2 | – | – | 0.779 | 0.560 |
| H | 2 | – | ✓ | 0.784 | 0.580 |
| B | 4 | – | – | 0.818 | 0.635 |
| C | 4 | – | ✓ | 0.812 | 0.629 |
| D | 4 | ✓ | – | 0.820 | 0.630 |
| **E** | **4** | **✓** | **✓** | **0.830** | **0.660** |

**What the ablation shows:**

- **Chain granularity is the dominant lever.** Going from a single concatenated chain (A/F ≈ 0.73–0.74 AUROC) to a 2-chain pMHC/TCR split (G/H ≈ 0.78) to full 4-chain per-chain pooling (B/C/D ≈ 0.82) is the largest source of gain — roughly +0.085 AUROC end to end. Keeping the chains separate before fusion preserves interface structure the model would otherwise average away.
- **Cross-attention and Rosetta energetics are individually modest but synergistic.** At 4 chains, adding only cross-attention (B→D) or only Rosetta (B→C) barely moves AUROC, but combining them (D→E: 0.820→0.830, and on AUPRC 0.630→0.660) gives the top single-mode result. The energetics signal is most useful once cross-attention has aligned the interface representations it can weight.
- **Rosetta features help more at coarse granularity.** The A→F and G→H gains show energetics compensating when chain structure is collapsed; by 4 chains most of that information is already captured by the learned representation, so Rosetta's marginal value shifts to the cross-attention setting.

![SPECTRA ablation and CV results](docs/images/results.png)

*Reproduce:* `python -m spectra.evaluation.aggregate --results_dir outputs/phase2_cv --group_by fold` (see [`hpc/README.md`](hpc/README.md) for the full seadragon train→evaluate flow).

## Status

Active research code. The two model tracks — the **flagship EGNN multimodal model** (`spectra/models/spectra_model.py`) and the **ESM + Rosetta ablation model** (`spectra/models/ablation_model.py`, modes A–H) — are migrated and import-clean, with all cluster paths parameterized through `spectra.config`. The exploration lineage (earlier architecture variants, ensembling, and HPC job scripts) is archived under `experiments/`. Not for clinical or commercial use.

## Citation

If you use SPECTRA, please cite this repository (see [`CITATION.cff`](CITATION.cff)) and the upstream works in [`NOTICE`](NOTICE).

## License

Distributed under the **Yale Non-Commercial License** (inherited from ImmunoStruct). Commercial use requires a separate license from Yale Ventures. See [`LICENSE`](LICENSE).
