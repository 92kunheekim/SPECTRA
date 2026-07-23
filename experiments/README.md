# experiments/ — architecture lineage (archived)

This folder preserves the exploration that led to the final SPECTRA model. It is
**not** part of the installable package and is not maintained for direct
execution — it documents *how the design was reached*. Absolute cluster paths
have been sanitized to `${SPECTRA_ROOT}` placeholders.

## `model_generation/` — the model evolution
The architecture converged through a deliberate sequence (see `../ARCHITECTURE.md`):

| Stage | File(s) | What it introduced |
|-------|---------|--------------------|
| v1 | `model.py`, `dataset.py` | GRU-VAE sequence + EGNN structure + scalar gated fusion |
| v2 | `model2.py`, `dataset2.py` | Pre-LN Transformer sequence encoder; residual/BN head |
| Model B | `model_llm.py` | pivot to **ESM-2** backbone + **LoRA** |
| Model B2 | `model_llm2.py` | STAG-LLM-style **pseudo-heterogeneous graph transformer** |
| Model B3 | `model_llm3.py` | **ESM embeddings injected as graph node features** |
| controls | `model_seq.py`, `model_struct.py` | single-modality ablations |
| pretraining | `pretrain_rosetta.py` | two-stage Rosetta ΔG + binding multi-task |
| training | `train*.py` | training loops for the above generation |

The **flagship** consolidation of this line lives in the package at
`spectra/models/spectra_model.py`.

## `ensemble_tcr.py`, `compare_ablation.py`
Multi-seed ensembling (5 combination strategies + variance uncertainty) and
ablation comparison. `ensemble_tcr.py` imports an unmerged `model_improved` /
`run_improved` refactor; rewiring it onto `spectra.models.ablation_model` +
`spectra.training.ablation` is tracked in `../CHANGELOG.md`.

## `cluster_jobs/`
LSF (`.lsf`) and job-generation (`.sh`) scripts used to run training/ablation on
an HPC cluster — kept as a record of the compute setup.
