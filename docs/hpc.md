# Training at scale — HPC (LSF) & multi-GPU

SPECTRA training runs on PyTorch Lightning, so the same code scales from a
laptop to a multi-GPU node with config only.

## Two scale-out patterns

**1. Single-GPU job arrays (data-parallel across folds/modes).**
The K-fold cross-validation and the A–H ablation sweep are embarrassingly
parallel — one GPU per fold/mode. This is the cheapest way to use a shared
cluster (many small jobs schedule faster than one big reservation).

```bash
bsub < hpc/train_single_gpu_array.lsf     # #BSUB -J spectra_cv[1-15]
bash hpc/generate_ablation_jobs.sh        # emits ablation_{A..H}.lsf
for f in ablation_*.lsf; do bsub < "$f"; done
```

**2. Multi-GPU DDP (data-parallel within one job).**
For a single large model, distribute each batch across GPUs with DDP:

```bash
bsub < hpc/train_multi_gpu_ddp.lsf        # requests num=4 GPUs on one host
# equivalently, interactively:
python -m spectra.training.ablation --modes E \
    --devices 4 --strategy ddp --sync-batchnorm --precision bf16-mixed
```

`spectra/training/distributed.py` (`make_trainer`) turns `--devices / --strategy /
--num-nodes / --precision / --sync-batchnorm / --accumulate-grad-batches` into a
configured Lightning `Trainer`. `--num-nodes > 1` enables multi-node DDP.

## LSF notes
- GPU request: `#BSUB -gpu "num=N:gmem=32:mode=exclusive_process"`.
- Keep all DDP GPUs on one host with `#BSUB -R span[hosts=1]`.
- Environment/modules: `module load cuda12.2/... gcc/...`, `conda activate spectra`.
- Paths come from `SPECTRA_DATA_DIR` / `SPECTRA_OUT_DIR` (no hardcoded paths).
