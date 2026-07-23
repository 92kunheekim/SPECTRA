# Setup

```bash
conda env create -f environment.yml   # or: python -m venv .venv
conda activate spectra
pip install -e .
```

## Graph backends (CUDA-specific)
`torch-geometric` and `dgl` wheels depend on your torch/CUDA build. Install the
matching wheels from their official index, e.g.:
```bash
pip install torch-geometric
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
```

## ESM-2 checkpoint
By default SPECTRA pulls `facebook/esm2_t12_35M_UR50D` from HuggingFace.
Point to a local copy with `export SPECTRA_ESM_CKPT=/path/to/esm2`.
