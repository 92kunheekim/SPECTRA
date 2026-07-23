"""Centralized, environment-driven paths and hyperparameters.

Replaces hardcoded absolute cluster paths (e.g. /rsrch3/...) with values read
from environment variables or a YAML config, so the code is portable.

Environment variables (all optional; sensible defaults for a local run):
  SPECTRA_DATA_DIR   root for CSVs / labels          (default: ./data)
  SPECTRA_PDB_DIR    directory of TCR-pMHC PDBs       (default: ./data/pdbs)
  SPECTRA_CACHE_DIR  graph/feature cache              (default: ./.cache)
  SPECTRA_OUT_DIR    training outputs / checkpoints   (default: ./outputs)
  SPECTRA_ESM_CKPT   HF id or local path to ESM-2     (default: facebook/esm2_t12_35M_UR50D)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


@dataclass
class Paths:
    data_dir: Path = _p("SPECTRA_DATA_DIR", "./data")
    pdb_dir: Path = _p("SPECTRA_PDB_DIR", "./data/pdbs")
    cache_dir: Path = _p("SPECTRA_CACHE_DIR", "./.cache")
    out_dir: Path = _p("SPECTRA_OUT_DIR", "./outputs")
    esm_ckpt: str = os.environ.get("SPECTRA_ESM_CKPT", "facebook/esm2_t12_35M_UR50D")

    def ensure(self) -> "Paths":
        for d in (self.data_dir, self.pdb_dir, self.cache_dir, self.out_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()
