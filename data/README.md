# Data

**No datasets are committed to this repository.** Training data (peptide-MHC /
TCR tables, PDB structures, Rosetta features) must be obtained separately and
placed here (or pointed to via environment variables — see `spectra/config.py`).

## Expected layout
```
data/
├── training.csv              # columns: peptide, CDR3a, CDR3b, MHC, binder, ...
├── splits_leakfree.json      # {"folds": [...]} peptide/TCR-aware splits
├── pdbs/                     # <structure_id>.pdb (TCR-pMHC complexes)
└── rosetta/                  # per-complex Rosetta interface features (CSV)
```

## Sources
- Public TCR-pMHC pairs: VDJdb, IEDB, McPAS-TCR, 10x Genomics, NetTCR datasets.
- Structures: model with TCRmodel2 (or use experimental PDB complexes).
- Respect each source's license; do not redistribute third-party or clinical
  data through this repository.

Set paths without moving files:
```bash
export SPECTRA_DATA_DIR=/path/to/data
export SPECTRA_PDB_DIR=/path/to/pdbs
```
