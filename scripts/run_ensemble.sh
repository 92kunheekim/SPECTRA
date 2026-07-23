#!/usr/bin/env bash
# Train a 5-seed ensemble of the full model (mode E). Combine the members'
# predictions with the strategies documented in experiments/ensemble_tcr.py.
set -euo pipefail
DATA="${SPECTRA_DATA_CSV:-${SPECTRA_DATA_DIR:-./data}/training.csv}"
OUT="${SPECTRA_OUT_DIR:-./outputs}/ensemble"
for s in 42 123 456 789 2024; do
  echo "== ensemble member: seed=$s =="
  python -m spectra.training.train \
    --modes E --seed "$s" \
    --data_csv "$DATA" \
    --out_dir "$OUT/seed_$s" \
    --devices "${DEVICES:-auto}"
done
