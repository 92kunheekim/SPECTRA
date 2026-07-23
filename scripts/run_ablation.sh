#!/usr/bin/env bash
# Run the full A–H architecture ablation (one process; loops modes internally,
# writes outputs/ablation_results.csv). Set DEVICES=4 for multi-GPU DDP.
set -euo pipefail
DATA="${SPECTRA_DATA_CSV:-${SPECTRA_DATA_DIR:-./data}/training.csv}"
python -m spectra.training.ablation \
  --modes A B C D E F G H \
  --data_csv "$DATA" \
  --devices "${DEVICES:-auto}"
