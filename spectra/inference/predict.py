"""Batch inference CLI.

Score TCR-pMHC pairs with a trained SPECTRA (ablation) model. Uses only
sequences (+ optional Rosetta features) — no PDB/graph pipeline — so it is
lightweight to deploy.

Usage:
    python -m spectra.inference.predict \
        --checkpoint model.pt --mode E \
        --input_csv pairs.csv --out_csv scored.csv

Input CSV columns: tcr_id, tra_seq, trb_seq, peptide, mhc_seq
(optional feat_* Rosetta columns for rosetta-enabled modes).
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import torch

from spectra.inference.serving import load_model, score_dataframe


def main():
    ap = argparse.ArgumentParser(description="SPECTRA batch inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="E", help="Ablation mode the checkpoint was trained with")
    ap.add_argument("--esm_checkpoint", default=os.environ.get("SPECTRA_ESM_CKPT", "facebook/esm2_t12_35M_UR50D"))
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    model, tokenizer = load_model(args.checkpoint, mode=args.mode,
                                  esm_checkpoint=args.esm_checkpoint, device=args.device)
    df = pd.read_csv(args.input_csv)
    df["probability"] = score_dataframe(model, tokenizer, df, mode=args.mode,
                                        device=args.device, batch_size=args.batch_size)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} predictions -> {args.out_csv}")


if __name__ == "__main__":
    main()
