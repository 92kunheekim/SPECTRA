"""Compute the metric suite from a predictions CSV (columns: peptide,label,prob).

    python -m spectra.evaluation.evaluate --pred_csv outputs/.../test_predictions.csv
"""
from __future__ import annotations
import argparse
import json
import pandas as pd
from spectra.evaluation.metrics import evaluate as _evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()
    df = pd.read_csv(args.pred_csv)
    peps = df["peptide"].values if "peptide" in df.columns else None
    m = _evaluate(df["label"].values, df["prob"].values, peps)
    print(json.dumps(m, indent=2))
    if args.out:
        json.dump(m, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
