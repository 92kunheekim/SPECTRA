"""Aggregate per-run held-out-test predictions into a mean±std table + figure.

Walks a results directory for `test_predictions.csv` files (one per fold/mode),
recomputes the full metric suite from predictions (single source of truth),
groups by mode or fold, and writes a Markdown table, a per-run CSV, and a PNG.

    python -m spectra.evaluation.aggregate --results_dir outputs --group_by fold
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

from spectra.evaluation.metrics import evaluate

METRICS = ["auroc", "auprc", "auc01", "mcc", "f1", "per_peptide_auroc_macro"]


def _collect(results_dir):
    rows = []
    for pred in glob.glob(os.path.join(results_dir, "**", "test_predictions.csv"), recursive=True):
        d = pd.read_csv(pred)
        peps = d["peptide"].values if "peptide" in d.columns else None
        m = evaluate(d["label"].values, d["prob"].values, peps)
        rj = os.path.join(os.path.dirname(pred), "result.json")
        if os.path.exists(rj):
            r = json.load(open(rj))
            m["mode"] = r.get("mode"); m["fold"] = r.get("fold")
        m["path"] = pred
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_prefix", default="docs/results")
    ap.add_argument("--group_by", default="mode", choices=["mode", "fold", "none"])
    args = ap.parse_args()

    df = _collect(args.results_dir)
    if df.empty:
        print(f"No test_predictions.csv found under {args.results_dir}")
        return
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    df.to_csv(args.out_prefix + "_per_run.csv", index=False)

    present = [m for m in METRICS if m in df.columns]
    gb = None if args.group_by == "none" or args.group_by not in df.columns else args.group_by

    lines = ["# Results (held-out test)\n"]
    if gb:
        agg = df.groupby(gb)[present].agg(["mean", "std"])
        lines.append("| " + gb + " | " + " | ".join(present) + " |")
        lines.append("|" + "---|" * (len(present) + 1))
        for key, row in agg.iterrows():
            cells = [f"{row[(m, 'mean')]:.3f} ± {0.0 if np.isnan(row[(m,'std')]) else row[(m,'std')]:.3f}" for m in present]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    else:
        agg = df[present].agg(["mean", "std"]).T
        lines.append("| metric | mean ± std |")
        lines.append("|---|---|")
        for m in present:
            sd = 0.0 if np.isnan(agg.loc[m, "std"]) else agg.loc[m, "std"]
            lines.append(f"| {m} | {agg.loc[m, 'mean']:.3f} ± {sd:.3f} |")
    md = "\n".join(lines) + "\n"
    open(args.out_prefix + ".md", "w").write(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        if gb:
            keys = list(df.groupby(gb).groups.keys())
            agg = df.groupby(gb)[present].agg(["mean", "std"])
            x = np.arange(len(keys)); w = 0.8 / max(1, len(present))
            for i, m in enumerate(present):
                means = [agg.loc[k, (m, "mean")] for k in keys]
                errs = [0.0 if np.isnan(agg.loc[k, (m, "std")]) else agg.loc[k, (m, "std")] for k in keys]
                ax.bar(x + i * w, means, w, yerr=errs, capsize=3, label=m)
            ax.set_xticks(x + 0.4); ax.set_xticklabels([str(k) for k in keys]); ax.set_xlabel(gb)
        else:
            ax.bar(present, [df[m].mean() for m in present],
                   yerr=[df[m].std(ddof=0) for m in present], capsize=3)
        ax.set_ylim(0, 1); ax.set_ylabel("score")
        ax.set_title("SPECTRA — held-out test metrics"); ax.legend(fontsize=8, ncol=3)
        fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=150)
        print("wrote", args.out_prefix + ".png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
