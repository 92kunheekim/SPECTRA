"""
compare_ablation.py — Post-hoc Ablation Results Comparison
============================================================

Run this AFTER all 8 ablation LSF jobs have completed.
Collects result.json from each mode_*/result.json, produces:
  1. Terminal comparison table
  2. ablation_results.csv (combined metrics)
  3. ablation_comparison.csv (pairwise deltas)
  4. ablation_report.txt (full text report)

Usage:
  python compare_ablation.py --results_dir /path/to/ablation/
  python compare_ablation.py --results_dir /path/to/ablation/ --output_csv results.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# Mode metadata (must match model_ablation.py)
# ============================================================

ABLATION_MODES = {
    "A": "concat_cls",
    "B": "4chain_pool",
    "C": "4chain_pool_rosetta",
    "D": "4chain_crossattn",
    "E": "4chain_crossattn_rosetta",
    "F": "concat_cls_rosetta",
    "G": "2chain_pool",
    "H": "2chain_pool_rosetta",
}

PAIRWISE_COMPARISONS = [
    # (baseline, test, description, what_it_tests)
    ("A", "G", "1-chain → 2-chain split",               "positional_encoding"),
    ("A", "B", "1-chain → 4-chain split",               "positional_encoding"),
    ("G", "B", "2-chain → 4-chain split",               "positional_encoding"),
    ("B", "D", "4-chain pool → 4-chain cross-attn",     "cross_attention"),
    ("A", "F", "1-chain → 1-chain + Rosetta",           "rosetta_value"),
    ("G", "H", "2-chain → 2-chain + Rosetta",           "rosetta_value"),
    ("B", "C", "4-chain pool → 4-chain pool + Rosetta", "rosetta_value"),
    ("D", "E", "4-chain xattn → 4-chain xattn + Rosetta", "rosetta_value"),
    ("F", "E", "Simple+Rosetta vs Full model",          "architecture_value"),
    ("H", "E", "2-chain+Rosetta vs Full model",         "architecture_value"),
    ("A", "E", "Baseline vs Full model",                "total_improvement"),
]


def load_results(results_dir: Path) -> dict:
    """Load result.json from each mode_X/ subdirectory."""
    results = {}
    for mode in ABLATION_MODES:
        json_path = results_dir / f"mode_{mode}" / "result.json"
        if json_path.exists():
            with open(json_path) as f:
                results[mode] = json.load(f)
    return results


def load_training_curves(results_dir: Path, mode: str) -> pd.DataFrame:
    """Load the Lightning CSV logger metrics for a given mode."""
    log_dir = results_dir / f"mode_{mode}" / "logs"
    if not log_dir.exists():
        return pd.DataFrame()
    # Find the version directory
    versions = sorted(log_dir.glob("version_*"))
    if not versions:
        return pd.DataFrame()
    metrics_file = versions[-1] / "metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return pd.DataFrame()


def build_results_table(results: dict) -> pd.DataFrame:
    """Build a sorted DataFrame from collected results."""
    rows = []
    for mode, r in sorted(results.items()):
        rows.append({
            "mode": mode,
            "config": r.get("mode_name", ABLATION_MODES.get(mode, mode)),
            "esm_passes": r.get("esm_passes", "?"),
            "chain_mode": r.get("chain_mode", "?"),
            "crossattn": r.get("crossattn", False),
            "rosetta": r.get("rosetta", False),
            "auroc": r.get("test_auroc", np.nan),
            "f1": r.get("test_f1", np.nan),
            "mcc": r.get("test_mcc", np.nan),
            "precision": r.get("test_prec", np.nan),
            "recall": r.get("test_rec", np.nan),
            "accuracy": r.get("test_acc", np.nan),
            "trainable_params": r.get("trainable_params", 0),
            "total_params": r.get("total_params", 0),
            "train_time_sec": r.get("train_time_sec", 0),
            "best_epoch": r.get("best_epoch", -1),
        })
    return pd.DataFrame(rows)


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build pairwise comparison table with deltas."""
    rows = []
    for base, test, desc, category in PAIRWISE_COMPARISONS:
        if base not in results or test not in results:
            continue
        rb, rt = results[base], results[test]
        for metric in ["test_auroc", "test_f1", "test_mcc"]:
            vb = rb.get(metric, np.nan)
            vt = rt.get(metric, np.nan)
            if pd.notna(vb) and pd.notna(vt):
                delta = vt - vb
                pct = 100 * delta / abs(vb) if vb != 0 else 0
            else:
                delta = np.nan
                pct = np.nan
            metric_short = metric.replace("test_", "")
            rows.append({
                "baseline": base,
                "test": test,
                "comparison": desc,
                "category": category,
                "metric": metric_short,
                "baseline_val": vb,
                "test_val": vt,
                "delta": delta,
                "pct_change": pct,
            })
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame):
    """Print formatted results table to terminal."""
    print(f"\n{'='*100}")
    print("  ABLATION RESULTS")
    print(f"{'='*100}")
    header = (f"{'Mode':<5} {'Config':<30} {'ESM':<5} {'Xattn':<6} "
              f"{'Ros':<5} {'AUROC':>8} {'F1':>8} {'MCC':>8} "
              f"{'Prec':>8} {'Rec':>8} {'Params':>12} {'Time':>8}")
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        xa = "Y" if r["crossattn"] else "N"
        ro = "Y" if r["rosetta"] else "N"
        auroc = f"{r['auroc']:.4f}" if pd.notna(r['auroc']) else "  N/A "
        f1 = f"{r['f1']:.4f}" if pd.notna(r['f1']) else "  N/A "
        mcc = f"{r['mcc']:.4f}" if pd.notna(r['mcc']) else "  N/A "
        prec = f"{r['precision']:.4f}" if pd.notna(r['precision']) else "  N/A "
        rec = f"{r['recall']:.4f}" if pd.notna(r['recall']) else "  N/A "
        params = f"{r['trainable_params']:>10,}" if r['trainable_params'] else "       N/A"
        time_s = f"{r['train_time_sec']:>6.0f}s" if r['train_time_sec'] else "    N/A"
        print(f"{r['mode']:<5} {r['config']:<30} {r['esm_passes']:<5} "
              f"{xa:<6} {ro:<5} {auroc:>8} {f1:>8} {mcc:>8} "
              f"{prec:>8} {rec:>8} {params:>12} {time_s:>8}")


def print_comparisons(comp_df: pd.DataFrame):
    """Print pairwise comparisons grouped by category."""
    print(f"\n{'='*100}")
    print("  PAIRWISE COMPARISONS")
    print(f"{'='*100}")

    categories = [
        ("positional_encoding", "POSITIONAL ENCODING (chain splitting)"),
        ("cross_attention",     "CROSS-ATTENTION VALUE"),
        ("rosetta_value",       "ROSETTA FEATURE VALUE"),
        ("architecture_value",  "ARCHITECTURE VALUE (is complexity worth it?)"),
        ("total_improvement",   "TOTAL IMPROVEMENT (baseline vs best)"),
    ]

    for cat_key, cat_name in categories:
        subset = comp_df[comp_df["category"] == cat_key]
        if subset.empty:
            continue

        print(f"\n  {cat_name}")
        print(f"  {'-'*70}")

        # Pivot to show AUROC, F1, MCC side by side
        auroc_rows = subset[subset["metric"] == "auroc"]
        for _, r in auroc_rows.iterrows():
            # Get matching F1 and MCC
            f1_row = subset[(subset["baseline"] == r["baseline"]) &
                            (subset["test"] == r["test"]) &
                            (subset["metric"] == "f1")]
            mcc_row = subset[(subset["baseline"] == r["baseline"]) &
                             (subset["test"] == r["test"]) &
                             (subset["metric"] == "mcc")]

            auroc_d = r["delta"]
            f1_d = f1_row.iloc[0]["delta"] if len(f1_row) else np.nan
            mcc_d = mcc_row.iloc[0]["delta"] if len(mcc_row) else np.nan

            # Direction arrows
            def _arrow(d):
                if pd.isna(d): return "?"
                if d > 0.005: return "▲"
                if d < -0.005: return "▼"
                return "≈"

            a_arrow = _arrow(auroc_d)
            f_arrow = _arrow(f1_d)
            m_arrow = _arrow(mcc_d)

            auroc_str = f"{a_arrow} {auroc_d:+.4f}" if pd.notna(auroc_d) else "   N/A "
            f1_str = f"{f_arrow} {f1_d:+.4f}" if pd.notna(f1_d) else "   N/A "
            mcc_str = f"{m_arrow} {mcc_d:+.4f}" if pd.notna(mcc_d) else "   N/A "

            print(f"    {r['baseline']}→{r['test']}  "
                  f"AUROC {auroc_str}  F1 {f1_str}  MCC {mcc_str}  "
                  f"— {r['comparison']}")


def print_summary(results: dict, comp_df: pd.DataFrame):
    """Print executive summary with key findings."""
    print(f"\n{'='*100}")
    print("  EXECUTIVE SUMMARY")
    print(f"{'='*100}")

    modes_available = sorted(results.keys())
    print(f"\n  Modes completed: {' '.join(modes_available)} "
          f"({len(modes_available)}/{len(ABLATION_MODES)})")

    missing = [m for m in ABLATION_MODES if m not in results]
    if missing:
        print(f"  Missing: {' '.join(missing)}")

    # Best model
    best_mode = None
    best_auroc = -1
    for mode, r in results.items():
        a = r.get("test_auroc", 0)
        if a > best_auroc:
            best_auroc = a
            best_mode = mode

    if best_mode:
        r = results[best_mode]
        print(f"\n  Best model: Mode {best_mode} ({ABLATION_MODES.get(best_mode, best_mode)})")
        print(f"    AUROC={r.get('test_auroc', 0):.4f}  "
              f"F1={r.get('test_f1', 0):.4f}  "
              f"MCC={r.get('test_mcc', 0):.4f}")

    # Key questions answered
    auroc_comp = comp_df[comp_df["metric"] == "auroc"]

    def _get_delta(base, test):
        row = auroc_comp[(auroc_comp["baseline"] == base) & (auroc_comp["test"] == test)]
        return row.iloc[0]["delta"] if len(row) else None

    questions = []

    d = _get_delta("A", "B")
    if d is not None:
        sig = "YES" if abs(d) > 0.01 else "MARGINAL" if abs(d) > 0.005 else "NO"
        questions.append(f"  Q: Does fixing positional encoding help (1→4 chains)?")
        questions.append(f"     {sig} — ΔAUROC = {d:+.4f}")

    d = _get_delta("A", "G")
    if d is not None:
        sig = "YES" if abs(d) > 0.01 else "MARGINAL" if abs(d) > 0.005 else "NO"
        questions.append(f"  Q: Does 2-chain split help?")
        questions.append(f"     {sig} — ΔAUROC = {d:+.4f}")

    d = _get_delta("B", "D")
    if d is not None:
        sig = "YES" if abs(d) > 0.01 else "MARGINAL" if abs(d) > 0.005 else "NO"
        questions.append(f"  Q: Does cross-attention help over plain pooling?")
        questions.append(f"     {sig} — ΔAUROC = {d:+.4f}")

    # Average Rosetta lift
    rosetta_deltas = []
    for b, t in [("A", "F"), ("G", "H"), ("B", "C"), ("D", "E")]:
        d = _get_delta(b, t)
        if d is not None:
            rosetta_deltas.append(d)
    if rosetta_deltas:
        avg_lift = np.mean(rosetta_deltas)
        questions.append(f"  Q: Do Rosetta features help?")
        questions.append(f"     Average ΔAUROC = {avg_lift:+.4f} across {len(rosetta_deltas)} comparisons")

    d = _get_delta("F", "E")
    if d is not None:
        sig = "YES" if d > 0.02 else "MARGINAL" if d > 0.01 else "NO"
        questions.append(f"  Q: Is the full model worth 4× compute over simple+Rosetta?")
        questions.append(f"     {sig} — ΔAUROC = {d:+.4f} (for 4× compute cost)")

    d = _get_delta("A", "E")
    if d is not None:
        questions.append(f"  Q: Total improvement from baseline to full model?")
        questions.append(f"     ΔAUROC = {d:+.4f}")

    if questions:
        print("\n  KEY FINDINGS:")
        for q in questions:
            print(q)

    # Efficiency analysis
    if "A" in results and "E" in results:
        a_time = results["A"].get("train_time_sec", 0)
        e_time = results["E"].get("train_time_sec", 0)
        a_auroc = results["A"].get("test_auroc", 0)
        e_auroc = results["E"].get("test_auroc", 0)
        if a_time > 0 and e_time > 0:
            speedup = e_time / a_time
            print(f"\n  EFFICIENCY:")
            print(f"    Mode A: AUROC={a_auroc:.4f} in {a_time:.0f}s")
            print(f"    Mode E: AUROC={e_auroc:.4f} in {e_time:.0f}s ({speedup:.1f}× slower)")
            if e_auroc > a_auroc and (e_auroc - a_auroc) > 0:
                per_point = (e_time - a_time) / (e_auroc - a_auroc) / 100
                print(f"    Cost: {per_point:.0f}s per 0.01 AUROC improvement")

    # Recommendation
    if "F" in results and "E" in results:
        f_auroc = results["F"].get("test_auroc", 0)
        e_auroc = results["E"].get("test_auroc", 0)
        delta = e_auroc - f_auroc
        print(f"\n  RECOMMENDATION:")
        if delta < 0.01:
            print(f"    → Use Mode F (concat+Rosetta): simpler, 1× compute, "
                  f"only {delta:+.4f} AUROC behind full model")
        elif delta < 0.03:
            print(f"    → Mode E is {delta:+.4f} better but 4× compute. "
                  f"Consider Mode H (2-chain+Rosetta) as a middle ground.")
        else:
            print(f"    → Mode E (full model) is substantially better ({delta:+.4f} AUROC). "
                  f"Worth the compute cost.")


def generate_report(results_dir: Path, results: dict, df: pd.DataFrame,
                    comp_df: pd.DataFrame) -> str:
    """Generate a full text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("ABLATION STUDY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Results directory: {results_dir}")
    lines.append(f"Modes completed: {sorted(results.keys())}")
    lines.append("=" * 80)
    lines.append("")

    # Results table
    lines.append("RESULTS TABLE")
    lines.append("-" * 80)
    lines.append(df.to_string(index=False))
    lines.append("")

    # Comparisons
    lines.append("PAIRWISE COMPARISONS (AUROC)")
    lines.append("-" * 80)
    auroc_comp = comp_df[comp_df["metric"] == "auroc"]
    if not auroc_comp.empty:
        lines.append(auroc_comp.to_string(index=False))
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Ablation Results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing mode_A/, mode_B/, ... subdirs")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: results_dir/ablation_results.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    # ---- Load results ----
    results = load_results(results_dir)

    if not results:
        print(f"No result.json files found in {results_dir}/mode_*/")
        print("Make sure ablation jobs have completed.")
        sys.exit(1)

    print(f"Found results for {len(results)} modes: {sorted(results.keys())}")
    missing = [m for m in ABLATION_MODES if m not in results]
    if missing:
        print(f"Missing modes: {missing}")
        print("(continuing with available results)\n")

    # ---- Build tables ----
    df = build_results_table(results)
    comp_df = build_comparison_table(results)

    # ---- Print to terminal ----
    print_results_table(df)
    print_comparisons(comp_df)
    print_summary(results, comp_df)

    # ---- Save files ----
    csv_path = Path(args.output_csv) if args.output_csv else results_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    comp_csv = results_dir / "ablation_comparisons.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"Comparisons CSV: {comp_csv}")

    json_path = results_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(list(results.values()), f, indent=2)
    print(f"Results JSON: {json_path}")

    report = generate_report(results_dir, results, df, comp_df)
    report_path = results_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()