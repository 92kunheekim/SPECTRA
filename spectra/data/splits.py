"""
Data Preparation & Training Pipeline for Multimodal TCR-pMHC Binding Model
===========================================================================

Key components:
  1. Levenshtein-aware data splitting:
     - Group TCRs by peptide
     - Within each peptide group, cluster CDR3β sequences at 0.9 similarity
     - Split *clusters* (not individual samples) into train/val/test
     - This guarantees no CDR3β pair across partitions exceeds 0.9 similarity
       when paired with the same peptide

  2. Multiple cross-validation folds

  3. Training loop with the MultimodalBindingModel

Dependencies:
  pip install python-Levenshtein scikit-learn pandas torch dgl
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. Levenshtein similarity utilities
# ============================================================

def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Levenshtein similarity ratio ∈ [0, 1].
    ratio = 1 - (edit_distance / max(len(s1), len(s2)))
    """
    try:
        from Levenshtein import ratio
        return ratio(s1, s2)
    except ImportError:
        # Fallback: pure Python (slower)
        n, m = len(s1), len(s2)
        if n == 0 or m == 0:
            return 0.0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        dist = dp[n][m]
        return 1.0 - dist / max(n, m)


# ============================================================
# 2. CDR3β clustering (single-linkage at threshold)
# ============================================================

class UnionFind:
    """Union-Find for clustering sequences by similarity."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def clusters(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return list(groups.values())


def cluster_cdr3b_sequences(sequences: List[str], threshold: float = 0.9) -> List[List[int]]:
    """
    Cluster CDR3β sequences using single-linkage clustering.
    Two sequences are in the same cluster if their Levenshtein similarity
    ratio > threshold (directly or transitively).

    Args:
        sequences: list of CDR3β strings
        threshold: similarity threshold (default 0.9)

    Returns:
        List of clusters, each cluster is a list of indices into `sequences`
    """
    n = len(sequences)
    if n == 0:
        return []

    uf = UnionFind(n)

    # Pairwise comparison — O(n²) per peptide group
    # For very large groups, consider approximate methods (MMseqs2, CD-HIT)
    for i in range(n):
        for j in range(i + 1, n):
            if levenshtein_ratio(sequences[i], sequences[j]) > threshold:
                uf.union(i, j)

    return uf.clusters()


# ============================================================
# 3. Peptide-aware, Levenshtein-aware data splitting
# ============================================================

def levenshtein_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    similarity_threshold: float = 0.9,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset ensuring no CDR3β pair across partitions exceeds
    `similarity_threshold` similarity when paired with the same peptide.

    Strategy:
      1. Group samples by peptide
      2. Within each peptide group, cluster CDR3β at the threshold
      3. Shuffle and assign *whole clusters* to train/val/test
      4. This guarantees the Levenshtein constraint

    Args:
        df: DataFrame with columns including 'peptide' and 'CDR3b'
        train_ratio, val_ratio, test_ratio: split proportions (must sum to 1)
        similarity_threshold: max allowed Levenshtein ratio across partitions
        seed: random seed

    Returns:
        train_indices, val_indices, test_indices (indices into df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1, got {train_ratio + val_ratio + test_ratio}"

    rng = random.Random(seed)

    train_idx, val_idx, test_idx = [], [], []

    # Group by peptide
    peptide_groups = df.groupby("peptide")

    for peptide, group in peptide_groups:
        group_indices = group.index.tolist()
        cdr3b_seqs = group["CDR3b"].tolist()

        # Cluster CDR3β sequences within this peptide group
        clusters = cluster_cdr3b_sequences(cdr3b_seqs, threshold=similarity_threshold)

        # Each cluster maps local indices → global df indices
        cluster_global = []
        for cluster in clusters:
            global_idxs = [group_indices[i] for i in cluster]
            cluster_global.append(global_idxs)

        # Shuffle clusters
        rng.shuffle(cluster_global)

        # Count total samples and compute split points
        total = len(group_indices)
        n_train = max(1, round(total * train_ratio))
        n_val = max(1, round(total * val_ratio))
        # rest goes to test

        # Assign clusters greedily to partitions
        counts = {"train": 0, "val": 0, "test": 0}
        assignments = {"train": [], "val": [], "test": []}

        for cluster_idxs in cluster_global:
            c_size = len(cluster_idxs)

            # Assign to the partition that is most under-filled relative to target
            train_gap = n_train - counts["train"]
            val_gap = n_val - counts["val"]
            test_gap = (total - n_train - n_val) - counts["test"]

            # Pick partition with largest remaining gap
            gaps = [("train", train_gap), ("val", val_gap), ("test", test_gap)]
            target = max(gaps, key=lambda x: x[1])[0]

            assignments[target].extend(cluster_idxs)
            counts[target] += c_size

        train_idx.extend(assignments["train"])
        val_idx.extend(assignments["val"])
        test_idx.extend(assignments["test"])

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"(total={len(df)})"
    )

    return train_idx, val_idx, test_idx


def collect_cross_partition_similarities(
    df: pd.DataFrame,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    threshold: float = 0.9,
) -> Tuple[bool, Dict[str, Dict[str, List[float]]]]:
    """
    Compute all pairwise CDR3β Levenshtein similarities across partitions,
    grouped by peptide and separated by label (binder vs non-binder).

    Returns:
        (is_clean, sim_data)
        is_clean: True if no violations exceed threshold
        sim_data: nested dict  sim_data[partition_pair][label_name] = [similarities]
            partition_pair ∈ {"Train ↔ Val", "Train ↔ Test", "Val ↔ Test"}
            label_name ∈ {"Binders", "Non-binders"}
    """
    partitions = {"Train": set(train_idx), "Val": set(val_idx), "Test": set(test_idx)}
    partition_pairs = [("Train", "Val"), ("Train", "Test"), ("Val", "Test")]
    label_map = {1: "Binders", 0: "Non-binders"}

    sim_data = {}
    for pa, pb in partition_pairs:
        pair_name = f"{pa} ↔ {pb}"
        sim_data[pair_name] = {"Binders": [], "Non-binders": []}

    violations = 0
    for peptide, group in df.groupby("peptide"):
        for label_val, label_name in label_map.items():
            grp = group[group["label"] == label_val]
            if len(grp) == 0:
                continue

            for pa, pb in partition_pairs:
                pair_name = f"{pa} ↔ {pb}"
                idxs_a = [i for i in grp.index if i in partitions[pa]]
                idxs_b = [i for i in grp.index if i in partitions[pb]]

                for ia in idxs_a:
                    for ib in idxs_b:
                        sim = levenshtein_ratio(df.loc[ia, "CDR3b"], df.loc[ib, "CDR3b"])
                        sim_data[pair_name][label_name].append(sim)
                        if sim > threshold:
                            violations += 1
                            if violations <= 5:
                                logger.warning(
                                    f"LEAKAGE: peptide={peptide}, "
                                    f"{pa}[{ia}] CDR3b={df.loc[ia, 'CDR3b']} ↔ "
                                    f"{pb}[{ib}] CDR3b={df.loc[ib, 'CDR3b']} "
                                    f"sim={sim:.3f} ({label_name})"
                                )

    is_clean = violations == 0
    if is_clean:
        logger.info("✓ No Levenshtein leakage detected across partitions.")
    else:
        logger.error(f"Total violations: {violations}")

    return is_clean, sim_data


def verify_split_leakage(
    df: pd.DataFrame,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    threshold: float = 0.9,
) -> bool:
    """Backward-compatible wrapper. Returns True if clean."""
    is_clean, _ = collect_cross_partition_similarities(
        df, train_idx, val_idx, test_idx, threshold
    )
    return is_clean


def plot_fold_leakage(
    fold_idx: int,
    sim_data: Dict[str, Dict[str, List[float]]],
    threshold: float = 0.9,
    output_path: Optional[str] = None,
):
    """
    Plot a 2×3 histogram figure for one fold.

    Rows: Binders, Non-binders
    Cols: Train↔Val, Train↔Test, Val↔Test

    Each panel shows the distribution of all pairwise cross-partition CDR3β
    Levenshtein similarities, with a red dashed threshold line and a status
    badge indicating whether leakage was detected.
    """
    import numpy as np

    pair_names = ["Train ↔ Val", "Train ↔ Test", "Val ↔ Test"]
    label_names = ["Binders", "Non-binders"]
    label_colors = {"Binders": "#2563EB", "Non-binders": "#9333EA"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(
        f"Fold {fold_idx} — Cross-Partition CDR3β Levenshtein Similarity",
        fontsize=16, fontweight="bold", y=1.02,
    )

    bins = np.linspace(0, 1, 51)

    for row, label_name in enumerate(label_names):
        for col, pair_name in enumerate(pair_names):
            ax = axes[row, col]
            sims = sim_data.get(pair_name, {}).get(label_name, [])

            if len(sims) == 0:
                ax.text(0.5, 0.5, "No pairs", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#9CA3AF")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                sims_arr = np.array(sims)
                max_sim = sims_arr.max()
                n_violations = int((sims_arr > threshold).sum())

                # Histogram
                counts, _, patches = ax.hist(
                    sims_arr, bins=bins,
                    color=label_colors[label_name],
                    alpha=0.7, edgecolor="white", linewidth=0.5,
                )

                # Color bins above threshold red
                for patch, left_edge in zip(patches, bins[:-1]):
                    if left_edge >= threshold:
                        patch.set_facecolor("#EF4444")
                        patch.set_alpha(0.9)

                # Threshold line
                ax.axvline(x=threshold, color="#EF4444", linestyle="--",
                           linewidth=2, alpha=0.8)

                # Status badge
                if n_violations == 0:
                    badge_color, text_color = "#DCFCE7", "#166534"
                    badge_text = f"✓ No leakage\nmax = {max_sim:.3f}\nn = {len(sims):,}"
                else:
                    badge_color, text_color = "#FEE2E2", "#991B1B"
                    badge_text = f"✗ {n_violations} violations\nmax = {max_sim:.3f}\nn = {len(sims):,}"

                ax.text(
                    0.03, 0.95, badge_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", fontfamily="monospace",
                    color=text_color,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color,
                              edgecolor=text_color, alpha=0.9, linewidth=1.5),
                )

            # Labels
            if row == 0:
                ax.set_title(pair_name, fontsize=13, fontweight="bold", pad=8)
            if row == 1:
                ax.set_xlabel("Levenshtein Similarity", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{label_name}\nCount", fontsize=11, fontweight="bold")

            ax.set_xlim(0, 1.02)
            ax.tick_params(labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved leakage figure: {output_path}")

    plt.close(fig)


# ============================================================
# 4. Multiple cross-validation fold generation
# ============================================================

def generate_cv_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    similarity_threshold: float = 0.9,
    base_seed: int = 42,
    verify: bool = True,
    figure_dir: Optional[str] = None,
) -> List[Dict[str, List[int]]]:
    """
    Generate multiple cross-validation folds with Levenshtein-aware splitting.

    Each fold uses a different random seed for cluster shuffling, producing
    different train/val/test partitions while always respecting the similarity
    constraint.

    Args:
        df: full dataset DataFrame
        n_folds: number of CV folds
        similarity_threshold: max CDR3β similarity across partitions (per peptide)
        verify: if True, run verification on each fold
        figure_dir: if set, save a 2×3 leakage histogram per fold to this directory

    Returns:
        List of dicts, each with keys 'train', 'val', 'test' → list of indices
    """
    if figure_dir is not None:
        Path(figure_dir).mkdir(parents=True, exist_ok=True)

    folds = []

    for fold_i in range(n_folds):
        seed = base_seed + fold_i * 1000  # well-separated seeds
        logger.info(f"--- Generating fold {fold_i + 1}/{n_folds} (seed={seed}) ---")

        train_idx, val_idx, test_idx = levenshtein_aware_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            similarity_threshold=similarity_threshold,
            seed=seed,
        )

        if verify or figure_dir is not None:
            clean, sim_data = collect_cross_partition_similarities(
                df, train_idx, val_idx, test_idx, similarity_threshold
            )
            if not clean:
                logger.warning(f"Fold {fold_i} has leakage! Consider re-examining clusters.")

            if figure_dir is not None:
                fig_path = os.path.join(figure_dir, f"fold_{fold_i}_leakage.png")
                plot_fold_leakage(fold_i, sim_data, similarity_threshold, fig_path)

        folds.append({
            "fold": fold_i,
            "seed": seed,
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })

        # Log class distribution per split
        for split_name in ["train", "val", "test"]:
            idxs = folds[-1][split_name]
            labels = df.loc[idxs, "label"]
            n_pos = (labels == 1).sum()
            n_neg = (labels == 0).sum()
            logger.info(f"  {split_name}: {len(idxs)} samples (pos={n_pos}, neg={n_neg})")

    return folds


def save_folds(folds: List[Dict], path: str):
    """Save fold indices to JSON for reproducibility."""
    serializable = []
    for f in folds:
        serializable.append({
            "fold": f["fold"],
            "seed": f["seed"],
            "train": [int(i) for i in f["train"]],
            "val": [int(i) for i in f["val"]],
            "test": [int(i) for i in f["test"]],
        })
    with open(path, "w") as fp:
        json.dump(serializable, fp, indent=2)
    logger.info(f"Saved {len(folds)} folds to {path}")


def load_folds(path: str) -> List[Dict]:
    with open(path) as fp:
        return json.load(fp)



# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Levenshtein-aware CV folds")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV with columns [CDR3a, CDR3b, MHC Sequence, peptide, "
                             "TCR_A_sequence, TCR_B_sequence, label]")
    parser.add_argument("--output", default="./cv_folds.json",
                        help="Output path for fold indices JSON")
    parser.add_argument("--n_folds", type=int, default=15)
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip leakage verification (faster)")
    parser.add_argument("--figure_dir", default=None,
                        help="Directory to save leakage histogram figures per fold")

    args = parser.parse_args()

    # Load and validate
    df = pd.read_csv(args.csv)
    required = {"CDR3a", "CDR3b", "MHC Sequence", "peptide",
                "TCR_A_sequence", "TCR_B_sequence", "label"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    df = df.dropna(subset=list(required)).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} samples | {df['label'].sum()} pos, "
                f"{(df['label']==0).sum()} neg | {df['peptide'].nunique()} peptides")

    # Generate folds
    folds = generate_cv_folds(
        df,
        n_folds=args.n_folds,
        similarity_threshold=args.similarity_threshold,
        base_seed=args.seed,
        verify=not args.no_verify,
        figure_dir=args.figure_dir,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_folds(folds, args.output)