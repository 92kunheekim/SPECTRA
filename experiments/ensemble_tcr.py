"""
ensemble_tcr.py — Ensemble Training & Inference for TCR-pMHC Binding
======================================================================

Trains N independent models (default 5) with different random seeds,
then combines their predictions at inference time using multiple
ensemble strategies.

Why ensembling works for this problem:
  - Each seed produces a different initialization → different local minimum
  - Different seeds cause different mini-batch orderings → different
    gradient trajectories
  - The models make partially uncorrelated errors, so averaging
    predictions cancels out individual mistakes
  - Uncertainty estimation: prediction variance across ensemble members
    provides a natural confidence score

Ensemble strategies:
  1. PROBABILITY AVERAGING: mean of sigmoid outputs (simplest, usually best)
  2. LOGIT AVERAGING:       mean of raw logits → sigmoid (preserves scale)
  3. MAJORITY VOTING:       threshold each model → count votes
  4. RANK AVERAGING:        rank-normalize probabilities, then average
     (robust to poorly calibrated members)
  5. LEARNED STACKING:      logistic regression on 5 member probabilities
     (learns optimal weights, needs validation data)

Workflow:
  Step 1: Train 5 models
    python ensemble_tcr.py train --data_csv data.csv --seeds 42 123 456 789 2024

  Step 2: Evaluate ensemble on held-out test set
    python ensemble_tcr.py evaluate --data_csv data.csv --ckpt_dir outputs/ensemble

  Step 3: Predict on new data
    python ensemble_tcr.py predict --input_csv new_samples.csv --ckpt_dir outputs/ensemble

Compatible with existing dataset_ablation.py and model_improved.py.
"""

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from torchmetrics.classification import (
    BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall,
    BinaryAccuracy, BinaryMatthewsCorrCoef,
)

from dataset_ablation import AblationDataset, make_ablation_collate_fn
from model_improved import ImprovedTCRpMHCModel

# Also import the data-splitting and Lightning module from run_improved
from run_improved import (
    load_datasets_3way,
    GraduatedUnfreeze,
    ImprovedLightning,
)


# ============================================================
# 1. Train a Single Ensemble Member
# ============================================================

def train_one_member(
    member_id: int,
    seed: int,
    train_ds,
    val_ds,
    collate_fn,
    esm_checkpoint: str,
    args,
    pos_weight: float,
    out_dir: Path,
):
    """
    Train one ensemble member. Returns the best checkpoint path and
    validation metrics.
    """
    L.seed_everything(seed)
    member_dir = out_dir / f"member_{member_id}_seed_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model ----
    esm_model = AutoModel.from_pretrained(esm_checkpoint)

    model = ImprovedTCRpMHCModel(
        esm_model=esm_model,
        esm_hidden=esm_model.config.hidden_size,
        freeze_esm=True,
        n_tune_layers=0,
        d_model=args.d_model,
        n_cross_heads=args.n_cross_heads,
        n_cross_layers=args.n_cross_layers,
        dropout=args.dropout,
        use_rosetta=True,
        d_rosetta=args.d_rosetta,
        d_fused=args.d_fused,
        clf_hidden=args.d_fused,
        pos_weight=pos_weight,
    )

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*65}")
    print(f"  ENSEMBLE MEMBER {member_id + 1} / {args.n_ensemble}")
    print(f"  Seed: {seed}")
    print(f"  Params: {tp:,} trainable / {tt:,} total")
    print(f"{'='*65}")

    # ---- Lightning module ----
    lit = ImprovedLightning(
        model=model, lr=args.lr, esm_lr=args.esm_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        pos_weight=pos_weight,
    )

    # ---- Data loaders ----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ---- Callbacks ----
    ckpt_cb = ModelCheckpoint(
        dirpath=member_dir / "ckpt", monitor="val_auroc", mode="max",
        save_top_k=1, filename=f"member{member_id}" + "-{epoch:02d}-{val_auroc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", devices=1,
        callbacks=[
            ckpt_cb,
            GraduatedUnfreeze(args.unfreeze_b, args.unfreeze_c),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_auroc", patience=args.patience, mode="max"),
        ],
        logger=CSVLogger(save_dir=str(member_dir), name="logs"),
        gradient_clip_val=1.0,
        enable_progress_bar=args.progress_bar,
    )

    # ---- Train ----
    t0 = time.time()
    trainer.fit(lit, train_loader, val_loader)
    train_time = time.time() - t0

    best_ckpt = ckpt_cb.best_model_path

    # ---- Save member metadata ----
    meta = {
        "member_id": member_id,
        "seed": seed,
        "best_checkpoint": str(best_ckpt),
        "best_val_auroc": float(ckpt_cb.best_model_score or 0),
        "train_time_sec": round(train_time, 1),
        "trainable_params": tp,
        "total_params": tt,
    }
    with open(member_dir / "member_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Member {member_id + 1} complete: "
          f"val_auroc={meta['best_val_auroc']:.4f}, "
          f"time={train_time:.0f}s")

    return meta


# ============================================================
# 2. Load Trained Ensemble Members for Inference
# ============================================================

def load_ensemble_members(
    ckpt_dir: Path,
    esm_checkpoint: str,
    args,
    device: torch.device,
):
    """
    Load all trained ensemble member checkpoints.

    Scans ckpt_dir for member_*/member_meta.json files, loads each
    model from its best checkpoint, and returns them in eval mode.

    Returns:
        models: list of ImprovedTCRpMHCModel in eval mode
        metas:  list of member metadata dicts
    """
    member_dirs = sorted(ckpt_dir.glob("member_*_seed_*"))
    if not member_dirs:
        raise FileNotFoundError(
            f"No member directories found in {ckpt_dir}. "
            f"Run 'ensemble_tcr.py train' first."
        )

    models = []
    metas = []

    for mdir in member_dirs:
        meta_path = mdir / "member_meta.json"
        if not meta_path.exists():
            print(f"  WARNING: {meta_path} not found, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        ckpt_path = meta["best_checkpoint"]
        if not Path(ckpt_path).exists():
            print(f"  WARNING: checkpoint {ckpt_path} not found, skipping")
            continue

        print(f"  Loading member {meta['member_id']} "
              f"(seed={meta['seed']}, val_auroc={meta['best_val_auroc']:.4f})")

        # Build model architecture
        esm_model = AutoModel.from_pretrained(esm_checkpoint)
        model = ImprovedTCRpMHCModel(
            esm_model=esm_model,
            esm_hidden=esm_model.config.hidden_size,
            freeze_esm=True,
            d_model=args.d_model,
            n_cross_heads=args.n_cross_heads,
            n_cross_layers=args.n_cross_layers,
            dropout=args.dropout,
            use_rosetta=True,
            d_rosetta=args.d_rosetta,
            d_fused=args.d_fused,
            clf_hidden=args.d_fused,
        )

        # Load weights from Lightning checkpoint
        lit = ImprovedLightning.load_from_checkpoint(
            ckpt_path, model=model,
        )
        model = lit.model
        model.eval()
        model.to(device)
        models.append(model)
        metas.append(meta)

    print(f"\n  Loaded {len(models)} ensemble members")
    return models, metas


# ============================================================
# 3. Ensemble Inference (Collect Per-Member Predictions)
# ============================================================

@torch.no_grad()
def collect_member_predictions(
    models: list,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Run inference with each ensemble member, collect raw outputs.

    Returns:
        all_logits:  [N_members, N_samples] numpy array
        all_probs:   [N_members, N_samples] numpy array
        all_labels:  [N_samples] numpy array (or None if no labels)
    """
    n_members = len(models)

    # Collect predictions per member
    member_logits = [[] for _ in range(n_members)]
    member_probs = [[] for _ in range(n_members)]
    labels_list = []
    has_labels = True

    for batch in dataloader:
        # Move batch to device
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        if has_labels and "labels" in batch_dev:
            labels_list.append(batch_dev["labels"].cpu().numpy())
        else:
            has_labels = False

        for i, model in enumerate(models):
            out = model(
                concat_ids=batch_dev.get("concat_ids"),
                concat_mask=batch_dev.get("concat_mask"),
                pmhc_ids=batch_dev.get("pmhc_ids"),
                pmhc_mask=batch_dev.get("pmhc_mask"),
                tcr_ids=batch_dev.get("tcr_ids"),
                tcr_mask=batch_dev.get("tcr_mask"),
                mhc_ids=batch_dev.get("mhc_ids"),
                mhc_mask=batch_dev.get("mhc_mask"),
                pep_ids=batch_dev.get("pep_ids"),
                pep_mask=batch_dev.get("pep_mask"),
                tra_ids=batch_dev.get("tra_ids"),
                tra_mask=batch_dev.get("tra_mask"),
                trb_ids=batch_dev.get("trb_ids"),
                trb_mask=batch_dev.get("trb_mask"),
                rosetta_features=batch_dev.get("rosetta"),
                rosetta_available=batch_dev.get("has_rosetta"),
                compute_loss=False,
            )
            member_logits[i].append(out["logit"].view(-1).cpu().numpy())
            member_probs[i].append(out["prob"].view(-1).cpu().numpy())

    # Stack into arrays
    all_logits = np.array([np.concatenate(ml) for ml in member_logits])
    all_probs = np.array([np.concatenate(mp) for mp in member_probs])
    all_labels = np.concatenate(labels_list) if has_labels and labels_list else None

    return all_logits, all_probs, all_labels


# ============================================================
# 4. Ensemble Combination Strategies
# ============================================================

def ensemble_probability_averaging(all_probs):
    """
    Average the sigmoid probabilities across ensemble members.

    This is the simplest and most commonly effective strategy.
    Each model contributes equally to the final prediction.

    Input:  [N_members, N_samples]
    Output: [N_samples]
    """
    return all_probs.mean(axis=0)


def ensemble_logit_averaging(all_logits):
    """
    Average raw logits, then apply sigmoid.

    Preserves the natural scale of the logits before squashing.
    Can be better than probability averaging when models are
    well-calibrated and the logit space is more linearly separable.

    Input:  [N_members, N_samples]
    Output: [N_samples]
    """
    mean_logit = all_logits.mean(axis=0)
    return 1.0 / (1.0 + np.exp(-mean_logit))


def ensemble_majority_voting(all_probs, threshold=0.5):
    """
    Each model votes binding/non-binding, majority wins.

    Good for deployment where you need a binary decision with
    interpretable consensus (e.g., "4 out of 5 models agree").

    Input:  [N_members, N_samples]
    Output: [N_samples] vote fractions (0.0 to 1.0)
    """
    votes = (all_probs > threshold).astype(float)
    return votes.mean(axis=0)


def ensemble_rank_averaging(all_probs):
    """
    Rank-normalize each member's probabilities, then average ranks.

    Robust to poorly calibrated members. If one model systematically
    predicts higher probabilities than others, rank normalization
    puts all members on the same scale before combining.

    Input:  [N_members, N_samples]
    Output: [N_samples] average normalized ranks (0 to 1)
    """
    from scipy.stats import rankdata
    n_members, n_samples = all_probs.shape
    ranks = np.zeros_like(all_probs)
    for i in range(n_members):
        ranks[i] = rankdata(all_probs[i]) / n_samples
    return ranks.mean(axis=0)


def ensemble_learned_stacking(all_probs, labels, val_probs=None, val_labels=None):
    """
    Train a logistic regression on member probabilities.

    Instead of equal averaging, this learns the optimal weight for
    each ensemble member. A model that consistently makes better
    predictions gets higher weight.

    If val_probs/val_labels are not provided, uses 5-fold CV on the
    provided data. Otherwise trains on val and predicts on the main data.

    Input:  all_probs [N_members, N_samples], labels [N_samples]
    Output: [N_samples] stacked probabilities
    """
    from sklearn.linear_model import LogisticRegressionCV

    X = all_probs.T  # [N_samples, N_members]

    if val_probs is not None and val_labels is not None:
        # Train on validation set, predict on test set
        X_train = val_probs.T
        y_train = val_labels
        lr = LogisticRegressionCV(cv=3, scoring="roc_auc", max_iter=1000)
        lr.fit(X_train, y_train)
        stacked = lr.predict_proba(X)[:, 1]
        weights = lr.coef_[0]
    else:
        # 5-fold CV on the provided data
        lr = LogisticRegressionCV(cv=5, scoring="roc_auc", max_iter=1000)
        lr.fit(X, labels)
        stacked = lr.predict_proba(X)[:, 1]
        weights = lr.coef_[0]

    return stacked, weights


# ============================================================
# 5. Compute Metrics for Ensemble
# ============================================================

def compute_metrics(probs, labels, threshold=0.5):
    """Compute all binary classification metrics."""
    probs_t = torch.tensor(probs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    metrics = {}
    metrics["auroc"] = float(BinaryAUROC()(probs_t, labels_t))
    metrics["f1"] = float(BinaryF1Score()(probs_t, labels_t))
    metrics["mcc"] = float(BinaryMatthewsCorrCoef()(probs_t, labels_t))
    metrics["precision"] = float(BinaryPrecision()(probs_t, labels_t))
    metrics["recall"] = float(BinaryRecall()(probs_t, labels_t))
    metrics["accuracy"] = float(BinaryAccuracy()(probs_t, labels_t))
    return metrics


def compute_uncertainty(all_probs):
    """
    Compute prediction uncertainty from ensemble disagreement.

    Returns per-sample metrics:
      - std:       standard deviation across members (simple spread)
      - entropy:   entropy of mean prediction (information-theoretic)
      - mi:        mutual information = total uncertainty - aleatoric
                   (epistemic uncertainty the ensemble is unsure about)
    """
    mean_prob = all_probs.mean(axis=0)
    std = all_probs.std(axis=0)

    # Predictive entropy: H[mean prediction]
    eps = 1e-10
    entropy = -(mean_prob * np.log(mean_prob + eps) +
                (1 - mean_prob) * np.log(1 - mean_prob + eps))

    # Mean member entropy (aleatoric)
    member_entropies = -(all_probs * np.log(all_probs + eps) +
                         (1 - all_probs) * np.log(1 - all_probs + eps))
    mean_member_entropy = member_entropies.mean(axis=0)

    # Mutual information (epistemic) = total - aleatoric
    mi = entropy - mean_member_entropy

    return {
        "std": std,
        "entropy": entropy,
        "mutual_information": mi,
    }


# ============================================================
# 6. Training Entrypoint
# ============================================================

def cmd_train(args):
    """Train N ensemble members."""
    out_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds[:args.n_ensemble]
    if len(seeds) < args.n_ensemble:
        # Generate additional seeds if not enough provided
        rng = np.random.RandomState(seeds[0] if seeds else 42)
        while len(seeds) < args.n_ensemble:
            seeds.append(int(rng.randint(0, 100000)))

    print(f"\n{'='*65}")
    print(f"  ENSEMBLE TRAINING: {args.n_ensemble} members")
    print(f"  Seeds: {seeds}")
    print(f"  Split: {'peptide-level' if args.peptide_split else 'standard stratified'}")
    print(f"  ESM: {args.esm_checkpoint}")
    print(f"{'='*65}")

    # ---- Data (consistent split across all members) ----
    train_ds, val_ds, test_ds, pos_weight = load_datasets_3way(
        data_csv=args.data_csv,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        crystal_weight=args.crystal_weight,
        seed=seeds[0],  # consistent split
        peptide_split=args.peptide_split,
    )

    print(f"\n  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  pos_weight: {pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_checkpoint)
    collate_fn = make_ablation_collate_fn(tokenizer)

    # ---- Train each member ----
    all_metas = []
    for i, seed in enumerate(seeds):
        meta = train_one_member(
            member_id=i,
            seed=seed,
            train_ds=train_ds,
            val_ds=val_ds,
            collate_fn=collate_fn,
            esm_checkpoint=args.esm_checkpoint,
            args=args,
            pos_weight=pos_weight,
            out_dir=out_dir,
        )
        all_metas.append(meta)

    # ---- Save ensemble config ----
    ensemble_config = {
        "n_members": len(all_metas),
        "seeds": seeds,
        "members": all_metas,
        "data_csv": args.data_csv,
        "esm_checkpoint": args.esm_checkpoint,
        "split_mode": "peptide" if args.peptide_split else "random",
        "d_model": args.d_model,
        "n_cross_heads": args.n_cross_heads,
        "n_cross_layers": args.n_cross_layers,
        "d_rosetta": args.d_rosetta,
        "d_fused": args.d_fused,
        "dropout": args.dropout,
    }
    with open(out_dir / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    # ---- Summary ----
    val_aurocs = [m["best_val_auroc"] for m in all_metas]
    print(f"\n{'='*65}")
    print(f"  ENSEMBLE TRAINING COMPLETE")
    print(f"  Members: {len(all_metas)}")
    print(f"  Val AUROC per member: {[f'{a:.4f}' for a in val_aurocs]}")
    print(f"  Val AUROC mean: {np.mean(val_aurocs):.4f} ± {np.std(val_aurocs):.4f}")
    print(f"  Config saved: {out_dir / 'ensemble_config.json'}")
    print(f"{'='*65}")


# ============================================================
# 7. Evaluation Entrypoint
# ============================================================

def cmd_evaluate(args):
    """Evaluate ensemble on held-out test set with all strategies."""
    out_dir = Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load ensemble ----
    print(f"\n  Loading ensemble from {out_dir}...")
    models, metas = load_ensemble_members(out_dir, args.esm_checkpoint, args, device)
    n_members = len(models)

    # ---- Load data ----
    config_path = out_dir / "ensemble_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        peptide_split = config.get("split_mode") == "peptide"
        split_seed = config["seeds"][0]
    else:
        peptide_split = args.peptide_split
        split_seed = args.seeds[0] if args.seeds else 42

    train_ds, val_ds, test_ds, pos_weight = load_datasets_3way(
        data_csv=args.data_csv,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        crystal_weight=1.0,  # no upweighting during evaluation
        seed=split_seed,
        peptide_split=peptide_split,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.esm_checkpoint)
    collate_fn = make_ablation_collate_fn(tokenizer)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ---- Collect predictions ----
    print(f"\n  Collecting test predictions from {n_members} members...")
    test_logits, test_probs, test_labels = collect_member_predictions(
        models, test_loader, device,
    )
    print(f"  Test set: {test_probs.shape[1]} samples")

    print(f"  Collecting val predictions for stacking...")
    val_logits, val_probs, val_labels = collect_member_predictions(
        models, val_loader, device,
    )

    # ---- Per-member metrics (baseline) ----
    print(f"\n{'='*90}")
    print("  PER-MEMBER RESULTS (individual models)")
    print(f"{'='*90}")
    header = f"{'Member':<10} {'Seed':<8} {'AUROC':>8} {'F1':>8} {'MCC':>8} {'ValAUROC':>10}"
    print(header)
    print("-" * len(header))

    for i in range(n_members):
        m = compute_metrics(test_probs[i], test_labels)
        print(f"{'  #' + str(i):<10} {metas[i]['seed']:<8} "
              f"{m['auroc']:>8.4f} {m['f1']:>8.4f} {m['mcc']:>8.4f} "
              f"{metas[i]['best_val_auroc']:>10.4f}")

    # ---- Ensemble strategies ----
    strategies = {}

    # 1. Probability averaging
    prob_avg = ensemble_probability_averaging(test_probs)
    strategies["prob_averaging"] = prob_avg

    # 2. Logit averaging
    logit_avg = ensemble_logit_averaging(test_logits)
    strategies["logit_averaging"] = logit_avg

    # 3. Majority voting
    vote_avg = ensemble_majority_voting(test_probs, threshold=0.5)
    strategies["majority_voting"] = vote_avg

    # 4. Rank averaging
    try:
        rank_avg = ensemble_rank_averaging(test_probs)
        strategies["rank_averaging"] = rank_avg
    except ImportError:
        print("  (scipy not available, skipping rank averaging)")

    # 5. Learned stacking
    try:
        stacked, weights = ensemble_learned_stacking(
            test_probs, test_labels,
            val_probs=val_probs, val_labels=val_labels,
        )
        strategies["learned_stacking"] = stacked

        print(f"\n  Stacking weights: {[f'{w:.3f}' for w in weights]}")
    except Exception as e:
        print(f"  (Stacking failed: {e})")

    # ---- Ensemble metrics ----
    print(f"\n{'='*90}")
    print("  ENSEMBLE RESULTS (combined predictions)")
    print(f"{'='*90}")
    header = f"{'Strategy':<22} {'AUROC':>8} {'F1':>8} {'MCC':>8} {'Prec':>8} {'Rec':>8}"
    print(header)
    print("-" * len(header))

    # Best individual for reference
    best_member_auroc = 0
    for i in range(n_members):
        m = compute_metrics(test_probs[i], test_labels)
        if m["auroc"] > best_member_auroc:
            best_member_auroc = m["auroc"]
            best_individual = m

    print(f"{'best_individual':<22} {best_individual['auroc']:>8.4f} "
          f"{best_individual['f1']:>8.4f} {best_individual['mcc']:>8.4f} "
          f"{best_individual['precision']:>8.4f} {best_individual['recall']:>8.4f}")

    all_strategy_results = {}
    for name, preds in strategies.items():
        m = compute_metrics(preds, test_labels)
        all_strategy_results[name] = m
        delta = m["auroc"] - best_individual["auroc"]
        arrow = "▲" if delta > 0.001 else "▼" if delta < -0.001 else "≈"
        print(f"{name:<22} {m['auroc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['mcc']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f" {arrow}{delta:+.4f}")

    # ---- Uncertainty analysis ----
    uncertainty = compute_uncertainty(test_probs)
    print(f"\n{'='*90}")
    print("  UNCERTAINTY ANALYSIS")
    print(f"{'='*90}")
    print(f"  Mean prediction std:            {uncertainty['std'].mean():.4f}")
    print(f"  Mean predictive entropy:        {uncertainty['entropy'].mean():.4f}")
    print(f"  Mean epistemic uncertainty (MI): {uncertainty['mutual_information'].mean():.4f}")

    # Uncertainty-accuracy correlation
    mean_probs = test_probs.mean(axis=0)
    correct = ((mean_probs > 0.5) == test_labels).astype(float)
    high_conf = uncertainty["std"] < np.median(uncertainty["std"])
    low_conf = ~high_conf

    if high_conf.sum() > 0 and low_conf.sum() > 0:
        acc_high = correct[high_conf].mean()
        acc_low = correct[low_conf].mean()
        print(f"\n  High-confidence samples ({high_conf.sum()}): accuracy = {acc_high:.4f}")
        print(f"  Low-confidence samples  ({low_conf.sum()}): accuracy = {acc_low:.4f}")
        print(f"  Confidence gap: {acc_high - acc_low:+.4f}")
        print(f"  → Ensemble uncertainty is {'informative' if acc_high > acc_low else 'NOT informative'}")

    # ---- Save results ----
    results = {
        "n_members": n_members,
        "n_test_samples": int(test_probs.shape[1]),
        "per_member": [
            {**compute_metrics(test_probs[i], test_labels), "seed": metas[i]["seed"]}
            for i in range(n_members)
        ],
        "ensemble_strategies": all_strategy_results,
        "uncertainty": {
            "mean_std": float(uncertainty["std"].mean()),
            "mean_entropy": float(uncertainty["entropy"].mean()),
            "mean_MI": float(uncertainty["mutual_information"].mean()),
        },
    }

    with open(out_dir / "ensemble_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_dir / 'ensemble_eval_results.json'}")


# ============================================================
# 8. Prediction Entrypoint (New Data)
# ============================================================

def cmd_predict(args):
    """
    Run ensemble prediction on new samples.

    Input CSV needs: tra_seq, trb_seq, peptide, mhc_seq
    (and optionally Rosetta features and tcr_id).

    Output CSV adds: prob_mean, prob_std, prediction, member_0..N probs
    """
    out_dir = Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load ensemble ----
    print(f"\n  Loading ensemble from {out_dir}...")
    models, metas = load_ensemble_members(out_dir, args.esm_checkpoint, args, device)
    n_members = len(models)

    # ---- Load input data ----
    df = pd.read_csv(args.input_csv)

    # Add required columns if missing
    if "tcr_id" not in df.columns:
        df["tcr_id"] = [f"sample_{i:06d}" for i in range(len(df))]
    if "binding_label" not in df.columns:
        df["binding_label"] = 0  # dummy, won't be used
    if "domain" not in df.columns:
        df["domain"] = 0

    print(f"  Input: {len(df)} samples from {args.input_csv}")

    # Build dataset
    ds = AblationDataset(df, crystal_weight=1.0, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(args.esm_checkpoint)
    collate_fn = make_ablation_collate_fn(tokenizer)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ---- Collect predictions ----
    print(f"  Running inference with {n_members} ensemble members...")
    all_logits, all_probs, _ = collect_member_predictions(models, loader, device)

    # ---- Ensemble prediction ----
    prob_mean = ensemble_probability_averaging(all_probs)
    prob_std = all_probs.std(axis=0)
    predictions = (prob_mean > args.threshold).astype(int)

    # ---- Build output DataFrame ----
    out_df = df[["tcr_id", "tra_seq", "trb_seq", "peptide", "mhc_seq"]].copy()
    out_df["prob_mean"] = np.round(prob_mean, 6)
    out_df["prob_std"] = np.round(prob_std, 6)
    out_df["prediction"] = predictions
    out_df["confidence"] = np.where(
        prob_std < np.median(prob_std), "high", "low"
    )
    out_df["n_members_agree"] = (all_probs > args.threshold).sum(axis=0)

    # Add per-member probabilities
    for i in range(n_members):
        out_df[f"member_{i}_prob"] = np.round(all_probs[i], 6)

    # ---- Save ----
    output_path = Path(args.output_csv) if args.output_csv else out_dir / "predictions.csv"
    out_df.to_csv(output_path, index=False)

    # ---- Summary ----
    n_bind = predictions.sum()
    n_total = len(predictions)
    print(f"\n{'='*65}")
    print(f"  ENSEMBLE PREDICTIONS")
    print(f"  Samples:         {n_total}")
    print(f"  Predicted bind:  {n_bind} ({100*n_bind/n_total:.1f}%)")
    print(f"  High confidence: {(out_df['confidence'] == 'high').sum()}")
    print(f"  Mean prob:       {prob_mean.mean():.4f} ± {prob_std.mean():.4f}")
    print(f"  Saved to:        {output_path}")
    print(f"{'='*65}")


# ============================================================
# 9. Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble TCR-pMHC Binding Predictor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- Shared arguments ----
    def add_common_args(p):
        p.add_argument("--esm_checkpoint", type=str,
                        default=os.environ.get("model_checkpoint", "facebook/esm2_t12_35M_UR50D"))
        p.add_argument("--d_model", type=int, default=128)
        p.add_argument("--n_cross_heads", type=int, default=4)
        p.add_argument("--n_cross_layers", type=int, default=2)
        p.add_argument("--d_rosetta", type=int, default=64)
        p.add_argument("--d_fused", type=int, default=256)
        p.add_argument("--dropout", type=float, default=0.2)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--ckpt_dir", type=str, default="outputs/ensemble",
                        help="Directory for checkpoints and results")

    # ---- Train ----
    p_train = subparsers.add_parser("train", help="Train ensemble members")
    add_common_args(p_train)
    p_train.add_argument("--data_csv", type=str, required=True)
    p_train.add_argument("--n_ensemble", type=int, default=5,
                         help="Number of ensemble members (default: 5)")
    p_train.add_argument("--seeds", nargs="+", type=int,
                         default=[42, 123, 456, 789, 2024])
    p_train.add_argument("--peptide_split", action="store_true", default=False)
    p_train.add_argument("--val_frac", type=float, default=0.15)
    p_train.add_argument("--test_frac", type=float, default=0.15)
    p_train.add_argument("--crystal_weight", type=float, default=5.0)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--esm_lr", type=float, default=1e-5)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--warmup_epochs", type=int, default=3)
    p_train.add_argument("--patience", type=int, default=7)
    p_train.add_argument("--unfreeze_b", type=int, default=10)
    p_train.add_argument("--unfreeze_c", type=int, default=20)
    p_train.add_argument("--progress_bar", action="store_true", default=False)

    # ---- Evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Evaluate ensemble on test set")
    add_common_args(p_eval)
    p_eval.add_argument("--data_csv", type=str, required=True)
    p_eval.add_argument("--peptide_split", action="store_true", default=False)
    p_eval.add_argument("--val_frac", type=float, default=0.15)
    p_eval.add_argument("--test_frac", type=float, default=0.15)
    p_eval.add_argument("--seeds", nargs="+", type=int,
                         default=[42, 123, 456, 789, 2024])

    # ---- Predict ----
    p_pred = subparsers.add_parser("predict", help="Predict on new data")
    add_common_args(p_pred)
    p_pred.add_argument("--input_csv", type=str, required=True,
                        help="CSV with tra_seq, trb_seq, peptide, mhc_seq columns")
    p_pred.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: ckpt_dir/predictions.csv)")
    p_pred.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()