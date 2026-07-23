"""
pretrain_rosetta.py — Multi-task Pretraining for STAG-LLM (Model B)
====================================================================

Joint pretraining with two objectives:
  1. Rosetta interface energy regression (binders only — continuous ΔG)
  2. Binary binding classification (binders + non-binders)

This exposes the EGNN to both binding and non-binding geometries while
leveraging the rich continuous Rosetta signal from binders.

Input CSV format (single file):
  tcr_id,binding_label,rosetta_score
  PEPTIDE_CDR3A_CDR3B,1,24.196        ← binder with Rosetta score
  PEPTIDE_CDR3A_CDR3B,0,              ← non-binder (rosetta_score is empty/NaN)

All tcr_ids must have corresponding PDB files in --pdb_dir.

Training stages:
  Stage 1: All structures (binders + non-binders), ESM frozen
           Trains: EGNN + projections + cross-attention + both heads
  Stage 2: Crystal structures only (subset), optionally unfreeze ESM
           Refines on high-confidence geometries

Usage:
  python pretrain_rosetta.py \
    --stage 1 \
    --data_csv /path/to/pretrain_combined.csv \
    --pdb_dir /path/to/top_structures/ \
    --epochs 30 --lr 1e-3 --batch_size 16

  python pretrain_rosetta.py \
    --stage 2 \
    --data_csv /path/to/crystal_combined.csv \
    --checkpoint /path/to/stage1_best.ckpt \
    --epochs 20 --lr 1e-4 --esm_lr 1e-5 --n_tune_layers 2
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

import dgl
from transformers import AutoTokenizer, AutoModel
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from dataset import (
    ProteinGraphDataset, make_esm_collate_fn,
    get_graphein_config, enc_dict, _safe_resname, char_to_int, PAD_IDX,
)
from model_llm import ESMMultimodalBindingModel


# ============================================================
# 1. Custom Dataset for Pretraining
#    Handles NaN rosetta_score without dropping rows
# ============================================================

class PretrainGraphDataset(ProteinGraphDataset):
    """
    Extends ProteinGraphDataset for multi-task pretraining.

    Key differences from parent:
      - Does NOT call dropna(how='any') — preserves rows where
        rosetta_score is NaN (non-binders)
      - Returns (binding_label, rosetta_score, has_rosetta) separately
      - 'label' column in the parent is repurposed: we store binding_label
        there for graph caching compatibility, and carry rosetta_score
        as a separate field
    """

    def __init__(self, pdb_dfs, pdb_dir, cache_dir, use_cache=True,
                 graphein_config=None):
        # We override __init__ to avoid the parent's dropna(how='any')
        if graphein_config is None:
            graphein_config = get_graphein_config()

        self.pdb_dfs = pdb_dfs.copy()

        # Only drop rows where REQUIRED columns are NaN (not rosetta_score)
        required = ['tcr_id', 'binding_label']
        self.pdb_dfs = self.pdb_dfs.dropna(subset=required).reset_index(drop=True)

        # Filter to available PDB files
        avail_pdbs = {Path(i).stem for i in glob(os.path.join(pdb_dir, '*.pdb'))}
        self.pdb_dfs = self.pdb_dfs[
            self.pdb_dfs['tcr_id'].isin(avail_pdbs)
        ].reset_index(drop=True)

        # Compat: parent expects 'label' column for caching
        # Store binding_label as 'label' for the parent's __getitem__
        self.pdb_dfs['label'] = self.pdb_dfs['binding_label'].astype(int)

        drop_cols = {
            'tcr_id', 'CDR3a', 'CDR3b', 'MHC_sequence', 'peptide',
            'TCR_A_sequence', 'TCR_B_sequence',
            'label', 'binding_label', 'rosetta_score',
        }
        self.feature_cols = [c for c in self.pdb_dfs.columns if c not in drop_cols]

        self.pdb_dir = pdb_dir
        self.graphein_config = graphein_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        self._mask = enc_dict["MASK"]
        self._enc_dict = enc_dict
        self._cache_hits = 0
        self._cache_misses = 0

        # Store rosetta scores separately (may contain NaN)
        self._rosetta_scores = self.pdb_dfs['rosetta_score'].values.astype(np.float32)
        self._binding_labels = self.pdb_dfs['binding_label'].values.astype(np.int64)

        n_binders = (self._binding_labels == 1).sum()
        n_nonbinders = (self._binding_labels == 0).sum()
        n_has_rosetta = (~np.isnan(self._rosetta_scores)).sum()
        print(f"  PretrainGraphDataset: {len(self.pdb_dfs)} samples "
              f"({n_binders} binders, {n_nonbinders} non-binders, "
              f"{n_has_rosetta} with Rosetta scores)")

    def __getitem__(self, idx):
        # Get the base tuple from parent (uses 'label' = binding_label)
        base = super().__getitem__(idx)
        # base = (tcr_id, label_tensor, pmhc_seq, tcr_seq, full_seq,
        #         graph, struct_features, pmhc_str, tcr_str)

        binding_label = self._binding_labels[idx]
        rosetta_score = self._rosetta_scores[idx]
        has_rosetta = not np.isnan(rosetta_score)

        # Replace NaN with 0.0 for tensor compatibility (masked out in loss)
        rosetta_val = rosetta_score if has_rosetta else 0.0

        return (
            *base,  # 9 elements from parent
            torch.tensor(binding_label, dtype=torch.long),
            torch.tensor(rosetta_val, dtype=torch.float32),
            torch.tensor(has_rosetta, dtype=torch.bool),
        )


def make_pretrain_esm_collate_fn(esm_tokenizer):
    """
    Collate function for pretraining that handles the extra fields
    (binding_label, rosetta_score, has_rosetta) beyond the base ESM collate.
    """

    def _tokenize_and_pad(sequences):
        clean_seqs = []
        for s in sequences:
            s = s.replace("J", "").replace("|", "")
            s = " ".join(list(s.strip()))
            clean_seqs.append(s)
        tok_out = esm_tokenizer(
            clean_seqs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        return tok_out["input_ids"], tok_out["attention_mask"].bool()

    def collate_fn(batch):
        # Unpack: 9 base fields + 3 pretrain fields
        (tcr_ids, _labels, pmhc_seq, tcr_seq, full_seq,
         graphs, struct_features, pmhc_str, tcr_str,
         binding_labels, rosetta_scores, has_rosetta) = zip(*batch)

        tcr_ids = list(tcr_ids)
        binding_labels = torch.stack(binding_labels)
        rosetta_scores = torch.stack(rosetta_scores)
        has_rosetta = torch.stack(has_rosetta)
        full_seq = torch.stack(full_seq)

        pmhc_esm_ids, pmhc_mask = _tokenize_and_pad(pmhc_str)
        tcr_esm_ids, tcr_mask = _tokenize_and_pad(tcr_str)

        batched_graph = dgl.batch(graphs)

        return (
            tcr_ids,           # List[str]
            binding_labels,    # [B] long: 0 or 1
            rosetta_scores,    # [B] float: Rosetta ΔG (0.0 if no score)
            has_rosetta,       # [B] bool: True if rosetta_score is valid
            pmhc_esm_ids,      # [B, L_mhc] ESM token IDs
            tcr_esm_ids,       # [B, L_tcr] ESM token IDs
            pmhc_mask,         # [B, L_mhc] bool
            tcr_mask,          # [B, L_tcr] bool
            batched_graph,     # DGL batched graph
            full_seq,          # [B, L_full] int (kept for compat)
        )

    return collate_fn


# ============================================================
# 2. Pretraining Model Wrapper
#    Adds regression head alongside the existing classifier
# ============================================================

class RosettaPretrainModel(nn.Module):
    """
    Wraps ESMMultimodalBindingModel for multi-task pretraining:
      - Task A: Rosetta interface energy regression (binders only)
      - Task B: Binary binding classification (all samples)

    The base model's classifier head IS used for Task B (so it also
    gets pretrained — unlike the previous version where it was discarded).
    A separate regression head is added for Task A and discarded after
    pretraining.
    """

    def __init__(self, base_model: ESMMultimodalBindingModel, d_fused: int = 256):
        super().__init__()
        self.base = base_model

        # Regression head for Rosetta energy (discarded after pretraining)
        self.regressor = nn.Sequential(
            nn.Linear(d_fused, d_fused),
            nn.LayerNorm(d_fused),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_fused, d_fused // 2),
            nn.LayerNorm(d_fused // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_fused // 2, 1),
        )

    def forward(
        self,
        tcr_ids, mhc_ids, tcr_mask, mhc_mask,
        struct_graph=None, struct_available=None,
        labels=None, compute_loss=False,
    ):
        # Run the full base model (including its classifier)
        out = self.base(
            tcr_ids, mhc_ids, tcr_mask, mhc_mask,
            struct_graph=struct_graph,
            struct_available=struct_available,
            labels=labels,
            compute_loss=compute_loss,
        )

        # Rosetta regression from fused features
        fused = out["fused"]
        out["energy_pred"] = self.regressor(fused).squeeze(-1)

        return out


# ============================================================
# 3. Lightning Module
# ============================================================

class RosettaPretrainLightning(L.LightningModule):
    """
    Multi-task pretraining:
      L = λ_bind * L_bind + λ_rosetta * L_rosetta

    L_bind:    BCE on all samples (binders + non-binders)
    L_rosetta: Huber on binders only (masked by has_rosetta)

    Huber loss is robust to outliers from noisy AlphaFold structures.
    Targets are z-score normalized for regression stability.
    """

    def __init__(
        self,
        model: RosettaPretrainModel,
        lr: float = 1e-3,
        esm_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        max_epochs: int = 30,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        lambda_bind: float = 1.0,
        lambda_rosetta: float = 1.0,
        pos_weight: float = 5.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))

        # Losses
        self.huber = nn.HuberLoss(delta=1.0, reduction='none')
        self.register_buffer("bce_pos_weight", torch.tensor([pos_weight]))

        # ---- Metrics ----
        # Regression metrics (on binders with Rosetta scores)
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.test_mse = MeanSquaredError()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

        # Classification metrics (on all samples)
        self.train_auroc = BinaryAUROC()
        self.train_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()

    def _compute_loss(self, out, binding_labels, rosetta_scores, has_rosetta):
        """Compute multi-task loss."""
        device = binding_labels.device
        B = binding_labels.size(0)

        # --- Task B: Binary binding classification (all samples) ---
        logit = out["logit"].view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            logit, binding_labels.float(),
            pos_weight=self.bce_pos_weight, reduction="mean",
        )

        # --- Task A: Rosetta regression (binders with scores only) ---
        rosetta_loss = torch.tensor(0.0, device=device)
        n_rosetta = has_rosetta.sum().item()

        if n_rosetta > 0:
            targets_norm = (
                (rosetta_scores - self.target_mean) /
                self.target_std.clamp(min=1e-8)
            )
            energy_pred = out["energy_pred"]
            per_sample_loss = self.huber(energy_pred, targets_norm)
            # Mask: only compute on samples with Rosetta scores
            rosetta_loss = (per_sample_loss * has_rosetta.float()).sum() / n_rosetta

        total = (
            self.hparams.lambda_bind * bind_loss +
            self.hparams.lambda_rosetta * rosetta_loss
        )

        return total, bind_loss, rosetta_loss

    def _forward_batch(self, batch):
        """Unpack batch and run forward."""
        (tcr_ids_list, binding_labels, rosetta_scores, has_rosetta,
         pmhc_esm_ids, tcr_esm_ids, pmhc_mask, tcr_mask,
         batched_graph, full_seq) = batch

        B = binding_labels.size(0)
        struct_available = torch.ones(B, dtype=torch.bool, device=self.device)

        out = self.model(
            tcr_ids=tcr_esm_ids.to(self.device),
            mhc_ids=pmhc_esm_ids.to(self.device),
            tcr_mask=tcr_mask.to(self.device),
            mhc_mask=pmhc_mask.to(self.device),
            struct_graph=batched_graph.to(self.device),
            struct_available=struct_available,
            compute_loss=False,
        )

        binding_labels = binding_labels.to(self.device)
        rosetta_scores = rosetta_scores.to(self.device)
        has_rosetta = has_rosetta.to(self.device)

        return out, binding_labels, rosetta_scores, has_rosetta

    # ---- Training ----
    def training_step(self, batch, batch_idx):
        out, binding_labels, rosetta_scores, has_rosetta = self._forward_batch(batch)
        total, bind_loss, rosetta_loss = self._compute_loss(
            out, binding_labels, rosetta_scores, has_rosetta
        )

        B = binding_labels.size(0)
        self.log("train_loss", total, prog_bar=True, batch_size=B)
        self.log("train_bind_loss", bind_loss, batch_size=B)
        self.log("train_rosetta_loss", rosetta_loss, batch_size=B)

        # Classification metrics
        probs = out["prob"].squeeze(-1)
        self.train_auroc.update(probs, binding_labels)
        self.train_f1.update(probs, binding_labels)

        # Regression metrics (binders only)
        if has_rosetta.any():
            preds_denorm = (
                out["energy_pred"][has_rosetta] * self.target_std + self.target_mean
            )
            targets = rosetta_scores[has_rosetta]
            self.train_mse.update(preds_denorm, targets)

        return total

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        if self.train_mse._update_count > 0:
            self.log("train_mse", self.train_mse.compute())
        self.train_auroc.reset()
        self.train_f1.reset()
        self.train_mse.reset()

    # ---- Validation ----
    def validation_step(self, batch, batch_idx):
        out, binding_labels, rosetta_scores, has_rosetta = self._forward_batch(batch)
        total, bind_loss, rosetta_loss = self._compute_loss(
            out, binding_labels, rosetta_scores, has_rosetta
        )

        B = binding_labels.size(0)
        self.log("val_loss", total, prog_bar=True, batch_size=B)
        self.log("val_bind_loss", bind_loss, batch_size=B)
        self.log("val_rosetta_loss", rosetta_loss, batch_size=B)

        probs = out["prob"].squeeze(-1)
        self.val_auroc.update(probs, binding_labels)
        self.val_f1.update(probs, binding_labels)

        if has_rosetta.any():
            preds_denorm = (
                out["energy_pred"][has_rosetta] * self.target_std + self.target_mean
            )
            targets = rosetta_scores[has_rosetta]
            self.val_mse.update(preds_denorm, targets)
            self.val_pearson.update(preds_denorm, targets)
            self.val_spearman.update(preds_denorm, targets)

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        if self.val_mse._update_count > 0:
            self.log("val_mse", self.val_mse.compute())
            self.log("val_pearson", self.val_pearson.compute(), prog_bar=True)
            self.log("val_spearman", self.val_spearman.compute())
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_mse.reset()
        self.val_pearson.reset()
        self.val_spearman.reset()

    # ---- Test ----
    def test_step(self, batch, batch_idx):
        out, binding_labels, rosetta_scores, has_rosetta = self._forward_batch(batch)
        total, bind_loss, rosetta_loss = self._compute_loss(
            out, binding_labels, rosetta_scores, has_rosetta
        )

        B = binding_labels.size(0)
        self.log("test_loss", total, batch_size=B)
        self.log("test_bind_loss", bind_loss, batch_size=B)
        self.log("test_rosetta_loss", rosetta_loss, batch_size=B)

        probs = out["prob"].squeeze(-1)
        self.test_auroc.update(probs, binding_labels)
        self.test_f1.update(probs, binding_labels)

        if has_rosetta.any():
            preds_denorm = (
                out["energy_pred"][has_rosetta] * self.target_std + self.target_mean
            )
            targets = rosetta_scores[has_rosetta]
            self.test_mse.update(preds_denorm, targets)
            self.test_pearson.update(preds_denorm, targets)
            self.test_spearman.update(preds_denorm, targets)

    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute())
        self.log("test_f1", self.test_f1.compute())
        if self.test_mse._update_count > 0:
            self.log("test_mse", self.test_mse.compute())
            self.log("test_pearson", self.test_pearson.compute())
            self.log("test_spearman", self.test_spearman.compute())
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_mse.reset()
        self.test_pearson.reset()
        self.test_spearman.reset()

    # ---- Optimizer ----
    def configure_optimizers(self):
        esm_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "esm_model" in name:
                esm_params.append(param)
            else:
                other_params.append(param)

        param_groups = [{"params": other_params, "lr": self.hparams.lr}]
        if esm_params:
            param_groups.append({"params": esm_params, "lr": self.hparams.esm_lr})

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_epochs
        total = self.hparams.max_epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-task Rosetta + Binding Pretraining for STAG-LLM"
    )

    # Data
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV: tcr_id, binding_label (0/1), rosetta_score (float or NaN)")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Separate validation CSV (same format)")
    parser.add_argument("--pdb_dir", type=str,
                        default="${SPECTRA_ROOT}/data/STAG-LLM/data/top_structures/")
    parser.add_argument("--cache_dir", type=str,
                        default="${SPECTRA_ROOT}/data/STAG-LLM/pretrain_cache")

    # Model
    parser.add_argument("--esm_checkpoint", type=str,
                        default=os.environ.get("model_checkpoint",
                                               "facebook/esm2_t12_35M_UR50D"))
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--egnn_hidden", type=int, default=128)
    parser.add_argument("--egnn_layers", type=int, default=5)
    parser.add_argument("--egnn_out_dim", type=int, default=128)
    parser.add_argument("--n_cross_heads", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=8)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--esm_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--n_tune_layers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda_bind", type=float, default=1.0,
                        help="Weight for binary binding loss")
    parser.add_argument("--lambda_rosetta", type=float, default=1.0,
                        help="Weight for Rosetta regression loss")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint for stage 2")
    parser.add_argument("--out_dir", type=str,
                        default="${SPECTRA_ROOT}/project/TCRpMHC/outputs/STAG-LLM/pretrain")

    args = parser.parse_args()
    L.seed_everything(args.seed)

    out_dir = Path(args.out_dir) / f"stage{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # ---- Load & validate data ----
    df = pd.read_csv(args.data_csv)
    assert "tcr_id" in df.columns, "CSV must have 'tcr_id' column"
    assert "binding_label" in df.columns, "CSV must have 'binding_label' column (0/1)"

    if "rosetta_score" not in df.columns:
        df["rosetta_score"] = np.nan

    n_binders = (df["binding_label"] == 1).sum()
    n_nonbinders = (df["binding_label"] == 0).sum()
    n_rosetta = df["rosetta_score"].notna().sum()

    # Train / val split (stratified by binding_label)
    if args.val_csv is not None:
        df_train = df
        df_val = pd.read_csv(args.val_csv)
        if "rosetta_score" not in df_val.columns:
            df_val["rosetta_score"] = np.nan
    else:
        df_train, df_val = train_test_split(
            df, test_size=args.val_split, random_state=args.seed,
            stratify=df["binding_label"],
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

    # Rosetta normalization stats from TRAINING binders only
    train_rosetta = df_train.loc[
        df_train["rosetta_score"].notna(), "rosetta_score"
    ]
    target_mean = float(train_rosetta.mean()) if len(train_rosetta) > 0 else 0.0
    target_std = float(train_rosetta.std()) if len(train_rosetta) > 1 else 1.0

    # pos_weight for class imbalance
    n_pos_train = (df_train["binding_label"] == 1).sum()
    n_neg_train = (df_train["binding_label"] == 0).sum()
    pos_weight = n_neg_train / max(n_pos_train, 1)
    pos_weight = min(pos_weight, 10.0)  # Cap for stability

    print(f"\n{'='*60}")
    print(f"  Multi-task Pretraining — Stage {args.stage}")
    print(f"  Total: {len(df)} ({n_binders} binders, {n_nonbinders} non-binders)")
    print(f"  Rosetta scores available: {n_rosetta}")
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")
    print(f"  Rosetta target mean: {target_mean:.2f} | std: {target_std:.2f}")
    print(f"  BCE pos_weight: {pos_weight:.2f}")
    print(f"  Loss weights: λ_bind={args.lambda_bind}, λ_rosetta={args.lambda_rosetta}")
    print(f"  ESM: {args.esm_checkpoint} | tune_layers={args.n_tune_layers}")
    print(f"{'='*60}\n")

    # ---- Datasets ----
    train_ds = PretrainGraphDataset(
        pdb_dfs=df_train, pdb_dir=args.pdb_dir,
        cache_dir=args.cache_dir, use_cache=True,
    )
    val_ds = PretrainGraphDataset(
        pdb_dfs=df_val, pdb_dir=args.pdb_dir,
        cache_dir=args.cache_dir, use_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.esm_checkpoint)
    collate_fn = make_pretrain_esm_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    # ---- Build model ----
    freeze_esm = (args.n_tune_layers == 0)
    esm_model = AutoModel.from_pretrained(args.esm_checkpoint)

    base_model = ESMMultimodalBindingModel(
        esm_model=esm_model,
        esm_hidden_size=esm_model.config.hidden_size,
        freeze_esm=freeze_esm,
        n_tune_layers=args.n_tune_layers,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        d_model=args.d_model,
        n_cross_heads=args.n_cross_heads,
        node_feat_size=20,
        edge_feat_size=7,
        egnn_hidden=args.egnn_hidden,
        egnn_layers=args.egnn_layers,
        egnn_out_dim=args.egnn_out_dim,
        struct_seq_cross_heads=4,
        d_fused=args.d_model,
        clf_hidden=args.d_model,
        pos_weight=pos_weight,
    )

    pretrain_model = RosettaPretrainModel(base_model, d_fused=args.d_model)

    # ---- Load stage 1 checkpoint for stage 2 ----
    if args.stage == 2 and args.checkpoint is not None:
        print(f"Loading stage 1 checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned[new_key] = v
        missing, unexpected = pretrain_model.load_state_dict(cleaned, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (sample): {missing[:5]}")

    # ---- Lightning module ----
    lit_model = RosettaPretrainLightning(
        model=pretrain_model,
        lr=args.lr,
        esm_lr=args.esm_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        target_mean=target_mean,
        target_std=target_std,
        lambda_bind=args.lambda_bind,
        lambda_rosetta=args.lambda_rosetta,
        pos_weight=pos_weight,
    )

    # Print param counts
    total_params = sum(p.numel() for p in pretrain_model.parameters())
    trainable_params = sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.1f}%)\n")

    # ---- Callbacks ----
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        monitor="val_auroc",
        mode="max",
        save_top_k=2,
        filename="best-{epoch:02d}-{val_auroc:.3f}-{val_pearson:.3f}",
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_auroc", patience=args.patience, mode="max"),
        ],
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        gradient_clip_val=1.0,
        enable_checkpointing=True,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")

    # ---- Test on val set ----
    test_res = trainer.test(lit_model, dataloaders=val_loader, ckpt_path=best_ckpt)

    # ---- Save summary ----
    summary = {
        "stage": args.stage,
        "data_csv": args.data_csv,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_binders": int(n_binders),
        "n_nonbinders": int(n_nonbinders),
        "n_rosetta": int(n_rosetta),
        "target_mean": target_mean,
        "target_std": target_std,
        "pos_weight": float(pos_weight),
        "lambda_bind": args.lambda_bind,
        "lambda_rosetta": args.lambda_rosetta,
        "best_checkpoint": str(best_ckpt),
        "esm_checkpoint": args.esm_checkpoint,
        "n_tune_layers": args.n_tune_layers,
        "use_lora": args.use_lora,
        "lr": args.lr,
        "esm_lr": args.esm_lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "results": test_res[0] if test_res else {},
    }

    json_path = out_dir / f"pretrain_stage{args.stage}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")

    # ---- Save base model weights for downstream fine-tuning ----
    # Saves the ESMMultimodalBindingModel state_dict (without regressor)
    # so the fine-tuning script can load it directly
    base_state = {
        k.replace("base.", ""): v
        for k, v in pretrain_model.state_dict().items()
        if k.startswith("base.")
    }
    base_weights_path = out_dir / f"base_model_stage{args.stage}.pt"
    torch.save({
        "model_state_dict": base_state,
        "target_mean": target_mean,
        "target_std": target_std,
        "pos_weight": float(pos_weight),
        "args": vars(args),
    }, base_weights_path)
    print(f"Base model weights saved to: {base_weights_path}")


if __name__ == "__main__":
    main()
