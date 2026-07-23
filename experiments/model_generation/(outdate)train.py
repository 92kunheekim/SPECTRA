"""
Lightning Training Script — Single Fold
=========================================

Usage:
    python train.py --csv /path/to/dataset.csv \
                    --pdb_dir /path/to/pdb_structures/ \
                    --folds_json ./cv_folds.json \
                    --fold 0 \
                    --output_dir ./experiments

    # Run all 15 folds sequentially:
    for i in $(seq 0 14); do
        python train.py --csv data.csv --pdb_dir pdbs/ --folds_json cv_folds.json --fold $i
    done

Dependencies:
    pip install pytorch-lightning torchmetrics python-Levenshtein scikit-learn
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)

from dataset import ProteinGraphDataset, custom_collate_fn, PAD_IDX, AA_VOCAB
from model import MultimodalBindingModel


# ============================================================
# 1. Lightning DataModule
# ============================================================

class TCRpMHCDataModule(L.LightningDataModule):
    """
    Wraps ProteinGraphDataset with fold-based train/val/test splits.
    """

    def __init__(
        self,
        csv_path: str,
        pdb_dir: str,
        fold_info: Dict,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.fold_info = fold_info
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        
        required = {"CDR3a", "CDR3b", "MHC_sequence", "peptide",
                     "TCR_A_sequence", "TCR_B_sequence", "label"}
        df = df.dropna(subset=list(required)).reset_index(drop=True)

        kwargs = dict(pdb_dfs=df, pdb_dir=self.pdb_dir)
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self.dataset = ProteinGraphDataset(**kwargs)

        self.train_idx = list(set(self.fold_info["train"]) & set(self.dataset.pdb_dfs.index))
        self.val_idx = list(set(self.fold_info["val"]) & set(self.dataset.pdb_dfs.index))
        self.test_idx = list(set(self.fold_info["test"]) & set(self.dataset.pdb_dfs.index))

        # Compute class weights for pos_weight
        train_labels = self.dataset.pdb_dfs.loc[self.train_idx, "label"]
        n_pos = (train_labels == 1).sum()
        n_neg = (train_labels == 0).sum()
        self.pos_weight = float(n_neg) / max(float(n_pos), 1.0)

    def train_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ============================================================
# 2. Lightning Module
# ============================================================

class BindingLitModule(L.LightningModule):
    """
    Lightning wrapper around MultimodalBindingModel.
    Handles training/validation/test steps, metrics, and optimiser config.
    """

    def __init__(
        self,
        # Model architecture
        vocab_size: int = len(AA_VOCAB),
        pad_id: int = PAD_IDX,
        d_model: int = 256,
        latent_dim: int = 64,
        n_enc_layers: int = 2,
        n_cross_heads: int = 4,
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        egnn_hidden: int = 128,
        egnn_layers: int = 5,
        egnn_out_dim: int = 128,
        struct_seq_cross_heads: int = 4,
        d_fused: int = 256,
        clf_hidden: int = 128,
        dropout: float = 0.1,
        kl_anneal_steps: int = 5000,
        # Loss weights
        lambda_bind: float = 1.0,
        lambda_recon: float = 0.3,
        lambda_kl: float = 0.2,
        pos_weight: float = 1.0,
        # Optimiser
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        max_steps: int = 50000,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultimodalBindingModel(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=d_model,
            latent_dim=latent_dim,
            n_enc_layers=n_enc_layers,
            n_cross_heads=n_cross_heads,
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            egnn_hidden=egnn_hidden,
            egnn_layers=egnn_layers,
            egnn_out_dim=egnn_out_dim,
            struct_seq_cross_heads=struct_seq_cross_heads,
            d_fused=d_fused,
            clf_hidden=clf_hidden,
            dropout=dropout,
            kl_anneal_steps=kl_anneal_steps,
            lambda_bind=lambda_bind,
            lambda_recon=lambda_recon,
            lambda_kl=lambda_kl,
            pos_weight=pos_weight,
        )

        # ---- Metrics ----
        # Separate metric objects for each stage to avoid cross-contamination
        metrics = lambda: torch.nn.ModuleDict({
            "auroc": BinaryAUROC(),
            "auprc": BinaryAveragePrecision(),
            "f1": BinaryF1Score(),
            "acc": BinaryAccuracy(),
            "prec": BinaryPrecision(),
            "rec": BinaryRecall(),
        })
        self.train_metrics = metrics()
        self.val_metrics = metrics()
        self.test_metrics = metrics()

        # For collecting test predictions
        self.test_probs = []
        self.test_labels = []

    # ---- Shared step ----
    def _shared_step(self, batch):
        tcr_ids, labels, pmhc_seq, tcr_seq, full_seq, bg, struct_features = batch

        labels_float = labels.float()
        pmhc_mask = (pmhc_seq != self.hparams.pad_id)
        tcr_mask = (tcr_seq != self.hparams.pad_id)
        B = labels.shape[0]
        struct_available = torch.ones(B, dtype=torch.bool, device=self.device)

        out = self.model(
            tcr_ids=tcr_seq,
            mhc_ids=pmhc_seq,
            tcr_mask=tcr_mask,
            mhc_mask=pmhc_mask,
            struct_graph=bg,
            struct_available=struct_available,
            labels=labels_float,
            compute_loss=True,
        )

        probs = out["prob"].squeeze(-1)   # [B]
        return out, probs, labels_float

    # ---- Training ----
    def training_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)
        # Log losses
        self.log("train/loss", out["loss"], batch_size = batch_size, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/bind_loss", out["bind_loss"], batch_size = batch_size, on_step=False, on_epoch=True)
        self.log("train/recon_tcr", out["recon_tcr"], batch_size = batch_size, on_step=False, on_epoch=True)
        self.log("train/recon_mhc", out["recon_mhc"], batch_size = batch_size, on_step=False, on_epoch=True)
        self.log("train/kl_tcr", out["kl_tcr"], batch_size = batch_size, on_step=False, on_epoch=True)
        self.log("train/kl_mhc", out["kl_mhc"], batch_size = batch_size, on_step=False, on_epoch=True)
        self.log("train/beta", out["beta"], batch_size = batch_size, on_step=True, on_epoch=False)

        # Metrics
        labels_int = labels.long()
        for name, metric in self.train_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"train/{name}", metric, on_step=False, on_epoch=True)

        return out["loss"]

    # ---- Validation ----
    def validation_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)
        self.log("val/loss", out["loss"], batch_size = batch_size, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/bind_loss", out["bind_loss"], batch_size = batch_size, on_step=False, on_epoch=True, sync_dist=True)

        labels_int = labels.long()
        for name, metric in self.val_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"val/{name}", metric, batch_size = batch_size, on_step=False, on_epoch=True, prog_bar=(name == "auroc"), sync_dist=True)

    # ---- Test ----
    def test_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)
        self.log("test/loss", out["loss"], batch_size = batch_size, on_step=False, on_epoch=True, sync_dist=True)

        labels_int = labels.long()
        for name, metric in self.test_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"test/{name}", metric, batch_size = batch_size, on_step=False, on_epoch=True, sync_dist=True)

        # Collect for post-hoc analysis
        self.test_probs.append(probs.cpu())
        self.test_labels.append(labels_int.cpu())

    def on_test_epoch_end(self):
        all_probs = torch.cat(self.test_probs)
        all_labels = torch.cat(self.test_labels)
        # Save predictions to file
        if self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path(".")
            torch.save(
                {"probs": all_probs, "labels": all_labels},
                out_dir / "test_predictions.pt",
            )
        self.test_probs.clear()
        self.test_labels.clear()

    # ---- Optimiser with warmup + cosine decay ----
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_step):
            warmup = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            if current_step < warmup:
                # Linear warmup
                return float(current_step) / float(max(1, warmup))
            # Cosine decay
            progress = float(current_step - warmup) / float(max(1, max_steps - warmup))
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ============================================================
# 3. Main: train a single fold
# ============================================================

def train_fold(args):
    # ---- Load fold info ----
    with open(args.folds_json) as fp:
        all_folds = json.load(fp)

    assert 0 <= args.fold < len(all_folds), \
        f"Fold {args.fold} out of range (have {len(all_folds)} folds)"
    fold_info = all_folds[args.fold]
    fold_idx = fold_info["fold"] if "fold" in fold_info else args.fold

    print(f"\n{'='*60}")
    print(f"  Training fold {fold_idx}")
    print(f"  Train: {len(fold_info['train'])} | Val: {len(fold_info['val'])} | Test: {len(fold_info['test'])}")
    print(f"{'='*60}\n")

    # ---- DataModule ----
    dm = TCRpMHCDataModule(
        csv_path=args.csv,
        pdb_dir=args.pdb_dir,
        fold_info=fold_info,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    dm.setup()

    # ---- Lightning Module ----
    model = BindingLitModule(
        # Architecture
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        n_enc_layers=args.n_enc_layers,
        n_cross_heads=args.n_cross_heads,
        node_feat_size=args.node_feat_size,
        edge_feat_size=args.edge_feat_size,
        egnn_hidden=args.egnn_hidden,
        egnn_layers=args.egnn_layers,
        egnn_out_dim=args.egnn_out_dim,
        struct_seq_cross_heads=args.struct_seq_cross_heads,
        d_fused=args.d_fused,
        clf_hidden=args.clf_hidden,
        dropout=args.dropout,
        kl_anneal_steps=args.kl_anneal_steps,
        # Loss
        lambda_bind=args.lambda_bind,
        lambda_recon=args.lambda_recon,
        lambda_kl=args.lambda_kl,
        pos_weight=dm.pos_weight,
        # Optimiser
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_epochs * (len(fold_info["train"]) // args.batch_size + 1),
        grad_clip=args.grad_clip,
    )

    # ---- Callbacks ----
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(fold_dir, "checkpoints"),
            filename="best-{epoch:03d}-{val/auprc:.4f}-{val/f1:.4f}",
            monitor="val/auprc",
            mode="max",
            save_top_k=2,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/auprc",
            mode="max",
            patience=args.patience,
            min_delta=1e-4,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # ---- Loggers ----
    loggers = [
        CSVLogger(save_dir=fold_dir, name="csv_logs"),
        TensorBoardLogger(save_dir=fold_dir, name="tb_logs"),
    ]

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=500,
        val_check_interval=args.val_check_interval,
        deterministic=False,
        enable_model_summary=True,
        # limit_train_batches=10, limit_val_batches=5, limit_test_batches=5,
    )

    # ---- Train ----
    trainer.fit(model, datamodule=dm)

    # ---- Test with best checkpoint ----
    best_ckpt = callbacks[0].best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    test_results = trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)

    # ---- Save test results ----
    results = {
        "fold": fold_idx,
        "best_checkpoint": best_ckpt,
        "best_val_auroc": float(callbacks[0].best_model_score or 0),
        "test_results": test_results[0] if test_results else {},
        "hparams": dict(model.hparams),
    }
    results_path = os.path.join(fold_dir, "results.json")
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results


# ============================================================
# 4. CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train multimodal TCR-pMHC binding model (single fold, Lightning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--csv", required=True, help="Path to dataset CSV")
    p.add_argument("--pdb_dir", required=True, help="Directory with .pdb structure files")
    p.add_argument("--folds_json", required=True, help="Path to cv_folds.json from data_split.py")
    p.add_argument("--fold", type=int, required=True, help="Fold index to train (0-indexed)")
    p.add_argument("--output_dir", default="./experiments", help="Output root directory")
    p.add_argument("--cache_dir", default=None, help="Graph cache directory (optional)")

    # Training
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--val_check_interval", type=float, default=1.0,
                    help="Check val every N epochs (float) or steps (int)")
    p.add_argument("--precision", default="32", help="16-mixed, bf16-mixed, or 32")
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--n_enc_layers", type=int, default=2)
    p.add_argument("--n_cross_heads", type=int, default=4)
    p.add_argument("--node_feat_size", type=int, default=20)
    p.add_argument("--edge_feat_size", type=int, default=7)
    p.add_argument("--egnn_hidden", type=int, default=128)
    p.add_argument("--egnn_layers", type=int, default=5)
    p.add_argument("--egnn_out_dim", type=int, default=128)
    p.add_argument("--struct_seq_cross_heads", type=int, default=4)
    p.add_argument("--d_fused", type=int, default=256)
    p.add_argument("--clf_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl_anneal_steps", type=int, default=5000)

    # Loss weights
    p.add_argument("--lambda_bind", type=float, default=1.0)
    p.add_argument("--lambda_recon", type=float, default=0.3)
    p.add_argument("--lambda_kl", type=float, default=0.2)

    # Seed
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_fold(args)