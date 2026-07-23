"""
Lightning Training Script — ESM Multimodal Binding Model (Model B)
===================================================================

Usage:
    python train_esm.py --csv /path/to/dataset.csv \
                        --pdb_dir /path/to/pdb_structures/ \
                        --folds_json ./cv_folds.json \
                        --fold 0 \
                        --esm_checkpoint facebook/esm2_t12_35M_UR50D \
                        --output_dir ./experiments_esm

    # With phased training (freeze ESM first, then fine-tune last 4 layers):
    python train_esm.py ... --freeze_esm --phase2_unfreeze_layers 4 --phase2_epochs 10

Dependencies:
    pip install pytorch-lightning torchmetrics transformers
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
from transformers import AutoTokenizer, EsmForMaskedLM

from dataset_llm import ProteinGraphDataset, custom_collate_fn, make_esm_collate_fn, PAD_IDX
from model_llm import ESMMultimodalBindingModel


# ============================================================
# 1. Lightning DataModule (same as Model A)
# ============================================================

class TCRpMHCDataModule(L.LightningDataModule):
    """Wraps ProteinGraphDataset with fold-based train/val/test splits.
    Uses ESM tokenizer for sequence collation."""

    def __init__(
        self,
        csv_path: str,
        pdb_dir: str,
        fold_info: Dict,
        esm_checkpoint: str = "facebook/esm2_t12_35M_UR50D",
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

        # Load ESM tokenizer and build collate function
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            esm_checkpoint, local_files_only=os.path.isdir(esm_checkpoint)
        )
        self.collate_fn = make_esm_collate_fn(self.esm_tokenizer)

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
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.test_idx),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ============================================================
# 2. Lightning Module for ESM Model
# ============================================================

class ESMBindingLitModule(L.LightningModule):
    """
    Lightning wrapper around ESMMultimodalBindingModel.

    Key differences from Model A's BindingLitModule:
      - No VAE losses (no recon, no KL, no beta annealing)
      - ESM loaded externally and passed in
      - Separate param groups: ESM params get lower LR
      - Supports phased training via set_esm_tuning()
    """

    def __init__(
        self,
        # ESM config
        esm_checkpoint: str = "facebook/esm2_t12_35M_UR50D",
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        # Architecture
        d_model: int = 256,
        n_cross_heads: int = 8,
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        egnn_hidden: int = 128,
        egnn_layers: int = 5,
        egnn_out_dim: int = 128,
        struct_seq_cross_heads: int = 4,
        d_fused: int = 256,
        clf_hidden: int = 256,
        dropout: float = 0.2,
        pos_weight: float = 1.0,
        # Optimiser
        lr: float = 3e-4,
        esm_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        max_steps: int = 50000,
        grad_clip: float = 1.0,
        # Dataset pad_id (for mask computation)
        pad_id: int = PAD_IDX,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Load ESM ----
        esm = EsmForMaskedLM.from_pretrained(
            esm_checkpoint, local_files_only=os.path.isdir(esm_checkpoint)
        )
        esm_hidden_size = esm.config.hidden_size

        # ---- Build model ----
        self.model = ESMMultimodalBindingModel(
            esm_encoder=esm.esm.encoder,
            esm_embedding=esm.get_input_embeddings(),
            esm_hidden_size=esm_hidden_size,
            freeze_esm=freeze_esm,
            n_tune_layers=n_tune_layers,
            d_model=d_model,
            n_cross_heads=n_cross_heads,
            dropout=dropout,
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            egnn_hidden=egnn_hidden,
            egnn_layers=egnn_layers,
            egnn_out_dim=egnn_out_dim,
            struct_seq_cross_heads=struct_seq_cross_heads,
            d_fused=d_fused,
            clf_hidden=clf_hidden,
            pos_weight=pos_weight,
        )

        # Free the HF wrapper — we only keep encoder + embedding inside our model
        del esm

        # ---- Metrics ----
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

        self.test_probs = []
        self.test_labels = []

    # ---- Shared step ----
    def _shared_step(self, batch):
        (tcr_ids, labels, pmhc_esm_ids, tcr_esm_ids, full_seq,
         bg, struct_features, pmhc_mask, tcr_mask) = batch

        labels_float = labels.float()
        B = labels.shape[0]
        struct_available = torch.ones(B, dtype=torch.bool, device=self.device)

        # Move graph to device
        if bg is not None:
            bg = bg.to(self.device)

        out = self.model(
            tcr_ids=tcr_esm_ids,
            mhc_ids=pmhc_esm_ids,
            tcr_mask=tcr_mask,
            mhc_mask=pmhc_mask,
            struct_graph=bg,
            struct_available=struct_available,
            labels=labels_float,
            compute_loss=True,
        )

        probs = out["prob"].squeeze(-1)
        return out, probs, labels_float

    # ---- Training ----
    def training_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)

        self.log("train/loss", out["loss"], batch_size=batch_size,
                 prog_bar=True, on_step=True, on_epoch=True)

        labels_int = labels.long()
        for name, metric in self.train_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"train/{name}", metric, batch_size=batch_size,
                     on_step=False, on_epoch=True)

        return out["loss"]

    # ---- Validation ----
    def validation_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)

        self.log("val/loss", out["loss"], batch_size=batch_size,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        labels_int = labels.long()
        for name, metric in self.val_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"val/{name}", metric, batch_size=batch_size,
                     on_step=False, on_epoch=True,
                     prog_bar=(name in ("auroc", "auprc")), sync_dist=True)

    # ---- Test ----
    def test_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        batch_size = len(labels)

        self.log("test/loss", out["loss"], batch_size=batch_size,
                 on_step=False, on_epoch=True, sync_dist=True)

        labels_int = labels.long()
        for name, metric in self.test_metrics.items():
            metric.update(probs, labels_int)
            self.log(f"test/{name}", metric, batch_size=batch_size,
                     on_step=False, on_epoch=True, sync_dist=True)

        self.test_probs.append(probs.cpu())
        self.test_labels.append(labels_int.cpu())

    def on_test_epoch_end(self):
        all_probs = torch.cat(self.test_probs)
        all_labels = torch.cat(self.test_labels)
        if self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path(".")
            torch.save(
                {"probs": all_probs, "labels": all_labels},
                out_dir / "test_predictions.pt",
            )
        self.test_probs.clear()
        self.test_labels.clear()

    # ---- Optimiser: separate param groups for ESM vs rest ----
    def configure_optimizers(self):
        esm_params = []
        other_params = []

        esm_modules = (
            self.model.esm_encoder.esm_encoder,
            self.model.esm_encoder.esm_embedding,
        )

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_esm = any(param is p for module in esm_modules for p in module.parameters())
            if is_esm:
                esm_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": self.hparams.lr},
        ]
        if esm_params:
            param_groups.append({
                "params": esm_params,
                "lr": self.hparams.esm_lr,
                "weight_decay": 0.01,
            })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_step):
            warmup = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
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
# 3. Main: train a single fold (with optional phased training)
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
    print(f"  Training fold {fold_idx} (ESM Model B)")
    print(f"  Train: {len(fold_info['train'])} | Val: {len(fold_info['val'])} | Test: {len(fold_info['test'])}")
    print(f"  ESM checkpoint: {args.esm_checkpoint}")
    print(f"  Freeze ESM: {args.freeze_esm} | Tune layers: {args.n_tune_layers}")
    print(f"{'='*60}\n")

    # ---- DataModule ----
    dm = TCRpMHCDataModule(
        csv_path=args.csv,
        pdb_dir=args.pdb_dir,
        fold_info=fold_info,
        esm_checkpoint=args.esm_checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    dm.setup()

    steps_per_epoch = len(fold_info["train"]) // args.batch_size + 1

    # ---- Lightning Module ----
    model = ESMBindingLitModule(
        esm_checkpoint=args.esm_checkpoint,
        freeze_esm=args.freeze_esm,
        n_tune_layers=args.n_tune_layers,
        d_model=args.d_model,
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
        pos_weight=dm.pos_weight,
        lr=args.lr,
        esm_lr=args.esm_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_epochs * steps_per_epoch,
        grad_clip=args.grad_clip,
    )

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")

    # ---- Callbacks ----
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

    loggers = [
        CSVLogger(save_dir=fold_dir, name="csv_logs"),
        TensorBoardLogger(save_dir=fold_dir, name="tb_logs"),
    ]

    # ---- Phase 1: Train with frozen (or partially frozen) ESM ----
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
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval,
        deterministic=False,
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=dm)
    best_ckpt = callbacks[0].best_model_path
    print(f"\nPhase 1 best checkpoint: {best_ckpt}")

    # ---- Phase 2 (optional): Unfreeze last N ESM layers and fine-tune ----
    if args.phase2_unfreeze_layers > 0 and args.phase2_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Unfreezing last {args.phase2_unfreeze_layers} ESM layers")
        print(f"  Fine-tuning for {args.phase2_epochs} additional epochs")
        print(f"{'='*60}\n")

        # Load best Phase 1 checkpoint
        model = ESMBindingLitModule.load_from_checkpoint(
            best_ckpt,
            esm_checkpoint=args.esm_checkpoint,
        )

        # Unfreeze ESM layers
        model.model.set_esm_tuning(
            freeze=True,
            n_tune_layers=args.phase2_unfreeze_layers,
        )

        # Update LR for phase 2 (lower for stability)
        model.hparams.lr = args.phase2_lr
        model.hparams.max_steps = args.phase2_epochs * steps_per_epoch

        # New callbacks for phase 2
        phase2_dir = os.path.join(fold_dir, "phase2")
        callbacks_p2 = [
            ModelCheckpoint(
                dirpath=os.path.join(phase2_dir, "checkpoints"),
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

        loggers_p2 = [
            CSVLogger(save_dir=phase2_dir, name="csv_logs"),
            TensorBoardLogger(save_dir=phase2_dir, name="tb_logs"),
        ]

        trainer_p2 = L.Trainer(
            max_epochs=args.phase2_epochs,
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=args.precision,
            gradient_clip_val=args.grad_clip,
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=callbacks_p2,
            logger=loggers_p2,
            log_every_n_steps=50,
            val_check_interval=args.val_check_interval,
            deterministic=False,
        )

        trainer_p2.fit(model, datamodule=dm)
        best_ckpt = callbacks_p2[0].best_model_path
        print(f"\nPhase 2 best checkpoint: {best_ckpt}")
        trainer = trainer_p2

    # ---- Test with best checkpoint ----
    print(f"\nTesting with: {best_ckpt}")
    test_results = trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)

    # ---- Save results ----
    results = {
        "fold": fold_idx,
        "best_checkpoint": best_ckpt,
        "best_val_auprc": float(callbacks[0].best_model_score or 0),
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
        description="Train ESM multimodal TCR-pMHC binding model (single fold)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--csv", required=True, help="Path to dataset CSV")
    p.add_argument("--pdb_dir", required=True, help="Directory with .pdb structure files")
    p.add_argument("--folds_json", required=True, help="Path to cv_folds.json")
    p.add_argument("--fold", type=int, required=True, help="Fold index (0-indexed)")
    p.add_argument("--output_dir", default="./experiments_esm", help="Output root directory")
    p.add_argument("--cache_dir", default=None, help="Graph cache directory")

    # ESM config
    p.add_argument("--esm_checkpoint", default="facebook/esm2_t12_35M_UR50D",
                    help="ESM-2 model name or local path")
    p.add_argument("--freeze_esm", action="store_true", default=False,
                    help="Freeze ESM encoder in Phase 1")
    p.add_argument("--n_tune_layers", type=int, default=0,
                    help="Number of ESM layers to fine-tune (from the end)")

    # Phase 2: optional ESM fine-tuning
    p.add_argument("--phase2_unfreeze_layers", type=int, default=0,
                    help="Unfreeze last N ESM layers in Phase 2 (0=skip phase 2)")
    p.add_argument("--phase2_epochs", type=int, default=10,
                    help="Number of epochs for Phase 2 fine-tuning")
    p.add_argument("--phase2_lr", type=float, default=5e-5,
                    help="Learning rate for Phase 2")

    # Training
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16,
                    help="Smaller than Model A due to ESM memory cost")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--esm_lr", type=float, default=1e-5,
                    help="Separate LR for ESM params (when not frozen)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--precision", default="bf16-mixed",
                    help="bf16-mixed recommended for ESM to save memory")
    p.add_argument("--num_workers", type=int, default=4)

    # Model architecture (non-ESM components)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_cross_heads", type=int, default=8)
    p.add_argument("--node_feat_size", type=int, default=20)
    p.add_argument("--edge_feat_size", type=int, default=7)
    p.add_argument("--egnn_hidden", type=int, default=128)
    p.add_argument("--egnn_layers", type=int, default=5)
    p.add_argument("--egnn_out_dim", type=int, default=128)
    p.add_argument("--struct_seq_cross_heads", type=int, default=4)
    p.add_argument("--d_fused", type=int, default=256)
    p.add_argument("--clf_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)

    # Seed
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")

    train_fold(args)