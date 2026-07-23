"""
Training Script — Structure-Only Ablation (No ESM Sequence Branch)
===================================================================

Trains StructureOnlyModel from model_struct.py. This is an ablation
experiment to isolate the contribution of the pseudo-heterogeneous
graph transformer using only one-hot AA node features (no ESM).

Usage:
    python train_struct.py --model_type struct \
        --csv data.csv --pdb_dir pdbs/ --folds_json cv_folds.json --fold 0

Compare with:
    - train_seq.py  (--model_type seq)   → ESM sequence-only
    - train2.py     (--model_type esm2)  → ESM + graph transformer
    - train3.py     (--model_type esm3)  → ESM + graph + ESM node features

NOTE: No Phase 2 (no ESM layers to unfreeze). No LoRA. Single LR group.
"""

import os, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from dataset import ProteinGraphDataset, custom_collate_fn, make_esm_collate_fn, PAD_IDX, AA_VOCAB

# Import shared base classes from train2
from train2 import TCRpMHCDataModule, _BaseLitModule, _make_metrics


# ============================================================
# Structure-Only Lightning Module
# ============================================================
class StructOnlyBindingLitModule(_BaseLitModule):
    """Lightning wrapper for StructureOnlyModel (no ESM, graph only)."""

    def __init__(self, node_feat_size=20, edge_feat_size=7,
                 struct_hidden_dim=320, struct_edge_hidden=32,
                 struct_n_layers=3, struct_n_heads=4, struct_out_dim=128,
                 d_fused=256, clf_hidden=256, dropout=0.2,
                 norm_type="layernorm",
                 pos_weight=1.0, lr=3e-4,
                 weight_decay=1e-4, warmup_steps=500, max_steps=50000, grad_clip=1.0):
        super().__init__()
        self.save_hyperparameters(); self._init_metrics()
        from model_struct import StructureOnlyModel
        self.model = StructureOnlyModel(
            node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
            struct_hidden_dim=struct_hidden_dim, struct_edge_hidden=struct_edge_hidden,
            struct_n_layers=struct_n_layers, struct_n_heads=struct_n_heads,
            struct_out_dim=struct_out_dim,
            d_fused=d_fused, clf_hidden=clf_hidden, dropout=dropout,
            norm_type=norm_type,
            pos_weight=pos_weight)

    def _shared_step(self, batch):
        # ESM collate format: (tcr_ids, labels, pmhc_esm, tcr_esm, full_seq, bg, sf, pmhc_mask, tcr_mask)
        # We only need labels and bg (the graph)
        tcr_ids, labels, pmhc_esm, tcr_esm, full_seq, bg, sf, pmhc_mask, tcr_mask = batch
        labels_f = labels.float()
        if bg is not None:
            bg = bg.to(self.device)
        out = self.model(struct_graph=bg, labels=labels_f, compute_loss=True)
        return out, out["prob"].squeeze(-1), labels_f

    def configure_optimizers(self):
        # Single LR group — no ESM, no LoRA
        groups = [{"params": self.parameters(), "lr": self.hparams.lr}]
        return self._build_optimizer(groups)


# ============================================================
# train_fold
# ============================================================
def train_fold(args):
    with open(args.folds_json) as fp: all_folds = json.load(fp)
    assert 0 <= args.fold < len(all_folds), f"Fold {args.fold} out of range"
    fold_info = all_folds[args.fold]
    fold_idx = fold_info.get("fold", args.fold)

    print(f"\n{'='*60}\n  Fold {fold_idx} | {args.model_type.upper()} (Structure-Only)")
    print(f"  Train: {len(fold_info['train'])} | Val: {len(fold_info['val'])} | Test: {len(fold_info['test'])}\n{'='*60}\n")

    dm = TCRpMHCDataModule(args.csv, args.pdb_dir, fold_info, args.model_type,
                           getattr(args, "esm_checkpoint", None), args.batch_size, args.num_workers, args.cache_dir)
    dm.setup()
    spe = len(fold_info["train"]) // args.batch_size + 1
    ms = args.max_epochs * spe

    model = StructOnlyBindingLitModule(
        node_feat_size=args.node_feat_size, edge_feat_size=args.edge_feat_size,
        struct_hidden_dim=args.struct_hidden_dim, struct_edge_hidden=args.struct_edge_hidden,
        struct_n_layers=args.struct_n_layers, struct_n_heads=args.struct_n_heads,
        struct_out_dim=args.struct_out_dim,
        d_fused=args.d_fused, clf_hidden=args.clf_hidden, dropout=args.dropout,
        norm_type=args.norm_type,
        pos_weight=dm.pos_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, max_steps=ms, grad_clip=args.grad_clip)

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    ckpt = ModelCheckpoint(dirpath=os.path.join(fold_dir, "checkpoints"),
                           filename="best-{epoch:03d}-{val/auprc:.4f}", monitor="val/auprc",
                           mode="max", save_top_k=2, save_last=True, verbose=True)
    cbs = [ckpt, EarlyStopping(monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-4, verbose=True),
           LearningRateMonitor(logging_interval="step"), RichProgressBar()]
    logs = [CSVLogger(save_dir=fold_dir, name="csv_logs"), TensorBoardLogger(save_dir=fold_dir, name="tb_logs")]

    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices="auto", strategy="auto",
                        precision=args.precision, gradient_clip_val=args.grad_clip,
                        accumulate_grad_batches=args.accumulate_grad_batches, callbacks=cbs, logger=logs,
                        log_every_n_steps=50, val_check_interval=args.val_check_interval,
                        deterministic=False, enable_model_summary=True)
    trainer.fit(model, datamodule=dm)
    best = ckpt.best_model_path
    print(f"\nBest checkpoint: {best}")

    # No Phase 2 — nothing to unfreeze

    # Testing
    print(f"\nTesting: {best}")
    test_trainer = L.Trainer(
        accelerator="auto", devices="auto", strategy="auto",
        precision=args.precision, logger=[CSVLogger(save_dir=fold_dir, name="test_logs")],
        enable_model_summary=False,
    )
    test_res = test_trainer.test(model, datamodule=dm, ckpt_path=best)
    os.makedirs(fold_dir, exist_ok=True)
    results = {"fold": fold_idx, "model_type": args.model_type, "best_checkpoint": best,
               "test_results": test_res[0] if test_res else {}, "hparams": {k: str(v) for k, v in vars(args).items()}}
    with open(os.path.join(fold_dir, "results.json"), "w") as fp: json.dump(results, fp, indent=2, default=str)
    return results


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Train structure-only ablation model (no ESM sequence branch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_type", default="struct", choices=["struct"],
                   help="Model type (struct = graph-only, no ESM)")
    p.add_argument("--csv", required=True); p.add_argument("--pdb_dir", required=True)
    p.add_argument("--folds_json", required=True); p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output_dir", default="./experiments"); p.add_argument("--cache_dir", default=None)
    # Training
    p.add_argument("--max_epochs", type=int, default=100); p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500); p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--val_check_interval", type=float, default=1.0); p.add_argument("--precision", default="32")
    p.add_argument("--num_workers", type=int, default=4); p.add_argument("--seed", type=int, default=42)
    # Structure encoder
    p.add_argument("--node_feat_size", type=int, default=20); p.add_argument("--edge_feat_size", type=int, default=7)
    h = p.add_argument_group("Hetero struct")
    h.add_argument("--struct_hidden_dim", type=int, default=320)
    h.add_argument("--struct_edge_hidden", type=int, default=32)
    h.add_argument("--struct_n_layers", type=int, default=3)
    h.add_argument("--struct_n_heads", type=int, default=4)
    h.add_argument("--struct_out_dim", type=int, default=128)
    # Classifier
    p.add_argument("--d_fused", type=int, default=256); p.add_argument("--clf_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm_type", choices=["layernorm", "batchnorm"], default="layernorm",
                   help="Normalization in classifier head")
    # ESM checkpoint is needed for the collate function (tokenizer)
    p.add_argument("--esm_checkpoint", default="facebook/esm2_t12_35M_UR50D",
                   help="ESM checkpoint (used only for tokenizer in DataModule collate)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    train_fold(args)
