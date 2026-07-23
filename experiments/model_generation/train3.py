"""
Training Script — Model B3: ESM + ESM Node Features + Pseudo-Hetero Graph Transformer
======================================================================================

Extends train2.py with model_type "esm3" which uses ESMMultimodalBindingModel3.
Key difference: ESM token embeddings are injected as node features in the
structure encoder (STAG-LLM style).

Usage:
    python train3.py --model_type esm3 \
        --csv data.csv --pdb_dir pdbs/ --folds_json cv_folds.json --fold 0 \
        --esm_checkpoint facebook/esm2_t12_35M_UR50D --freeze_esm --use_lora \
        --esm_node_dim 64

IMPORTANT: Requires rebuilt graph cache with chain_id/chain_pos node data.
           Delete old cache dir before first run.
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
# ESM3 Lightning Module (ESM node features variant)
# ============================================================
class ESM3BindingLitModule(_BaseLitModule):
    """Lightning wrapper for ESMMultimodalBindingModel3 (ESM node features)."""

    def __init__(self, esm_checkpoint="facebook/esm2_t12_35M_UR50D", freeze_esm=True,
                 n_tune_layers=0, d_model=256, n_cross_heads=8, node_feat_size=20,
                 edge_feat_size=7, esm_node_dim=64,
                 struct_hidden_dim=320, struct_edge_hidden=32,
                 struct_n_layers=3, struct_n_heads=4, struct_out_dim=128,
                 struct_seq_cross_heads=4, d_fused=256, clf_hidden=256, dropout=0.2,
                 norm_type="layernorm",
                 pos_weight=1.0, lr=3e-4, esm_lr=1e-5, lora_lr=2e-4,
                 weight_decay=1e-4, warmup_steps=500, max_steps=50000, grad_clip=1.0,
                 use_lora=False, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
                 lora_pmhc=False, pad_id=PAD_IDX):
        super().__init__()
        self.save_hyperparameters(); self._init_metrics()
        from transformers import EsmForMaskedLM
        from model_llm3 import ESMMultimodalBindingModel3
        esm = EsmForMaskedLM.from_pretrained(esm_checkpoint, local_files_only=os.path.isdir(esm_checkpoint))
        self.model = ESMMultimodalBindingModel3(
            esm_encoder=esm.esm.encoder, esm_embedding=esm.get_input_embeddings(),
            esm_hidden_size=esm.config.hidden_size, freeze_esm=freeze_esm,
            n_tune_layers=n_tune_layers,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_n_layers=lora_n_layers, lora_pmhc=lora_pmhc,
            d_model=d_model, n_cross_heads=n_cross_heads,
            dropout=dropout, esm_node_dim=esm_node_dim,
            node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
            struct_hidden_dim=struct_hidden_dim, struct_edge_hidden=struct_edge_hidden,
            struct_n_layers=struct_n_layers, struct_n_heads=struct_n_heads,
            struct_out_dim=struct_out_dim, struct_seq_cross_heads=struct_seq_cross_heads,
            d_fused=d_fused, clf_hidden=clf_hidden, norm_type=norm_type,
            pos_weight=pos_weight)
        del esm

    def _shared_step(self, batch):
        tcr_ids, labels, pmhc_esm, tcr_esm, full_seq, bg, sf, pmhc_mask, tcr_mask = batch
        labels_f = labels.float()
        B = labels.shape[0]
        if bg is not None: bg = bg.to(self.device)
        out = self.model(tcr_ids=tcr_esm, mhc_ids=pmhc_esm, tcr_mask=tcr_mask, mhc_mask=pmhc_mask,
                         struct_graph=bg, struct_available=torch.ones(B, dtype=torch.bool, device=self.device),
                         labels=labels_f, compute_loss=True)
        return out, out["prob"].squeeze(-1), labels_f

    def configure_optimizers(self):
        esm_mods = (self.model.esm_seq_encoder.esm_encoder, self.model.esm_seq_encoder.esm_embedding)
        # Collect LoRA parameters
        lora_param_ids = set()
        lora_p = []
        for adapter in [self.model.tcr_lora, self.model.pmhc_lora]:
            if adapter is not None:
                for p in adapter.parameters():
                    if p.requires_grad:
                        lora_p.append(p)
                        lora_param_ids.add(id(p))

        esm_p, other_p = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in lora_param_ids:
                continue
            if any(p is ep for m in esm_mods for ep in m.parameters()):
                esm_p.append(p)
            else:
                other_p.append(p)

        groups = [{"params": other_p, "lr": self.hparams.lr}]
        if esm_p:
            groups.append({"params": esm_p, "lr": self.hparams.esm_lr, "weight_decay": 0.01})
        if lora_p:
            groups.append({"params": lora_p, "lr": self.hparams.lora_lr, "weight_decay": 0.01})
        return self._build_optimizer(groups)


# ============================================================
# train_fold
# ============================================================
def train_fold(args):
    with open(args.folds_json) as fp: all_folds = json.load(fp)
    assert 0 <= args.fold < len(all_folds), f"Fold {args.fold} out of range"
    fold_info = all_folds[args.fold]
    fold_idx = fold_info.get("fold", args.fold)

    print(f"\n{'='*60}\n  Fold {fold_idx} | {args.model_type.upper()}")
    print(f"  Train: {len(fold_info['train'])} | Val: {len(fold_info['val'])} | Test: {len(fold_info['test'])}\n{'='*60}\n")

    dm = TCRpMHCDataModule(args.csv, args.pdb_dir, fold_info, args.model_type,
                           getattr(args, "esm_checkpoint", None), args.batch_size, args.num_workers, args.cache_dir)
    dm.setup()
    spe = len(fold_info["train"]) // args.batch_size + 1
    ms = args.max_epochs * spe

    model = ESM3BindingLitModule(
        esm_checkpoint=args.esm_checkpoint, freeze_esm=args.freeze_esm,
        n_tune_layers=args.n_tune_layers, d_model=args.d_model, n_cross_heads=args.n_cross_heads,
        node_feat_size=args.node_feat_size, edge_feat_size=args.edge_feat_size,
        esm_node_dim=args.esm_node_dim,
        struct_hidden_dim=args.struct_hidden_dim, struct_edge_hidden=args.struct_edge_hidden,
        struct_n_layers=args.struct_n_layers, struct_n_heads=args.struct_n_heads,
        struct_out_dim=args.struct_out_dim,
        struct_seq_cross_heads=args.struct_seq_cross_heads, d_fused=args.d_fused,
        clf_hidden=args.clf_hidden, dropout=args.dropout,
        norm_type=args.norm_type,
        pos_weight=dm.pos_weight,
        lr=args.lr, esm_lr=args.esm_lr, lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, max_steps=ms, grad_clip=args.grad_clip,
        use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_n_layers=args.lora_n_layers, lora_pmhc=args.lora_pmhc)

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
    print(f"\nPhase 1 best: {best}")

    # Phase 2: unfreeze ESM layers
    if args.phase2_unfreeze_layers > 0 and args.phase2_epochs > 0:
        print(f"\n{'='*60}\n  Phase 2: unfreeze {args.phase2_unfreeze_layers} ESM layers, {args.phase2_epochs} epochs\n{'='*60}\n")
        model = ESM3BindingLitModule.load_from_checkpoint(best, esm_checkpoint=args.esm_checkpoint)
        model.model.set_esm_tuning(freeze=True, n_tune_layers=args.phase2_unfreeze_layers)

        p2_max_steps = args.phase2_epochs * spe
        model.hparams.lr = args.phase2_lr
        model.hparams.esm_lr = args.phase2_esm_lr
        model.hparams.lora_lr = args.phase2_lr * 0.5
        model.hparams.warmup_steps = args.phase2_warmup_steps
        model.hparams.max_steps = p2_max_steps

        print(f"  Phase 2 LRs: base={args.phase2_lr}, esm={args.phase2_esm_lr}, "
              f"lora={args.phase2_lr * 0.5:.1e}, warmup={args.phase2_warmup_steps} steps")

        p2dir = os.path.join(fold_dir, "phase2")
        ckpt2 = ModelCheckpoint(dirpath=os.path.join(p2dir, "checkpoints"), filename="best-{epoch:03d}-{val/auprc:.4f}",
                                monitor="val/auprc", mode="max", save_top_k=2, save_last=True, verbose=True)
        cbs2 = [ckpt2, EarlyStopping(monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-4, verbose=True),
                LearningRateMonitor(logging_interval="step"), RichProgressBar()]
        logs2 = [CSVLogger(save_dir=p2dir, name="csv_logs"), TensorBoardLogger(save_dir=p2dir, name="tb_logs")]
        trainer = L.Trainer(max_epochs=args.phase2_epochs, accelerator="auto", devices="auto", strategy="auto",
                            precision=args.precision, gradient_clip_val=args.grad_clip,
                            accumulate_grad_batches=args.accumulate_grad_batches, callbacks=cbs2, logger=logs2,
                            log_every_n_steps=50, val_check_interval=args.val_check_interval, deterministic=False)
        trainer.fit(model, datamodule=dm)
        best = ckpt2.best_model_path
        print(f"\nPhase 2 best: {best}")

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
        description="Train ESM3 (ESM node features) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_type", default="esm3", choices=["esm3"],
                   help="Model type (esm3 = ESM + ESM node features + hetero graph transformer)")
    p.add_argument("--csv", required=True); p.add_argument("--pdb_dir", required=True)
    p.add_argument("--folds_json", required=True); p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output_dir", default="./experiments"); p.add_argument("--cache_dir", default=None)
    # Training
    p.add_argument("--max_epochs", type=int, default=100); p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500); p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8); p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--val_check_interval", type=float, default=1.0); p.add_argument("--precision", default="32")
    p.add_argument("--num_workers", type=int, default=4); p.add_argument("--seed", type=int, default=42)
    # Shared arch
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_cross_heads", type=int, default=4)
    p.add_argument("--node_feat_size", type=int, default=20); p.add_argument("--edge_feat_size", type=int, default=7)
    p.add_argument("--struct_seq_cross_heads", type=int, default=4)
    p.add_argument("--d_fused", type=int, default=256); p.add_argument("--clf_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm_type", choices=["layernorm", "batchnorm"], default="layernorm",
                   help="Normalization in classifier head")
    # ESM node feature
    p.add_argument("--esm_node_dim", type=int, default=64,
                   help="Projection dim for ESM embeddings used as graph node features")
    # Hetero structure encoder
    h = p.add_argument_group("Hetero struct")
    h.add_argument("--struct_hidden_dim", type=int, default=320)
    h.add_argument("--struct_edge_hidden", type=int, default=32)
    h.add_argument("--struct_n_layers", type=int, default=3)
    h.add_argument("--struct_n_heads", type=int, default=4)
    h.add_argument("--struct_out_dim", type=int, default=128)
    # ESM
    e = p.add_argument_group("ESM"); e.add_argument("--esm_checkpoint", default="facebook/esm2_t12_35M_UR50D")
    e.add_argument("--freeze_esm", action="store_true", default=False)
    e.add_argument("--n_tune_layers", type=int, default=0); e.add_argument("--esm_lr", type=float, default=1e-5)
    e.add_argument("--phase2_unfreeze_layers", type=int, default=0)
    e.add_argument("--phase2_epochs", type=int, default=10); e.add_argument("--phase2_lr", type=float, default=5e-5)
    e.add_argument("--phase2_esm_lr", type=float, default=2e-6)
    e.add_argument("--phase2_warmup_steps", type=int, default=1000)
    # LoRA
    lo = p.add_argument_group("LoRA"); lo.add_argument("--use_lora", action="store_true", default=False)
    lo.add_argument("--lora_rank", type=int, default=8); lo.add_argument("--lora_alpha", type=float, default=16.0)
    lo.add_argument("--lora_n_layers", type=int, default=4); lo.add_argument("--lora_lr", type=float, default=2e-4)
    lo.add_argument("--lora_pmhc", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    train_fold(args)
