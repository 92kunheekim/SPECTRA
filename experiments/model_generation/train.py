"""
Unified Lightning Training Script — Model A (VAE) & Model B (ESM)
==================================================================

Usage (Model A — VAE):
    python train.py --model_type vae \
        --csv data.csv --pdb_dir pdbs/ --folds_json cv_folds.json --fold 0

Usage (Model B — ESM):
    python train.py --model_type esm \
        --csv data.csv --pdb_dir pdbs/ --folds_json cv_folds.json --fold 0 \
        --esm_checkpoint facebook/esm2_t12_35M_UR50D --freeze_esm
"""

import os, json, argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from dataset import ProteinGraphDataset, custom_collate_fn, make_esm_collate_fn, PAD_IDX, AA_VOCAB

# ============================================================
# 1. DataModule (shared — selects collate by model_type)
# ============================================================
class TCRpMHCDataModule(L.LightningDataModule):
    def __init__(self, csv_path, pdb_dir, fold_info, model_type="vae",
                 esm_checkpoint=None, batch_size=32, num_workers=4, cache_dir=None):
        super().__init__()
        self.csv_path, self.pdb_dir, self.fold_info = csv_path, pdb_dir, fold_info
        self.batch_size, self.num_workers, self.cache_dir = batch_size, num_workers, cache_dir
        if model_type == "esm":
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(esm_checkpoint, local_files_only=os.path.isdir(esm_checkpoint))
            self.collate_fn = make_esm_collate_fn(tok)
        else:
            self.collate_fn = custom_collate_fn

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        required = {"CDR3a","CDR3b","MHC_sequence","peptide","TCR_A_sequence","TCR_B_sequence","label"}
        df = df.dropna(subset=list(required)).reset_index(drop=True)
        kwargs = dict(pdb_dfs=df, pdb_dir=self.pdb_dir)
        if self.cache_dir: kwargs["cache_dir"] = self.cache_dir
        self.dataset = ProteinGraphDataset(**kwargs)
        self.train_idx = list(set(self.fold_info["train"]) & set(self.dataset.pdb_dfs.index))
        self.val_idx   = list(set(self.fold_info["val"])   & set(self.dataset.pdb_dfs.index))
        self.test_idx  = list(set(self.fold_info["test"])  & set(self.dataset.pdb_dfs.index))
        train_labels = self.dataset.pdb_dfs.loc[self.train_idx, "label"]
        n_pos, n_neg = (train_labels == 1).sum(), (train_labels == 0).sum()
        self.pos_weight = float(n_neg) / max(float(n_pos), 1.0)

    def _loader(self, idx, shuffle=False, drop_last=False):
        return DataLoader(Subset(self.dataset, idx), batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0, drop_last=drop_last)
    def train_dataloader(self): return self._loader(self.train_idx, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_idx)
    def test_dataloader(self):  return self._loader(self.test_idx)

# ============================================================
# 2. Base Lightning Module (shared metrics, logging, optimizer)
# ============================================================
def _make_metrics():
    return torch.nn.ModuleDict({
        "auroc": BinaryAUROC(), "auprc": BinaryAveragePrecision(),
        "f1": BinaryF1Score(), "acc": BinaryAccuracy(),
        "prec": BinaryPrecision(), "rec": BinaryRecall(),
    })

class _BaseLitModule(L.LightningModule):
    def _init_metrics(self):
        self.train_metrics, self.val_metrics, self.test_metrics = _make_metrics(), _make_metrics(), _make_metrics()
        self.test_probs, self.test_labels = [], []
        self._nan_grad_count = 0

    def _shared_step(self, batch):
        raise NotImplementedError

    # FIX: Sanitize NaN gradients BEFORE optimizer step.
    # In bf16/fp16, the backward pass can produce NaN gradients even when
    # the loss is finite. Without this, NaN gradients corrupt parameters
    # permanently, causing all subsequent forward passes to produce NaN.
    def on_before_optimizer_step(self, optimizer):
        has_nan = False
        for p in self.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                has_nan = True
        if has_nan:
            self._nan_grad_count += 1
            if self._nan_grad_count <= 5:
                print(f"[WARN] NaN/Inf gradients detected and zeroed (occurrence #{self._nan_grad_count})")

    def training_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        B = len(labels)
        loss = out["loss"]

        # Guard against NaN — skip step to protect optimizer state
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("train/nan_count", 1.0, batch_size=B, on_step=True, on_epoch=True,
                     reduce_fx="sum", prog_bar=True)
            return None  # Lightning skips optimizer.step() when None is returned

        self.log("train/loss", loss, batch_size=B, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/bind_loss", out["bind_loss"], batch_size=B, on_step=False, on_epoch=True)
        for k in ("recon_tcr","recon_mhc","kl_tcr","kl_mhc"):
            if k in out: self.log(f"train/{k}", out[k], batch_size=B, on_step=False, on_epoch=True)
        if "beta" in out: self.log("train/beta", out["beta"], batch_size=B, on_step=True, on_epoch=False)
        li = labels.long()
        # FIX: Only update metrics if probs are finite (no NaN in predictions)
        if not torch.isnan(probs).any():
            for name, m in self.train_metrics.items():
                m.update(probs, li); self.log(f"train/{name}", m, batch_size=B, on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        B = len(labels)
        loss = out["loss"]
        # Always log val/loss so Lightning sees the metric exists (needed for
        # ModelCheckpoint / EarlyStopping / progress bar display).
        self.log("val/loss", loss, batch_size=B, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("val/nan_count", 1.0, batch_size=B, on_step=False, on_epoch=True,
                     reduce_fx="sum", sync_dist=True)
            return
        li = labels.long()
        # Only feed non-NaN predictions into torchmetrics
        if not torch.isnan(probs).any():
            for name, m in self.val_metrics.items():
                m.update(probs, li)
                self.log(f"val/{name}", m, batch_size=B, on_step=False, on_epoch=True,
                         prog_bar=(name in ("auroc","auprc")), sync_dist=True)

    def test_step(self, batch, batch_idx):
        out, probs, labels = self._shared_step(batch)
        B = len(labels); li = labels.long()
        loss = out["loss"]
        if not (torch.isnan(loss) or torch.isinf(loss)):
            self.log("test/loss", loss, batch_size=B, on_step=False, on_epoch=True, sync_dist=True)
        if not torch.isnan(probs).any():
            for name, m in self.test_metrics.items():
                m.update(probs, li); self.log(f"test/{name}", m, batch_size=B, on_step=False, on_epoch=True, sync_dist=True)
            self.test_probs.append(probs.cpu()); self.test_labels.append(li.cpu())

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            d = Path(self.trainer.log_dir) if self.trainer.log_dir else Path(".")
            d.mkdir(parents=True, exist_ok=True)
            if self.test_probs:
                torch.save({"probs": torch.cat(self.test_probs), "labels": torch.cat(self.test_labels)}, d / "test_predictions.pt")
        self.test_probs.clear(); self.test_labels.clear()

    def _build_optimizer(self, param_groups):
        opt = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        def lr_lambda(step):
            w, ms = self.hparams.warmup_steps, self.hparams.max_steps
            if step < w: return float(step) / max(1, w)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * (step - w) / max(1, ms - w))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

# ============================================================
# 3. Model A: VAE
# ============================================================
class VAEBindingLitModule(_BaseLitModule):
    def __init__(self, vocab_size=len(AA_VOCAB), pad_id=PAD_IDX, d_model=256, latent_dim=64,
                 n_enc_layers=2, n_cross_heads=4, node_feat_size=20, edge_feat_size=7,
                 egnn_hidden=128, egnn_layers=5, egnn_out_dim=128, struct_seq_cross_heads=4,
                 d_fused=256, clf_hidden=128, dropout=0.1, kl_anneal_steps=5000,
                 lambda_bind=1.0, lambda_recon=0.3, lambda_kl=0.2, pos_weight=1.0,
                 lr=1e-3, weight_decay=1e-5, warmup_steps=500, max_steps=50000, grad_clip=1.0):
        super().__init__()
        self.save_hyperparameters(); self._init_metrics()
        from model import MultimodalBindingModel
        self.model = MultimodalBindingModel(
            vocab_size=vocab_size, pad_id=pad_id, d_model=d_model, latent_dim=latent_dim,
            n_enc_layers=n_enc_layers, n_cross_heads=n_cross_heads,
            node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
            egnn_hidden=egnn_hidden, egnn_layers=egnn_layers, egnn_out_dim=egnn_out_dim,
            struct_seq_cross_heads=struct_seq_cross_heads, d_fused=d_fused, clf_hidden=clf_hidden,
            dropout=dropout, kl_anneal_steps=kl_anneal_steps,
            lambda_bind=lambda_bind, lambda_recon=lambda_recon, lambda_kl=lambda_kl, pos_weight=pos_weight)

    def _shared_step(self, batch):
        tcr_ids, labels, pmhc_seq, tcr_seq, full_seq, bg, struct_features = batch
        labels_f = labels.float()
        pmhc_mask = (pmhc_seq != self.hparams.pad_id)
        tcr_mask  = (tcr_seq  != self.hparams.pad_id)
        B = labels.shape[0]
        if bg is not None: bg = bg.to(self.device)
        out = self.model(tcr_ids=tcr_seq, mhc_ids=pmhc_seq, tcr_mask=tcr_mask, mhc_mask=pmhc_mask,
                         struct_graph=bg, struct_available=torch.ones(B, dtype=torch.bool, device=self.device),
                         labels=labels_f, compute_loss=True)
        return out, out["prob"].squeeze(-1), labels_f

    def configure_optimizers(self):
        return self._build_optimizer([{"params": self.parameters(), "lr": self.hparams.lr}])

# ============================================================
# 4. Model B: ESM
# ============================================================
class ESMBindingLitModule(_BaseLitModule):
    def __init__(self, esm_checkpoint="facebook/esm2_t12_35M_UR50D", freeze_esm=True,
                 n_tune_layers=0, d_model=256, n_cross_heads=8, node_feat_size=20,
                 edge_feat_size=7, egnn_hidden=128, egnn_layers=5, egnn_out_dim=128,
                 struct_seq_cross_heads=4, d_fused=256, clf_hidden=256, dropout=0.2,
                 pos_weight=1.0, lr=3e-4, esm_lr=1e-5, lora_lr=2e-4,
                 weight_decay=1e-4, warmup_steps=500, max_steps=50000, grad_clip=1.0,
                 use_lora=False, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
                 lora_pmhc=False, pad_id=PAD_IDX):
        super().__init__()
        self.save_hyperparameters(); self._init_metrics()
        from transformers import EsmForMaskedLM
        from model_llm import ESMMultimodalBindingModel
        esm = EsmForMaskedLM.from_pretrained(esm_checkpoint, local_files_only=os.path.isdir(esm_checkpoint))
        self.model = ESMMultimodalBindingModel(
            esm_model=esm.esm,
            esm_hidden_size=esm.config.hidden_size, freeze_esm=freeze_esm,
            n_tune_layers=n_tune_layers,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_n_layers=lora_n_layers, lora_pmhc=lora_pmhc,
            d_model=d_model, n_cross_heads=n_cross_heads,
            dropout=dropout, node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
            egnn_hidden=egnn_hidden, egnn_layers=egnn_layers, egnn_out_dim=egnn_out_dim,
            struct_seq_cross_heads=struct_seq_cross_heads, d_fused=d_fused, clf_hidden=clf_hidden,
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
        esm_backbone = self.model.esm_encoder.esm_model
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
                continue  # already in lora group
            if any(p is ep for ep in esm_backbone.parameters()):
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
# 5. train_fold
# ============================================================
def train_fold(args):
    with open(args.folds_json) as fp: all_folds = json.load(fp)
    assert 0 <= args.fold < len(all_folds), f"Fold {args.fold} out of range"
    fold_info = all_folds[args.fold]
    fold_idx = fold_info.get("fold", args.fold)

    print(f"\n{'='*60}\n  Fold {fold_idx} | {args.model_type.upper()}")
    print(f"  Train: {len(fold_info['train'])} | Val: {len(fold_info['val'])} | Test: {len(fold_info['test'])}\n{'='*60}\n")

    dm = TCRpMHCDataModule(args.csv, args.pdb_dir, fold_info, args.model_type,
                           getattr(args,"esm_checkpoint",None), args.batch_size, args.num_workers, args.cache_dir)
    dm.setup()
    spe = len(fold_info["train"]) // args.batch_size + 1
    ms = args.max_epochs * spe

    if args.model_type == "vae":
        model = VAEBindingLitModule(
            d_model=args.d_model, latent_dim=args.latent_dim, n_enc_layers=args.n_enc_layers,
            n_cross_heads=args.n_cross_heads, node_feat_size=args.node_feat_size,
            edge_feat_size=args.edge_feat_size, egnn_hidden=args.egnn_hidden, egnn_layers=args.egnn_layers,
            egnn_out_dim=args.egnn_out_dim, struct_seq_cross_heads=args.struct_seq_cross_heads,
            d_fused=args.d_fused, clf_hidden=args.clf_hidden, dropout=args.dropout,
            kl_anneal_steps=args.kl_anneal_steps, lambda_bind=args.lambda_bind,
            lambda_recon=args.lambda_recon, lambda_kl=args.lambda_kl,
            pos_weight=dm.pos_weight, lr=args.lr, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, max_steps=ms, grad_clip=args.grad_clip)
    else:
        model = ESMBindingLitModule(
            esm_checkpoint=args.esm_checkpoint, freeze_esm=args.freeze_esm,
            n_tune_layers=args.n_tune_layers, d_model=args.d_model, n_cross_heads=args.n_cross_heads,
            node_feat_size=args.node_feat_size, edge_feat_size=args.edge_feat_size,
            egnn_hidden=args.egnn_hidden, egnn_layers=args.egnn_layers, egnn_out_dim=args.egnn_out_dim,
            struct_seq_cross_heads=args.struct_seq_cross_heads, d_fused=args.d_fused,
            clf_hidden=args.clf_hidden, dropout=args.dropout, pos_weight=dm.pos_weight,
            lr=args.lr, esm_lr=args.esm_lr, lora_lr=args.lora_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, max_steps=ms, grad_clip=args.grad_clip,
            use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_n_layers=args.lora_n_layers, lora_pmhc=args.lora_pmhc)

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    ckpt = ModelCheckpoint(dirpath=os.path.join(fold_dir,"checkpoints"),
                           filename="best-{epoch:03d}-{val/auprc:.4f}", monitor="val/auprc",
                           mode="max", save_top_k=2, save_last=True, verbose=True)
    cbs = [ckpt, EarlyStopping(monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-4, verbose=True),
           LearningRateMonitor(logging_interval="step"), RichProgressBar()]
    logs = [CSVLogger(save_dir=fold_dir, name="csv_logs"), TensorBoardLogger(save_dir=fold_dir, name="tb_logs")]

    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices="auto", strategy="auto",
                        precision=args.precision, gradient_clip_val=args.grad_clip,
                        accumulate_grad_batches=args.accumulate_grad_batches, callbacks=cbs, logger=logs,
                        log_every_n_steps=100, val_check_interval=args.val_check_interval,
                        deterministic=False, enable_model_summary=True)
    trainer.fit(model, datamodule=dm)
    best = ckpt.best_model_path
    print(f"\nPhase 1 best: {best}")

    # Phase 2 (ESM only)
    if args.model_type == "esm" and args.phase2_unfreeze_layers > 0 and args.phase2_epochs > 0:
        print(f"\n{'='*60}\n  Phase 2: unfreeze {args.phase2_unfreeze_layers} ESM layers, {args.phase2_epochs} epochs\n{'='*60}\n")
        model = ESMBindingLitModule.load_from_checkpoint(best, esm_checkpoint=args.esm_checkpoint)
        model.model.set_esm_tuning(freeze=True, n_tune_layers=args.phase2_unfreeze_layers)
        model.hparams.lr = args.phase2_lr; model.hparams.max_steps = args.phase2_epochs * spe
        p2dir = os.path.join(fold_dir, "phase2")
        ckpt2 = ModelCheckpoint(dirpath=os.path.join(p2dir,"checkpoints"), filename="best-{epoch:03d}-{val/auprc:.4f}",
                                monitor="val/auprc", mode="max", save_top_k=2, save_last=True, verbose=True)
        cbs2 = [ckpt2, EarlyStopping(monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-4, verbose=True),
                LearningRateMonitor(logging_interval="step"), RichProgressBar()]
        logs2 = [CSVLogger(save_dir=p2dir, name="csv_logs"), TensorBoardLogger(save_dir=p2dir, name="tb_logs")]
        trainer = L.Trainer(max_epochs=args.phase2_epochs, accelerator="auto", devices="auto", strategy="auto",
                            precision=args.precision, gradient_clip_val=args.grad_clip,
                            accumulate_grad_batches=args.accumulate_grad_batches, callbacks=cbs2, logger=logs2,
                            log_every_n_steps=100, val_check_interval=args.val_check_interval, deterministic=False)
        trainer.fit(model, datamodule=dm)
        best = ckpt2.best_model_path
        print(f"\nPhase 2 best: {best}")

    print(f"\nTesting: {best}")
    test_res = trainer.test(model, datamodule=dm, ckpt_path=best)
    os.makedirs(fold_dir, exist_ok=True)
    results = {"fold": fold_idx, "model_type": args.model_type, "best_checkpoint": best,
               "test_results": test_res[0] if test_res else {}, "hparams": {k: str(v) for k, v in vars(args).items()}}
    with open(os.path.join(fold_dir, "results.json"), "w") as fp: json.dump(results, fp, indent=2, default=str)
    return results

# ============================================================
# 6. CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_type", choices=["vae","esm"], required=True)
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
    # Shared arch
    p.add_argument("--d_model", type=int, default=256); p.add_argument("--n_cross_heads", type=int, default=4)
    p.add_argument("--node_feat_size", type=int, default=20); p.add_argument("--edge_feat_size", type=int, default=7)
    p.add_argument("--egnn_hidden", type=int, default=128); p.add_argument("--egnn_layers", type=int, default=5)
    p.add_argument("--egnn_out_dim", type=int, default=128); p.add_argument("--struct_seq_cross_heads", type=int, default=4)
    p.add_argument("--d_fused", type=int, default=256); p.add_argument("--clf_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # VAE-specific
    v = p.add_argument_group("VAE"); v.add_argument("--latent_dim", type=int, default=64)
    v.add_argument("--n_enc_layers", type=int, default=2); v.add_argument("--kl_anneal_steps", type=int, default=5000)
    v.add_argument("--lambda_bind", type=float, default=1.0); v.add_argument("--lambda_recon", type=float, default=0.3)
    v.add_argument("--lambda_kl", type=float, default=0.2)
    # ESM-specific
    e = p.add_argument_group("ESM"); e.add_argument("--esm_checkpoint", default="facebook/esm2_t12_35M_UR50D")
    e.add_argument("--freeze_esm", action="store_true", default=False)
    e.add_argument("--n_tune_layers", type=int, default=0); e.add_argument("--esm_lr", type=float, default=1e-5)
    e.add_argument("--phase2_unfreeze_layers", type=int, default=0)
    e.add_argument("--phase2_epochs", type=int, default=10); e.add_argument("--phase2_lr", type=float, default=5e-5)
    # LoRA-specific
    lo = p.add_argument_group("LoRA"); lo.add_argument("--use_lora", action="store_true", default=False)
    lo.add_argument("--lora_rank", type=int, default=8); lo.add_argument("--lora_alpha", type=float, default=16.0)
    lo.add_argument("--lora_n_layers", type=int, default=4); lo.add_argument("--lora_lr", type=float, default=2e-4)
    lo.add_argument("--lora_pmhc", action="store_true", default=False,
                    help="Also create a separate LoRA adapter for pMHC (default: TCR only)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    train_fold(args)