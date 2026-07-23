"""
run_ablation.py — Architecture Ablation Experiments
=====================================================

Runs one or more ablation modes (A-H) with identical data splits,
hyperparameters, and evaluation. Produces a comparison CSV.

Modes:
  A: concat_cls              1 ESM pass, [CLS], no Rosetta
  B: 4chain_pool             4 ESM pass, pool, no cross-attn, no Rosetta
  C: 4chain_pool_rosetta     4 ESM pass, pool, no cross-attn, + Rosetta
  D: 4chain_crossattn        4 ESM pass, 3 cross-attn, no Rosetta
  E: 4chain_crossattn_rosetta  FULL MODEL
  F: concat_cls_rosetta      1 ESM pass, [CLS], + Rosetta
  G: 2chain_pool             2 ESM pass (pMHC, TCR), no Rosetta
  H: 2chain_pool_rosetta     2 ESM pass (pMHC, TCR), + Rosetta

Usage:
  # Run all 8 modes
  python run_ablation.py --modes A B C D E F G H --data_csv data.csv

  # Run just the key comparison
  python run_ablation.py --modes A F E --data_csv data.csv

  # Run single mode
  python run_ablation.py --modes E --data_csv data.csv --epochs 30
"""

import argparse
import os
import json
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
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

from spectra.data.dataset_ablation import (
    AblationDataset, make_ablation_collate_fn, load_datasets,
)
from spectra.models.ablation_model import (
    AblationModel, MODE_CONFIG, ABLATION_MODES,
)
from spectra.training.distributed import add_distributed_args, make_trainer


# ============================================================
# 1. Graduated Unfreezing Callback
# ============================================================

class GraduatedUnfreeze(Callback):
    def __init__(self, phase_b=10, phase_c=20):
        self.phase_b = phase_b
        self.phase_c = phase_c
        self.current = "A"

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        if e >= self.phase_c and self.current != "C":
            pl_module.model.set_esm_tuning(freeze=False, n_tune_layers=4)
            self.current = "C"
            for pg in trainer.optimizers[0].param_groups:
                if pg.get("is_esm"):
                    pg["lr"] = pl_module.hparams.esm_lr
        elif e >= self.phase_b and self.current == "A":
            pl_module.model.set_esm_tuning(freeze=False, n_tune_layers=2)
            self.current = "B"
            for pg in trainer.optimizers[0].param_groups:
                if pg.get("is_esm"):
                    pg["lr"] = pl_module.hparams.esm_lr


# ============================================================
# 4. Lightning Module
# ============================================================

class AblationLightning(L.LightningModule):

    def __init__(self, model, lr=3e-4, esm_lr=1e-5, weight_decay=1e-4,
                 warmup_epochs=3, max_epochs=30, pos_weight=5.0):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.register_buffer("bce_pw", torch.tensor([min(pos_weight, 10.0)]))

        for s in ['train', 'val', 'test']:
            setattr(self, f'{s}_auroc', BinaryAUROC())
            setattr(self, f'{s}_f1', BinaryF1Score())
            setattr(self, f'{s}_prec', BinaryPrecision())
            setattr(self, f'{s}_rec', BinaryRecall())
            setattr(self, f'{s}_acc', BinaryAccuracy())
            setattr(self, f'{s}_mcc', BinaryMatthewsCorrCoef())

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _step(self, batch):
        batch = self._to_device(batch)
        out = self.model(
            concat_ids=batch["concat_ids"], concat_mask=batch["concat_mask"],
            pmhc_ids=batch["pmhc_ids"], pmhc_mask=batch["pmhc_mask"],
            tcr_ids=batch["tcr_ids"], tcr_mask=batch["tcr_mask"],
            mhc_ids=batch["mhc_ids"], mhc_mask=batch["mhc_mask"],
            pep_ids=batch["pep_ids"], pep_mask=batch["pep_mask"],
            tra_ids=batch["tra_ids"], tra_mask=batch["tra_mask"],
            trb_ids=batch["trb_ids"], trb_mask=batch["trb_mask"],
            rosetta_features=batch["rosetta"],
            rosetta_available=batch["has_rosetta"],
            compute_loss=False,
        )
        labels = batch["labels"]
        weights = batch["weights"]
        logit = out["logit"].view(-1).clamp(-10.0, 10.0)
        per_sample = F.binary_cross_entropy_with_logits(
            logit, labels.float(), pos_weight=self.bce_pw, reduction='none')
        out["loss"] = (per_sample * weights).sum() / weights.sum()
        return out, labels

    def _update_metrics(self, split, probs, labels):
        getattr(self, f'{split}_auroc').update(probs, labels)
        getattr(self, f'{split}_f1').update(probs, labels)
        getattr(self, f'{split}_prec').update(probs, labels)
        getattr(self, f'{split}_rec').update(probs, labels)
        getattr(self, f'{split}_acc').update(probs, labels)
        getattr(self, f'{split}_mcc').update(probs, labels)

    def _log_and_reset(self, split, prog_bar_keys=None):
        if prog_bar_keys is None:
            prog_bar_keys = {"auroc", "f1"}
        for metric_name in ['auroc', 'f1', 'prec', 'rec', 'acc', 'mcc']:
            m = getattr(self, f'{split}_{metric_name}')
            pb = metric_name in prog_bar_keys
            self.log(f'{split}_{metric_name}', m.compute(), prog_bar=pb)
            m.reset()

    def training_step(self, batch, batch_idx):
        out, labels = self._step(batch)
        self.log("train_loss", out["loss"], prog_bar=True, batch_size=labels.size(0))
        self._update_metrics("train", out["prob"].squeeze(-1), labels)
        return out["loss"]

    def on_train_epoch_end(self):
        self._log_and_reset("train")

    def validation_step(self, batch, batch_idx):
        out, labels = self._step(batch)
        self.log("val_loss", out["loss"], prog_bar=True, batch_size=labels.size(0))
        self._update_metrics("val", out["prob"].squeeze(-1), labels)

    def on_validation_epoch_end(self):
        self._log_and_reset("val")

    def test_step(self, batch, batch_idx):
        out, labels = self._step(batch)
        self._update_metrics("test", out["prob"].squeeze(-1), labels)

    def on_test_epoch_end(self):
        self._log_and_reset("test", prog_bar_keys=set())

    def configure_optimizers(self):
        esm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "esm" in name and "proj" not in name:
                esm_params.append(p)
            else:
                other_params.append(p)

        groups = [{"params": other_params, "lr": self.hparams.lr}]
        if esm_params:
            groups.append({"params": esm_params, "lr": self.hparams.esm_lr, "is_esm": True})

        opt = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        w, t = self.hparams.warmup_epochs, self.hparams.max_epochs

        def lr_lambda(epoch):
            if epoch < w:
                return (epoch + 1) / w
            return 0.5 * (1 + math.cos(math.pi * (epoch - w) / max(1, t - w)))

        return [opt], [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)]


# ============================================================
# 5. Run a single ablation experiment
# ============================================================

def _dump_predictions(model, ckpt_path, eval_ds, collate_fn, args, out_dir):
    """Predict on eval_ds (order-preserving) with best weights; save peptide,label,prob CSV."""
    import pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_path and os.path.exists(str(ckpt_path)):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)
    model.eval().to(dev)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)
    probs, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}
            out = model(
                concat_ids=b["concat_ids"], concat_mask=b["concat_mask"],
                pmhc_ids=b["pmhc_ids"], pmhc_mask=b["pmhc_mask"],
                tcr_ids=b["tcr_ids"], tcr_mask=b["tcr_mask"],
                mhc_ids=b["mhc_ids"], mhc_mask=b["mhc_mask"],
                pep_ids=b["pep_ids"], pep_mask=b["pep_mask"],
                tra_ids=b["tra_ids"], tra_mask=b["tra_mask"],
                trb_ids=b["trb_ids"], trb_mask=b["trb_mask"],
                rosetta_features=b["rosetta"], rosetta_available=b["has_rosetta"],
                compute_loss=False,
            )
            probs += out["prob"].view(-1).cpu().tolist()
            ys += b["labels"].view(-1).cpu().tolist()
    peps = list(getattr(eval_ds, "_pep_seqs", [None] * len(ys)))[:len(ys)]
    path = out_dir / "test_predictions.csv"
    pd.DataFrame({"peptide": peps, "label": ys, "prob": probs}).to_csv(path, index=False)
    return path


def run_single_experiment(
    mode, train_ds, val_ds, collate_fn, esm_checkpoint,
    args, pos_weight, out_base, seed=42, test_ds=None,
):
    """Train one ablation mode, return test metrics dict."""
    L.seed_everything(seed)
    out_dir = out_base / f"mode_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    esm_model = AutoModel.from_pretrained(esm_checkpoint)

    model = AblationModel(
        esm_model=esm_model,
        esm_hidden=esm_model.config.hidden_size,
        mode=mode,
        freeze_esm=True,
        n_tune_layers=0,
        d_model=args.d_model,
        n_cross_heads=args.n_cross_heads,
        dropout=args.dropout,
        d_rosetta=args.d_rosetta,
        d_fused=args.d_fused,
        clf_hidden=args.d_fused,
        pos_weight=pos_weight,
    )

    print(f"\n{'='*65}")
    print(model._describe())
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"  Params: {tp:,} trainable / {tt:,} total ({100*tp/tt:.1f}%)")
    print(f"{'='*65}")

    lit = AblationLightning(
        model=model, lr=args.lr, esm_lr=args.esm_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        pos_weight=pos_weight,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "ckpt", monitor="val_auroc", mode="max",
        save_top_k=1, filename=f"{mode}" + "-{epoch:02d}-{val_auroc:.4f}")

    trainer = make_trainer(
        args,
        max_epochs=args.epochs,
        callbacks=[
            ckpt_cb,
            GraduatedUnfreeze(args.unfreeze_b, args.unfreeze_c),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_auroc", patience=args.patience, mode="max"),
        ],
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        gradient_clip_val=1.0,
        enable_progress_bar=args.progress_bar,
    )

    t0 = time.time()
    trainer.fit(lit, train_loader, val_loader)
    train_time = time.time() - t0

    # ---- Evaluate on the HELD-OUT TEST set (falls back to val if no split) ----
    eval_ds = test_ds if test_ds is not None else val_ds
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers)
    best = ckpt_cb.best_model_path
    test_res = trainer.test(lit, dataloaders=eval_loader, ckpt_path=best)
    metrics = test_res[0] if test_res else {}

    # Add metadata
    result = {
        "mode": mode,
        "fold": getattr(args, "fold", None),
        "mode_name": ABLATION_MODES.get(mode, mode),
        "esm_passes": MODE_CONFIG[mode]["esm_passes"],
        "crossattn": MODE_CONFIG[mode]["crossattn"],
        "rosetta": MODE_CONFIG[mode]["rosetta"],
        "chain_mode": MODE_CONFIG[mode]["chain"],
        "trainable_params": tp,
        "total_params": tt,
        "train_time_sec": round(train_time, 1),
        "best_epoch": int(ckpt_cb.best_model_path.split("epoch=")[1].split("-")[0]) if "epoch=" in str(ckpt_cb.best_model_path) else -1,
        "best_checkpoint": str(best),
    }
    result.update(metrics)

    # ---- Rich held-out-test metrics from saved predictions (rank 0 only) ----
    if getattr(trainer, "is_global_zero", True):
        try:
            pred_path = _dump_predictions(model, best, eval_ds, collate_fn, args, out_dir)
            import pandas as _pd
            from spectra.evaluation.metrics import evaluate as _evaluate
            _pdf = _pd.read_csv(pred_path)
            rich = _evaluate(_pdf["label"].values, _pdf["prob"].values,
                             _pdf["peptide"].values if "peptide" in _pdf.columns else None)
            result.update({f"test_{k}": v for k, v in rich.items()})
        except Exception as e:
            print(f"[warn] rich-metric computation failed: {e}")

    # Save individual result
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# 6. Main
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(description="SPECTRA ablation / training")
    parser.add_argument("--modes", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H"],
                        choices=list(MODE_CONFIG.keys()),
                        help="Which ablation modes to run")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV with tcr_id, tra_seq, trb_seq, peptide, mhc_seq, binding_label, "
                             "feat_* columns. Crystal structures included as domain=1 rows.")
    parser.add_argument("--esm_checkpoint", type=str,
                        default=os.environ.get("model_checkpoint", "facebook/esm2_t12_35M_UR50D"))
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_cross_heads", type=int, default=4)
    parser.add_argument("--d_rosetta", type=int, default=64)
    parser.add_argument("--d_fused", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--esm_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--crystal_weight", type=float, default=5.0)
    parser.add_argument("--unfreeze_b", type=int, default=10)
    parser.add_argument("--unfreeze_c", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--split_json", type=str, default=None,
                        help="Leak-free split JSON (folds with train/val/test row indices). "
                             "If set, evaluates on the held-out test split.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index within --split_json.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--progress_bar", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str,
                        default=os.environ.get('SPECTRA_OUT_DIR', './outputs'))
    parser.add_argument("--config", type=str, default=None,
                        help="YAML hyperparameter file; CLI flags override it.")
    add_distributed_args(parser)
    return parser


def _parse_with_config(parser, argv=None):
    """Parse argv, applying a --config YAML as defaults (CLI flags win)."""
    import yaml
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "config", None):
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)


def run(args):
    import warnings
    torch.set_float32_matmul_precision("high")   # use A40/A100 tensor cores for fp32 matmuls
    warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*")
    warnings.filterwarnings("ignore", message=r".*weights_only=False.*")
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ---- Data: leak-free split JSON (held-out test) or random val split ----
    test_ds = None
    if getattr(args, "split_json", None):
        from spectra.data.dataset_ablation import load_datasets_from_split
        train_ds, val_ds, test_ds, pos_weight = load_datasets_from_split(
            data_csv=args.data_csv,
            split_json=args.split_json,
            fold=args.fold if args.fold is not None else 0,
            crystal_weight=args.crystal_weight,
        )
    else:
        train_ds, val_ds, pos_weight = load_datasets(
            data_csv=args.data_csv,
            val_split=args.val_split,
            crystal_weight=args.crystal_weight,
            seed=args.seed,
        )

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY: {len(args.modes)} experiments")
    print(f"  Modes: {' '.join(args.modes)}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}" +
          (f" | Test: {len(test_ds)}" if test_ds is not None else ""))
    print(f"  pos_weight: {pos_weight:.2f}")
    print(f"  ESM: {args.esm_checkpoint}")
    print(f"{'='*65}")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_checkpoint)
    collate_fn = make_ablation_collate_fn(tokenizer)

    # ---- Run experiments ----
    all_results = []
    for mode in args.modes:
        print(f"\n{'#'*65}")
        print(f"  EXPERIMENT: Mode {mode} — {ABLATION_MODES.get(mode, mode)}")
        print(f"{'#'*65}")

        result = run_single_experiment(
            mode=mode,
            train_ds=train_ds,
            val_ds=val_ds,
            collate_fn=collate_fn,
            esm_checkpoint=args.esm_checkpoint,
            args=args,
            pos_weight=pos_weight,
            out_base=out_base,
            seed=args.seed,
            test_ds=test_ds,
        )
        all_results.append(result)

    # ---- Comparison table ----
    results_df = pd.DataFrame(all_results)

    # Reorder columns for readability
    col_order = [
        "mode", "mode_name", "esm_passes", "chain_mode",
        "crossattn", "rosetta",
        "test_auroc", "test_f1", "test_mcc", "test_prec", "test_rec", "test_acc",
        "trainable_params", "train_time_sec", "best_epoch",
    ]
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order + [c for c in results_df.columns if c not in col_order]]

    # Save
    csv_path = out_base / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    json_path = out_base / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Print comparison ----
    print(f"\n\n{'='*90}")
    print("  ABLATION RESULTS COMPARISON")
    print(f"{'='*90}")

    header = f"{'Mode':<6} {'Config':<28} {'ESM':<5} {'Xattn':<6} {'Ros':<5} {'AUROC':>7} {'F1':>7} {'MCC':>7} {'Params':>10} {'Time':>7}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        auroc = r.get("test_auroc", 0)
        f1 = r.get("test_f1", 0)
        mcc = r.get("test_mcc", 0)
        print(f"{r['mode']:<6} {r['mode_name']:<28} {r['esm_passes']:<5} "
              f"{'Y' if r['crossattn'] else 'N':<6} {'Y' if r['rosetta'] else 'N':<5} "
              f"{auroc:>7.4f} {f1:>7.4f} {mcc:>7.4f} "
              f"{r['trainable_params']:>10,} {r['train_time_sec']:>6.0f}s")

    print(f"\n{'='*90}")

    # ---- Print analysis ----
    if len(all_results) >= 2:
        print("\n  KEY COMPARISONS:")
        def _get(mode):
            return next((r for r in all_results if r['mode'] == mode), None)

        comparisons = [
            ("A", "B", "Positional encoding fix (1→4 chains)"),
            ("A", "G", "1-chain vs 2-chain (pMHC+TCR split)"),
            ("G", "B", "2-chain vs 4-chain"),
            ("B", "D", "Cross-attention value"),
            ("A", "F", "Rosetta value (on 1-chain)"),
            ("B", "C", "Rosetta value (on 4-chain pool)"),
            ("D", "E", "Rosetta value (on 4-chain cross-attn)"),
            ("G", "H", "Rosetta value (on 2-chain)"),
            ("F", "E", "Cross-attn worth 4x compute? (both have Rosetta)"),
            ("H", "E", "4-chain+xattn vs 2-chain? (both have Rosetta)"),
        ]

        for m1, m2, desc in comparisons:
            r1, r2 = _get(m1), _get(m2)
            if r1 and r2:
                a1 = r1.get("test_auroc", 0)
                a2 = r2.get("test_auroc", 0)
                diff = a2 - a1
                arrow = "▲" if diff > 0.005 else "▼" if diff < -0.005 else "≈"
                print(f"    {m1}→{m2}: {arrow} {diff:+.4f} AUROC  — {desc}")

    print(f"\nResults saved to: {csv_path}")
    return all_results


def main(argv=None):
    return run(_parse_with_config(build_parser(), argv))


if __name__ == "__main__":
    main()