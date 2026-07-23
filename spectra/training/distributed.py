"""Distributed / multi-GPU training helpers (PyTorch Lightning).

Turns single-GPU training into multi-GPU (or multi-node) DDP by config only.
`add_distributed_args` registers the flags; `make_trainer` builds a configured
Lightning Trainer from them. Robust to args objects that lack some fields.
"""
from __future__ import annotations
import argparse
import os


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("distributed / hardware")
    g.add_argument("--devices", default="auto",
                   help="GPUs per node, or 'auto' (default). e.g. 4")
    g.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
    g.add_argument("--strategy", default="auto",
                   help="'auto' | 'ddp' | 'ddp_find_unused_parameters_true' | 'fsdp'")
    g.add_argument("--precision", default="16-mixed",
                   help="'32-true' | '16-mixed' | 'bf16-mixed'")
    g.add_argument("--accumulate-grad-batches", type=int, default=1)
    g.add_argument("--sync-batchnorm", action="store_true",
                   help="Sync BatchNorm across GPUs (recommended for multi-GPU).")
    return parser


def make_trainer(args, callbacks=None, logger=None, **overrides):
    """Build a Lightning Trainer configured for 1..N GPUs / nodes.

    CPU and single GPU work unchanged (strategy stays 'auto'); pass
    --devices 4 --strategy ddp to scale out. num_nodes>1 enables multi-node DDP.
    `max_epochs` may be passed as an override or inferred from args.epochs.
    """
    import pytorch_lightning as L

    devices = getattr(args, "devices", "auto")
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    strategy = getattr(args, "strategy", "auto")
    if strategy == "auto" and isinstance(devices, int) and devices > 1:
        strategy = "ddp"

    max_epochs = overrides.pop("max_epochs", None)
    if max_epochs is None:
        max_epochs = getattr(args, "max_epochs", None) or getattr(args, "epochs", None) or 40

    cfg = dict(
        accelerator="auto",
        devices=devices,
        num_nodes=getattr(args, "num_nodes", 1),
        strategy=strategy,
        precision=getattr(args, "precision", "16-mixed"),
        accumulate_grad_batches=getattr(args, "accumulate_grad_batches", 1),
        sync_batchnorm=bool(getattr(args, "sync_batchnorm", False)),
        max_epochs=max_epochs,
        log_every_n_steps=25,
        callbacks=callbacks or [],
        logger=logger,
    )
    cfg.update(overrides)
    return L.Trainer(**cfg)
