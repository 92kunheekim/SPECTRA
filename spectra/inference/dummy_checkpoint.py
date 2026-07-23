"""Create an UNTRAINED checkpoint to smoke-test deployment before HPC training.

The resulting model outputs meaningless probabilities — it exists only to
exercise the load -> tokenize -> forward -> response path end to end so the
container can be validated before a real checkpoint exists.

    python -m spectra.inference.dummy_checkpoint --mode E --out model.pt
"""
from __future__ import annotations
import argparse
import os
import torch

from spectra.models.ablation_model import AblationModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="E")
    ap.add_argument("--esm_checkpoint",
                    default=os.environ.get("SPECTRA_ESM_CKPT", "facebook/esm2_t6_8M_UR50D"))
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    from transformers import AutoModel
    esm = AutoModel.from_pretrained(args.esm_checkpoint)
    model = AblationModel(esm_model=esm, esm_hidden=esm.config.hidden_size,
                          mode=args.mode, freeze_esm=True)
    torch.save(model.state_dict(), args.out)
    print(f"[dummy_checkpoint] wrote UNTRAINED {args.mode} checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
