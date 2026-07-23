"""Export a trained model for serving.

Saves a clean, framework-portable bundle: weights + the config needed to
rebuild the model. (Full-model TorchScript of ESM-2 is brittle; we export
weights + config, which the serving loader consumes.)

    python -m spectra.inference.export --checkpoint ckpt.pt --mode E --out serving_bundle.pt
"""
from __future__ import annotations
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="E")
    ap.add_argument("--esm_checkpoint", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--out", default="serving_bundle.pt")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    torch.save({"state_dict": state, "mode": args.mode,
                "esm_checkpoint": args.esm_checkpoint,
                "format": "spectra-serving-v1"}, args.out)
    print(f"Wrote serving bundle -> {args.out}")


if __name__ == "__main__":
    main()
