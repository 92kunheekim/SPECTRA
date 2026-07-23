"""Train the full SPECTRA model (default: mode E — 4-chain cross-attention + Rosetta).

Thin wrapper over the ablation runner: same data pipeline, model, and Lightning
loop, defaulting to the full model. Scales to multi-GPU via --devices/--strategy.

    python -m spectra.training.train --data_csv data/training.csv
    python -m spectra.training.train --data_csv data/training.csv --devices 4 --strategy ddp
    python -m spectra.training.train --config configs/model/full_fusion.yaml --data_csv data/training.csv
"""
from spectra.training.ablation import build_parser, _parse_with_config, run


def main(argv=None):
    parser = build_parser()
    parser.set_defaults(modes=["E"])   # full model by default
    return run(_parse_with_config(parser, argv))


if __name__ == "__main__":
    main()
