# Training

```bash
# full multimodal model
python -m spectra.training.train --config configs/model/full_fusion.yaml

# ablations A-H
bash scripts/run_ablation.sh

# 5-seed ensemble
bash scripts/run_ensemble.sh
```
Outputs (checkpoints, metrics, predictions) are written under `SPECTRA_OUT_DIR`.
