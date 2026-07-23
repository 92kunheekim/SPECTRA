---
title: SPECTRA TCR-pMHC Binding API
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
---

# SPECTRA — TCR–pMHC binding prediction (live API)

A containerized FastAPI service for the SPECTRA mode-E model (ESM-2 + Rosetta
interface energetics). Interactive docs at **`/docs`**; endpoints:

- `GET /health` — liveness + loaded mode
- `POST /predict` — one TCR–pMHC pair → `{"probability": ...}`
- `POST /predict/batch` — many pairs → `{"probabilities": [...]}`

**Note on weights.** This public demo serves the SPECTRA *architecture* with an
**untrained smoke checkpoint**, so the returned probabilities are placeholders —
its purpose is to demonstrate the live serving path (container → API → inference).
The trained 15-fold-CV weights (AUROC 0.825) are swapped in by replacing the
build-time `dummy_checkpoint` step with a real `model.pt` (git-lfs) and
pointing `SPECTRA_CHECKPOINT` at it.

Source: https://github.com/92kunheekim/SPECTRA
