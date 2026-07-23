"""FastAPI REST service for SPECTRA.

Run locally:
    SPECTRA_CHECKPOINT=model.pt uvicorn spectra.inference.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          -> {"status": "ok"|"no_model_loaded", "mode": ...}
    POST /predict         -> single pair -> {"probability": float}
    POST /predict/batch   -> list of pairs -> {"probabilities": [...]}
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spectra.inference.serving import load_model, score_batch

# Service state (populated by the lifespan handler at startup).
_STATE = {"model": None, "tok": None, "mode": os.environ.get("SPECTRA_MODE", "E")}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint (if provided) once at startup; release on shutdown."""
    ckpt = os.environ.get("SPECTRA_CHECKPOINT")
    if ckpt:
        _STATE["model"], _STATE["tok"] = load_model(
            ckpt, mode=_STATE["mode"], device=os.environ.get("SPECTRA_DEVICE", "cpu"))
    yield
    _STATE["model"] = None
    _STATE["tok"] = None


app = FastAPI(
    title="SPECTRA",
    version="0.1.0",
    description="TCR-pMHC binding prediction (ESM-2 + Rosetta).",
    lifespan=lifespan,
)


class Pair(BaseModel):
    tra_seq: str
    trb_seq: str
    peptide: str
    mhc_seq: str
    features: Optional[dict] = None   # optional Rosetta features, keyed feat_<name>


def _rows(pairs: List[Pair]):
    out = []
    for p in pairs:
        d = {"tra_seq": p.tra_seq, "trb_seq": p.trb_seq,
             "peptide": p.peptide, "mhc_seq": p.mhc_seq}
        if p.features:
            d.update(p.features)
        out.append(d)
    return out


def _require_model():
    if _STATE["model"] is None:
        raise HTTPException(status_code=503,
                            detail="No model loaded — set SPECTRA_CHECKPOINT and restart.")


def _device():
    return os.environ.get("SPECTRA_DEVICE", "cpu")


@app.get("/health")
def health():
    return {"status": "ok" if _STATE["model"] is not None else "no_model_loaded",
            "mode": _STATE["mode"]}


@app.post("/predict")
def predict(pair: Pair):
    _require_model()
    probs = score_batch(_STATE["model"], _STATE["tok"], _rows([pair]),
                        mode=_STATE["mode"], device=_device())
    return {"probability": probs[0]}


@app.post("/predict/batch")
def predict_batch(pairs: List[Pair]):
    _require_model()
    probs = score_batch(_STATE["model"], _STATE["tok"], _rows(pairs),
                        mode=_STATE["mode"], device=_device())
    return {"probabilities": probs}
