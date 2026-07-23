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

from fastapi import FastAPI, HTTPException, Response
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


# ----------------------------------------------------------------------------
# Lightweight in-process metrics (zero external deps): a rolling window of
# request latencies + status counters, exposed as JSON (/stats) and Prometheus
# text (/metrics) so latency percentiles and throughput are observable in prod.
# ----------------------------------------------------------------------------
import time as _time
from collections import deque, Counter
from threading import Lock

_METRICS = {
    "start": _time.time(),
    "lat_ms": deque(maxlen=2000),
    "total": 0,
    "errors": 0,
    "by_status": Counter(),
    "lock": Lock(),
}


def _percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p / 100.0
    f = int(k)
    if f + 1 >= len(sorted_xs):
        return sorted_xs[f]
    return sorted_xs[f] + (sorted_xs[f + 1] - sorted_xs[f]) * (k - f)


@app.middleware("http")
async def _track_metrics(request, call_next):
    t0 = _time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        dt = (_time.perf_counter() - t0) * 1000.0
        if request.url.path not in ("/metrics", "/stats", "/health", "/docs", "/openapi.json"):
            with _METRICS["lock"]:
                _METRICS["lat_ms"].append(dt)
                _METRICS["total"] += 1
                _METRICS["by_status"][status] += 1
                if status >= 500:
                    _METRICS["errors"] += 1


def _snapshot():
    with _METRICS["lock"]:
        xs = sorted(_METRICS["lat_ms"])
        total, errors = _METRICS["total"], _METRICS["errors"]
        by_status = dict(_METRICS["by_status"])
        uptime = _time.time() - _METRICS["start"]
    return xs, total, errors, by_status, uptime


@app.get("/stats")
def stats():
    """Human-readable metrics: throughput and exact latency percentiles (ms)."""
    xs, total, errors, by_status, uptime = _snapshot()
    return {
        "uptime_s": round(uptime, 1),
        "requests_total": total,
        "errors_total": errors,
        "requests_per_s": round(total / uptime, 2) if uptime > 0 else 0.0,
        "latency_ms": {
            "p50": round(_percentile(xs, 50), 1),
            "p95": round(_percentile(xs, 95), 1),
            "p99": round(_percentile(xs, 99), 1),
            "mean": round(sum(xs) / len(xs), 1) if xs else 0.0,
            "max": round(max(xs), 1) if xs else 0.0,
            "window": len(xs),
        },
        "by_status": by_status,
    }


@app.get("/metrics")
def metrics():
    """Prometheus text exposition (scrapeable)."""
    xs, total, errors, by_status, _ = _snapshot()
    lines = [
        "# HELP spectra_requests_total Total prediction requests served.",
        "# TYPE spectra_requests_total counter",
        f"spectra_requests_total {total}",
        "# HELP spectra_request_errors_total Total 5xx responses.",
        "# TYPE spectra_request_errors_total counter",
        f"spectra_request_errors_total {errors}",
        "# HELP spectra_request_latency_ms Request latency quantiles (ms).",
        "# TYPE spectra_request_latency_ms summary",
        f'spectra_request_latency_ms{{quantile="0.5"}} {_percentile(xs, 50):.1f}',
        f'spectra_request_latency_ms{{quantile="0.95"}} {_percentile(xs, 95):.1f}',
        f'spectra_request_latency_ms{{quantile="0.99"}} {_percentile(xs, 99):.1f}',
        f"spectra_request_latency_ms_count {len(xs)}",
    ]
    for st, n in sorted(by_status.items()):
        lines.append(f'spectra_responses_total{{status="{st}"}} {n}')
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
