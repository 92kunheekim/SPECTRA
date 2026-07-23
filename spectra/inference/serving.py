"""Model loading + scoring shared by the CLI and the REST API."""
from __future__ import annotations
import os
from typing import List, Optional
import torch

from spectra.models.ablation_model import AblationModel, ROSETTA_FEATURE_NAMES


def load_model(checkpoint: str, mode: str = "E",
               esm_checkpoint: Optional[str] = None, device: str = "cpu"):
    """Load ESM-2 backbone + AblationModel weights for inference."""
    from transformers import AutoTokenizer, AutoModel
    esm_checkpoint = esm_checkpoint or os.environ.get(
        "SPECTRA_ESM_CKPT", "facebook/esm2_t12_35M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained(esm_checkpoint)
    esm = AutoModel.from_pretrained(esm_checkpoint)
    esm_hidden = esm.config.hidden_size

    model = AblationModel(esm_model=esm, esm_hidden=esm_hidden, mode=mode, freeze_esm=True)
    state = torch.load(checkpoint, map_location=device)
    state = state.get("state_dict", state)
    # tolerate a Lightning "model." prefix
    state = { k.replace("model.", "", 1): v for k, v in state.items() }
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, tokenizer


def _tok(tokenizer, seqs: List[str], device: str):
    enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device).bool()


@torch.no_grad()
def score_batch(model, tokenizer, rows, mode: str, device: str = "cpu") -> List[float]:
    """rows: list of dicts with tra_seq, trb_seq, peptide, mhc_seq (+ feat_* optional)."""
    mhc = [r["mhc_seq"] for r in rows]; pep = [r["peptide"] for r in rows]
    tra = [r["tra_seq"] for r in rows]; trb = [r["trb_seq"] for r in rows]
    kw = {}
    chain = model.cfg["chain"]
    if chain == "1chain":
        concat = [f"{m}.{p}.{a}.{b}" for m, p, a, b in zip(mhc, pep, tra, trb)]
        kw["concat_ids"], kw["concat_mask"] = _tok(tokenizer, concat, device)
    elif chain == "2chain":
        kw["pmhc_ids"], kw["pmhc_mask"] = _tok(tokenizer, [f"{m}.{p}" for m, p in zip(mhc, pep)], device)
        kw["tcr_ids"], kw["tcr_mask"] = _tok(tokenizer, [f"{a}.{b}" for a, b in zip(tra, trb)], device)
    else:  # 4chain
        kw["mhc_ids"], kw["mhc_mask"] = _tok(tokenizer, mhc, device)
        kw["pep_ids"], kw["pep_mask"] = _tok(tokenizer, pep, device)
        kw["tra_ids"], kw["tra_mask"] = _tok(tokenizer, tra, device)
        kw["trb_ids"], kw["trb_mask"] = _tok(tokenizer, trb, device)
    if model.cfg["rosetta"] and all(ROSETTA_FEATURE_NAMES[0] and (f"feat_{n}" in r) for n in ROSETTA_FEATURE_NAMES for r in rows):
        feats = torch.tensor([[float(r[f"feat_{n}"]) for n in ROSETTA_FEATURE_NAMES] for r in rows],
                             dtype=torch.float32, device=device)
        kw["rosetta_features"] = feats
        kw["rosetta_available"] = torch.ones(len(rows), dtype=torch.bool, device=device)
    out = model(**kw)
    return out["prob"].view(-1).cpu().tolist()


def score_dataframe(model, tokenizer, df, mode: str, device="cpu", batch_size=64) -> List[float]:
    rows = df.to_dict("records")
    probs: List[float] = []
    for i in range(0, len(rows), batch_size):
        probs.extend(score_batch(model, tokenizer, rows[i:i + batch_size], mode, device))
    return probs
