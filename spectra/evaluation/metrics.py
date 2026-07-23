"""Binary classification metrics for TCR-pMHC binding.

Reports the metrics that matter under heavy class imbalance and for
peptide-level generalization:
  AUROC, AUPRC (average precision), AUC0.1 (McClish-standardized partial AUC in
  the low-FPR regime), MCC, F1, precision/recall/accuracy, and per-peptide
  macro-AUROC.
"""
from __future__ import annotations
import numpy as np

__all__ = ["evaluate", "partial_auc", "per_peptide_auroc"]


def _safe_auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def partial_auc(y_true, y_score, max_fpr=0.1):
    """McClish-standardized partial AUC over FPR in [0, max_fpr] (0.5=random, 1=perfect)."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def per_peptide_auroc(y_true, y_score, peptides, min_pos=1, min_neg=1):
    """Macro-average of per-peptide AUROC over peptides with both classes present."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); peptides = np.asarray(peptides)
    per = {}
    for pep in np.unique(peptides):
        m = peptides == pep
        yy = y_true[m]
        if (yy == 1).sum() >= min_pos and (yy == 0).sum() >= min_neg:
            per[str(pep)] = _safe_auroc(yy, y_score[m])
    vals = [v for v in per.values() if v == v]
    return {
        "per_peptide_auroc_macro": float(np.mean(vals)) if vals else float("nan"),
        "n_peptides_scored": len(vals),
        "per_peptide": per,
    }


def evaluate(y_true, y_score, peptides=None, threshold=0.5):
    """Full metric suite as a flat dict (per-peptide summarized to a macro value)."""
    from sklearn.metrics import (
        average_precision_score, matthews_corrcoef, f1_score,
        precision_score, recall_score, accuracy_score,
    )
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    two = len(np.unique(y_true)) > 1
    out = {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": float(average_precision_score(y_true, y_score)) if two else float("nan"),
        "auc01": partial_auc(y_true, y_score, 0.1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if two else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
    }
    if peptides is not None:
        pp = per_peptide_auroc(y_true, y_score, peptides)
        out["per_peptide_auroc_macro"] = pp["per_peptide_auroc_macro"]
        out["n_peptides_scored"] = pp["n_peptides_scored"]
    return out
