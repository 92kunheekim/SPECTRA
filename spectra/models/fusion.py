"""Gated multimodal fusion.

- VectorGatedFusion : per-dimension gate (flagship).
- GatedFusion       : seq + Rosetta gate with missing-modality fallback (ablation).
"""
from spectra.models.spectra_model import VectorGatedFusion
from spectra.models.ablation_model import GatedFusion

__all__ = ["VectorGatedFusion", "GatedFusion"]
