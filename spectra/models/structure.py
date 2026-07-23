"""Structural encoder — EGNN (default).

E(n)-equivariant GNN over the TCR-pMHC complex graph, with enriched node
features (one-hot AA + hbond acc/don + sidechain vector + chain-id) and
optional ESM-embedding node features.
"""
from spectra.models.spectra_model import StructureEGNN

__all__ = ["StructureEGNN"]
