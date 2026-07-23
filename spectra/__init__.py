"""SPECTRA: Structure, Protein-language, and Energetics via Cross-attention
for TR-pMHC Affinity.

A multimodal deep-learning framework for TCR-pMHC binding prediction that fuses:
  - a protein language model (ESM-2, LoRA-adapted),
  - 3D structure (equivariant / heterogeneous graph neural network),
  - Rosetta interface energetics,
via learned gated cross-attention.
"""
__version__ = "0.1.0"
