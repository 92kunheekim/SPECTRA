"""ESM-2 sequence encoders.

- ESMSequenceEncoder : flagship encoder (masked-mean pool, LoRA/partial tune).
- ESMChainEncoder    : per-chain encoder used by the ablation model.
"""
from spectra.models.spectra_model import ESMSequenceEncoder
from spectra.models.ablation_model import ESMChainEncoder

__all__ = ["ESMSequenceEncoder", "ESMChainEncoder"]
