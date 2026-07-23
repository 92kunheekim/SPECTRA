"""Cross-attention modules (re-exported from the flagship model)."""
from spectra.models.spectra_model import (
    CrossAttention,
    BidirectionalStructureSequenceCrossAttention,
)

__all__ = ["CrossAttention", "BidirectionalStructureSequenceCrossAttention"]
