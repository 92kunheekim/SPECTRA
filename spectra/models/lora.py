"""LoRA modules — re-exported from the flagship model.

Chain-specific low-rank adapters for parameter-efficient ESM-2 fine-tuning.
"""
from spectra.models.spectra_model import LoRALinear, LoRAAdapter

__all__ = ["LoRALinear", "LoRAAdapter"]
