from __future__ import annotations

from typing import TYPE_CHECKING

from torch import float32, float64

if TYPE_CHECKING:
    from torch import dtype
    from torch.nn import Module


def get_model_dtype(model: Module) -> dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == float64:
        return "float64"
    if mode_dtype == float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")
