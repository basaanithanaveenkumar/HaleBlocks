"""dtype translation between torch and MLX."""

from __future__ import annotations

from typing import Any

import torch

_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float32",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool_",
}


def mlx_dtype(dtype: Any) -> Any:
    """torch dtype (or ``None``) → mlx dtype (``None`` passes through). float64 maps to float32
    because Metal has no fp64."""
    if dtype is None:
        return None
    import mlx.core as mx

    if isinstance(dtype, torch.dtype):
        return getattr(mx, _NAMES[dtype])
    if isinstance(dtype, str):
        return getattr(mx, {"bool": "bool_"}.get(dtype, dtype))
    return dtype


def to_mlx(tensor: torch.Tensor, dtype: Any = None) -> Any:
    """Copy a torch tensor into an mlx array (bf16 goes through fp32: numpy has no bf16)."""
    import mlx.core as mx

    t = tensor.detach().cpu()
    if t.dtype in (torch.bfloat16, torch.float64):
        t = t.to(torch.float32)
    array = mx.array(t.numpy())
    if tensor.dtype == torch.bfloat16 and dtype is None:
        array = array.astype(mx.bfloat16)
    return array.astype(mlx_dtype(dtype)) if dtype is not None else array
