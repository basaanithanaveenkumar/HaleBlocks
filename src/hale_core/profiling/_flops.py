"""Analytical FLOPs and matmul-count hooks for known PyTorch layer types.

Each hook is registered as a forward_hook on a specific module. It writes into
the LayerProfile keyed by `name` inside the profiler's `_profiles` dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from hale_core.profiling._profiler import LayerProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numel(x: Any) -> int:
    import torch
    return x.numel() if isinstance(x, torch.Tensor) else 0


def _first_tensor(inp: Any) -> Any:
    import torch
    if isinstance(inp, torch.Tensor):
        return inp
    if isinstance(inp, (list, tuple)):
        for item in inp:
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


# ---------------------------------------------------------------------------
# Per-layer FLOPs factories
# ---------------------------------------------------------------------------

def make_linear_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs = 2 * tokens * in_features * out_features (one multiply-add = 2 ops)."""
    def hook(mod: nn.Linear, inp, out) -> None:
        x = _first_tensor(inp)
        if x is None:
            return
        tokens = x.numel() // x.shape[-1]
        flops = 2 * tokens * mod.in_features * mod.out_features
        profiles[name].flops += flops
        profiles[name].matmul_count += 1
    return hook


def make_conv2d_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs = 2 * B * Cout * kH * kW * Cin/groups * oH * oW."""
    def hook(mod: nn.Conv2d, inp, out) -> None:
        x = _first_tensor(inp)
        if x is None or out is None:
            return
        import torch
        if not isinstance(out, torch.Tensor):
            return
        b = x.shape[0]
        oh, ow = out.shape[-2], out.shape[-1]
        kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        flops = 2 * b * mod.out_channels * kh * kw * (mod.in_channels // mod.groups) * oh * ow
        profiles[name].flops += flops
        profiles[name].matmul_count += 1
    return hook


def make_mha_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs for nn.MultiheadAttention: QKV proj + QK scores + softmax·V + output proj."""
    def hook(mod: nn.MultiheadAttention, inp, out) -> None:
        import torch
        x = _first_tensor(inp)
        if x is None:
            return
        if x.dim() == 3:
            b, s, _ = x.shape
        elif x.dim() == 2:
            s, _ = x.shape
            b = 1
        else:
            return
        d = mod.embed_dim
        h = mod.num_heads
        dh = d // h
        # 3 separate Q/K/V projections  (or 1 in_proj + out_proj)
        qkv_flops = 3 * 2 * b * s * d * d
        # QK^T  [b, h, s, dh] x [b, h, dh, s] -> [b, h, s, s]
        attn_score_flops = 2 * b * h * s * s * dh
        # softmax(QK^T) · V  [b, h, s, s] x [b, h, s, dh] -> [b, h, s, dh]
        attn_val_flops = 2 * b * h * s * s * dh
        # output projection  [b, s, d] x [d, d]
        out_proj_flops = 2 * b * s * d * d
        profiles[name].flops += qkv_flops + attn_score_flops + attn_val_flops + out_proj_flops
        # 1 in_proj matmul + 1 attn-scores + 1 attn-values + 1 out_proj
        profiles[name].matmul_count += 4
    return hook


def make_layernorm_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs ≈ 5 × numel (mean, var, normalize, scale, bias)."""
    def hook(mod, inp, out) -> None:
        x = _first_tensor(inp)
        if x is None:
            return
        profiles[name].flops += 5 * x.numel()
    return hook


def make_embedding_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """Embedding is a lookup — 0 FLOPs, 0 matmuls; still record for completeness."""
    def hook(mod, inp, out) -> None:
        pass
    return hook


def make_gelu_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs ≈ 8 × numel (tanh-approx GELU)."""
    def hook(mod, inp, out) -> None:
        x = _first_tensor(inp)
        if x is None:
            return
        profiles[name].flops += 8 * x.numel()
    return hook


def make_relu_silu_hook(
    name: str, profiles: dict[str, LayerProfile]
) -> Callable:
    """FLOPs ≈ 1 × numel."""
    def hook(mod, inp, out) -> None:
        x = _first_tensor(inp)
        if x is None:
            return
        profiles[name].flops += x.numel()
    return hook


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HOOK_FACTORIES: list[tuple[type, Callable]] = [
    (nn.Linear,               make_linear_hook),
    (nn.Conv2d,               make_conv2d_hook),
    (nn.MultiheadAttention,   make_mha_hook),
    (nn.LayerNorm,            make_layernorm_hook),
    (nn.GELU,                 make_gelu_hook),
    (nn.ReLU,                 make_relu_silu_hook),
    (nn.SiLU,                 make_relu_silu_hook),
    (nn.Embedding,            make_embedding_hook),
]

# RMSNorm is a common custom class — match by class name
_NAME_HOOKS: dict[str, Callable] = {
    "RMSNorm": make_layernorm_hook,
}


def get_flops_hook(
    name: str,
    module: nn.Module,
    profiles: dict[str, LayerProfile],
) -> Callable | None:
    """Return a FLOPs hook for *module*, or None if unknown."""
    for cls, factory in _HOOK_FACTORIES:
        if isinstance(module, cls):
            return factory(name, profiles)
    cls_name = type(module).__name__
    if cls_name in _NAME_HOOKS:
        return _NAME_HOOKS[cls_name](name, profiles)
    return None
