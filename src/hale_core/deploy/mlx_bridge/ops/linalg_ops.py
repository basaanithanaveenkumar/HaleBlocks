"""Matrix products and attention ATen ops → MLX."""

from __future__ import annotations

import math

import mlx.core as mx

from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


@register(name=["aten.linear.default"])
def linear(x, weight, bias=None):
    """Lower ``aten.linear.default`` to MLX."""
    y = x @ weight.T
    return y + bias if bias is not None else y


@register(name=["hale.quantized_linear"])
def quantized_linear(x, w_q, scales, biases, bias=None, *, group_size: int, bits: int):
    """Weight-only quantized linear produced by the converter's quantization pass."""
    y = mx.quantized_matmul(
        x, w_q, scales, biases, transpose=True, group_size=group_size, bits=bits
    )
    return y + bias if bias is not None else y


@register(name=["aten.mm.default", "aten.bmm.default", "aten.matmul.default"])
def matmul(a, b):
    """Lower ``aten.mm.default``, ``aten.bmm.default``, ``aten.matmul.default`` to MLX: ``a @ b``.
    """
    return a @ b


@register(name=["aten.addmm.default"])
def addmm(bias, a, b, beta=1, alpha=1):
    """Lower ``aten.addmm.default`` to MLX."""
    out = a @ b
    if alpha != 1:
        out = out * alpha
    return out + (bias * beta if beta != 1 else bias)


@register(name=["aten.baddbmm.default"])
def baddbmm(bias, a, b, beta=1, alpha=1):
    """Lower ``aten.baddbmm.default`` to MLX: ``addmm(bias, a, b, beta, alpha)``."""
    return addmm(bias, a, b, beta, alpha)


@register(name=["aten.t.default", "aten.numpy_T.default"])
def t(x):
    """Lower ``aten.t.default``, ``aten.numpy_T.default`` to MLX: ``x.T if x.ndim == 2 else x``."""
    return x.T if x.ndim == 2 else x


@register(name=["aten.scaled_dot_product_attention.default"])
def scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """``mx.fast.scaled_dot_product_attention`` (fused Metal kernel, native GQA).

    torch's ``is_causal`` aligns the mask top-left (``tril`` with diagonal 0); it is built
    explicitly so the semantics match for ``Lq != Lk`` too. Dropout is an eval-time no-op.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    mask = attn_mask
    if is_causal:
        lq, lk = q.shape[-2], k.shape[-2]
        mask = mx.tril(mx.ones((lq, lk), dtype=mx.bool_))
    if mask is not None and mask.dtype != mx.bool_ and mask.dtype != q.dtype:
        mask = mask.astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
