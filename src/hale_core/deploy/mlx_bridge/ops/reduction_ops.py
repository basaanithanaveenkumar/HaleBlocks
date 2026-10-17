"""Reductions, softmax and scans: ATen → MLX."""

from __future__ import annotations

import mlx.core as mx

from hale_core.deploy.mlx_bridge.dtype_map import mlx_dtype
from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _axes(dim):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return None
    return list(dim) if isinstance(dim, (list, tuple)) else dim


@register(name=["aten.sum.dim_IntList", "aten.sum.default"])
def sum_(x, dim=None, keepdim=False, dtype=None):
    """Lower ``aten.sum.dim_IntList``, ``aten.sum.default`` to MLX."""
    out = mx.sum(x, axis=_axes(dim), keepdims=keepdim)
    return out.astype(mlx_dtype(dtype)) if dtype is not None else out


@register(name=["aten.mean.dim", "aten.mean.default"])
def mean(x, dim=None, keepdim=False, dtype=None):
    """Lower ``aten.mean.dim``, ``aten.mean.default`` to MLX."""
    out = mx.mean(x, axis=_axes(dim), keepdims=keepdim)
    return out.astype(mlx_dtype(dtype)) if dtype is not None else out


@register(name=["aten.var.correction", "aten.var.dim"])
def var(x, dim=None, correction=None, keepdim=False, unbiased=None):
    """Lower ``aten.var.correction``, ``aten.var.dim`` to MLX."""
    ddof = correction if correction is not None else (0 if unbiased is False else 1)
    return mx.var(x, axis=_axes(dim), keepdims=keepdim, ddof=int(ddof))


@register(name=["aten.amax.default"])
def amax(x, dim=(), keepdim=False):
    """Lower ``aten.amax.default`` to MLX: ``mx.max(x, axis=_axes(dim), keepdims=keepdim)``."""
    return mx.max(x, axis=_axes(dim), keepdims=keepdim)


@register(name=["aten.amin.default"])
def amin(x, dim=(), keepdim=False):
    """Lower ``aten.amin.default`` to MLX: ``mx.min(x, axis=_axes(dim), keepdims=keepdim)``."""
    return mx.min(x, axis=_axes(dim), keepdims=keepdim)


@register(name=["aten.max.dim"])
def max_dim(x, dim, keepdim=False):
    """Lower ``aten.max.dim`` to MLX."""
    return mx.max(x, axis=dim, keepdims=keepdim), mx.argmax(x, axis=dim, keepdims=keepdim)


@register(name=["aten.min.dim"])
def min_dim(x, dim, keepdim=False):
    """Lower ``aten.min.dim`` to MLX."""
    return mx.min(x, axis=dim, keepdims=keepdim), mx.argmin(x, axis=dim, keepdims=keepdim)


@register(name=["aten.argmax.default"])
def argmax(x, dim=None, keepdim=False):
    """Lower ``aten.argmax.default`` to MLX: ``mx.argmax(x, axis=dim, keepdims=keepdim)``."""
    return mx.argmax(x, axis=dim, keepdims=keepdim)


@register(name=["aten.any.dim", "aten.any.default", "aten.any.dims"])
def any_(x, dim=None, keepdim=False):
    """Lower ``aten.any.dim``, ``aten.any.default``, ``aten.any.dims`` to MLX: ``mx.any(x,
    axis=_axes(dim), keepdims=keepdim)``.
    """
    return mx.any(x, axis=_axes(dim), keepdims=keepdim)


@register(name=["aten.cumsum.default"])
def cumsum(x, dim, dtype=None):
    """Lower ``aten.cumsum.default`` to MLX."""
    out = mx.cumsum(x, axis=dim)
    return out.astype(mlx_dtype(dtype)) if dtype is not None else out


@register(name=["aten._softmax.default"])
def softmax_(x, dim, half_to_float=False):
    """Lower ``aten._softmax.default`` to MLX: ``mx.softmax(x, axis=dim, precise=True)``."""
    return mx.softmax(x, axis=dim, precise=True)


@register(name=["aten.softmax.int"])
def softmax(x, dim, dtype=None):
    """Lower ``aten.softmax.int`` to MLX."""
    out = mx.softmax(x, axis=dim, precise=True)
    return out.astype(mlx_dtype(dtype)) if dtype is not None else out


@register(name=["aten._log_softmax.default", "aten.log_softmax.int"])
def log_softmax(x, dim, half_to_float=False):
    """Lower ``aten._log_softmax.default``, ``aten.log_softmax.int`` to MLX: ``x - mx.logsumexp(x,
    axis=dim, keepdims=True)``.
    """
    return x - mx.logsumexp(x, axis=dim, keepdims=True)


@register(name=["aten.logsumexp.default"])
def logsumexp(x, dim, keepdim=False):
    """Lower ``aten.logsumexp.default`` to MLX: ``mx.logsumexp(x, axis=_axes(dim),
    keepdims=keepdim)``.
    """
    return mx.logsumexp(x, axis=_axes(dim), keepdims=keepdim)


@register(name=["aten.linalg_vector_norm.default"])
def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):  # noqa: A002 - ATen names
    """Lower ``aten.linalg_vector_norm.default`` to MLX."""
    axes = _axes(dim)
    if ord == 2:
        out = mx.sqrt(mx.sum(x * x, axis=axes, keepdims=keepdim))
    elif ord == 1:
        out = mx.sum(mx.abs(x), axis=axes, keepdims=keepdim)
    elif ord == float("inf"):
        out = mx.max(mx.abs(x), axis=axes, keepdims=keepdim)
    else:
        out = mx.power(mx.sum(mx.power(mx.abs(x), ord), axis=axes, keepdims=keepdim), 1.0 / ord)
    return out.astype(mlx_dtype(dtype)) if dtype is not None else out
