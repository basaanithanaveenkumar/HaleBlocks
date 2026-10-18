"""Tensor creation, dtype casts, identities and graph bookkeeping ops: ATen → MLX."""

from __future__ import annotations

import operator
from typing import Any

import mlx.core as mx

from hale_core.deploy.mlx_bridge.dtype_map import mlx_dtype
from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _shape(size):
    return [int(s) for s in size]


@register(name=["aten.arange.start_step", "aten.arange.start", "aten.arange.default"])
def arange(start: Any, end: Any = None, step: Any = 1, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.arange.start_step``, ``aten.arange.start``, ``aten.arange.default`` to MLX."""
    if end is None:
        start, end = 0, start
    return mx.arange(
        start,
        end,
        step,
        dtype=mlx_dtype(dtype)
        or (mx.int64 if all(isinstance(v, int) for v in (start, end, step)) else mx.float32),
    )


@register(name=["aten.full.default"])
def full(size: Any, fill_value: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.full.default`` to MLX."""
    return mx.full(
        _shape(size),
        fill_value,
        dtype=mlx_dtype(dtype)
        or (
            mx.bool_
            if isinstance(fill_value, bool)
            else (mx.int64 if isinstance(fill_value, int) else mx.float32)
        ),
    )


@register(name=["aten.zeros.default"])
def zeros(size: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.zeros.default`` to MLX: ``mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or
    mx.float32)``.
    """
    return mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten.ones.default"])
def ones(size: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.ones.default`` to MLX: ``mx.ones(_shape(size), dtype=mlx_dtype(dtype) or
    mx.float32)``.
    """
    return mx.ones(_shape(size), dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten.full_like.default"])
def full_like(x: Any, fill_value: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.full_like.default`` to MLX."""
    return mx.full(x.shape, fill_value, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.zeros_like.default"])
def zeros_like(x: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.zeros_like.default`` to MLX: ``mx.zeros(x.shape, dtype=mlx_dtype(dtype) or
    x.dtype)``.
    """
    return mx.zeros(x.shape, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.ones_like.default"])
def ones_like(x: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.ones_like.default`` to MLX: ``mx.ones(x.shape, dtype=mlx_dtype(dtype) or
    x.dtype)``.
    """
    return mx.ones(x.shape, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.new_zeros.default"])
def new_zeros(x: Any, size: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.new_zeros.default`` to MLX: ``mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or
    x.dtype)``.
    """
    return mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.new_ones.default"])
def new_ones(x: Any, size: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.new_ones.default`` to MLX: ``mx.ones(_shape(size), dtype=mlx_dtype(dtype) or
    x.dtype)``.
    """
    return mx.ones(_shape(size), dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.scalar_tensor.default"])
def scalar_tensor(value: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten.scalar_tensor.default`` to MLX: ``mx.array(value, dtype=mlx_dtype(dtype) or
    mx.float32)``.
    """
    return mx.array(value, dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten._to_copy.default", "aten.to.dtype", "aten._to_dim_order_copy.default"])
def to_copy(x: Any, dtype: Any = None, **_: Any) -> Any:
    """Lower ``aten._to_copy.default``, ``aten.to.dtype``, ``aten._to_dim_order_copy.default`` to
    MLX:
    ``x.astype(mlx_dtype(dtype)) if dtype is not None else x``.
    """
    return x.astype(mlx_dtype(dtype)) if dtype is not None else x


@register(name=["aten.type_as.default"])
def type_as(x: Any, other: Any) -> Any:
    """Lower ``aten.type_as.default`` to MLX: ``x.astype(other.dtype)``."""
    return x.astype(other.dtype)


@register(
    name=[
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.alias.default",
        "aten.detach.default",
        "aten.lift_fresh_copy.default",
        "aten.dropout.default",
        "aten.alias_copy.default",
    ]
)
def identity(x: Any, *_: Any, **__: Any) -> Any:
    """Lower ``aten.clone.default``, ``aten.contiguous.default``, ``aten.alias.default``,
    ``aten.detach.default``, ``aten.lift_fresh_copy.default``, ``aten.dropout.default``,
    ``aten.alias_copy.default`` to MLX: ``x``.
    """
    return x


@register(name=["aten.sym_size.int"])
def sym_size(x: Any, dim: Any) -> Any:
    """Lower ``aten.sym_size.int`` to MLX: ``x.shape[dim]``."""
    return x.shape[dim]


@register(
    name=[
        "aten._assert_tensor_metadata.default",
        "aten._assert_scalar.default",
        "aten.sym_constrain_range_for_size.default",
        "aten._assert_async.msg",
        "aten.sym_constrain_range.default",
    ]
)
def no_op(*_: Any, **__: Any) -> Any:
    """Lower ``aten._assert_tensor_metadata.default``, ``aten._assert_scalar.default``,
    ``aten.sym_constrain_range_for_size.default``, ``aten._assert_async.msg``,
    ``aten.sym_constrain_range.default`` to MLX: ``None``.
    """
    return None


# Python-level ops on symbolic ints and tuples (shape arithmetic, multi-output selection).
register(name=["operator.getitem"], module=operator.getitem)
register(name=["operator.add"], module=operator.add)
register(name=["operator.sub"], module=operator.sub)
register(name=["operator.mul"], module=operator.mul)
register(name=["operator.floordiv"], module=operator.floordiv)
register(name=["operator.truediv"], module=operator.truediv)
register(name=["operator.mod"], module=operator.mod)
register(name=["operator.neg"], module=operator.neg)
register(name=["operator.eq"], module=operator.eq)
register(name=["operator.ne"], module=operator.ne)
register(name=["operator.lt"], module=operator.lt)
register(name=["operator.le"], module=operator.le)
register(name=["operator.gt"], module=operator.gt)
register(name=["operator.ge"], module=operator.ge)
register(name=["operator.and_"], module=operator.and_)
register(name=["operator.or_"], module=operator.or_)
register(name=["torch.sym_min"], module=min)
register(name=["torch.sym_max"], module=max)
register(name=["torch.sym_not"], module=operator.not_)
