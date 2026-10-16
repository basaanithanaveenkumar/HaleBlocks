"""Tensor creation, dtype casts, identities and graph bookkeeping ops: ATen → MLX."""

from __future__ import annotations

import operator

import mlx.core as mx

from hale_core.deploy.mlx_bridge.dtype_map import mlx_dtype
from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _shape(size):
    return [int(s) for s in size]


@register(name=["aten.arange.start_step", "aten.arange.start", "aten.arange.default"])
def arange(start, end=None, step=1, dtype=None, **_):
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
def full(size, fill_value, dtype=None, **_):
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
def zeros(size, dtype=None, **_):
    return mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten.ones.default"])
def ones(size, dtype=None, **_):
    return mx.ones(_shape(size), dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten.full_like.default"])
def full_like(x, fill_value, dtype=None, **_):
    return mx.full(x.shape, fill_value, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.zeros_like.default"])
def zeros_like(x, dtype=None, **_):
    return mx.zeros(x.shape, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.ones_like.default"])
def ones_like(x, dtype=None, **_):
    return mx.ones(x.shape, dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.new_zeros.default"])
def new_zeros(x, size, dtype=None, **_):
    return mx.zeros(_shape(size), dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.new_ones.default"])
def new_ones(x, size, dtype=None, **_):
    return mx.ones(_shape(size), dtype=mlx_dtype(dtype) or x.dtype)


@register(name=["aten.scalar_tensor.default"])
def scalar_tensor(value, dtype=None, **_):
    return mx.array(value, dtype=mlx_dtype(dtype) or mx.float32)


@register(name=["aten._to_copy.default", "aten.to.dtype", "aten._to_dim_order_copy.default"])
def to_copy(x, dtype=None, **_):
    return x.astype(mlx_dtype(dtype)) if dtype is not None else x


@register(name=["aten.type_as.default"])
def type_as(x, other):
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
def identity(x, *_, **__):
    return x


@register(name=["aten.sym_size.int"])
def sym_size(x, dim):
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
def no_op(*_, **__):
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
