"""Element-wise ATen ops → MLX (arithmetic, activations, comparisons, logic).

Converters receive resolved arguments (mlx arrays, Python scalars, lists) in ATen's argument
order. Python scalars broadcast natively in MLX, so ``x + 1`` and ``1 + x`` need no wrapping.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _is_array(x) -> bool:
    return isinstance(x, mx.array)


@register(name=["aten.add.Tensor", "aten.add.Scalar"])
def add(a: Any, b: Any, alpha: Any = 1) -> Any:
    """Lower ``aten.add.Tensor``, ``aten.add.Scalar`` to MLX: ``a + (b * alpha if alpha != 1 else
    b)``.
    """
    return a + (b * alpha if alpha != 1 else b)


@register(name=["aten.sub.Tensor", "aten.sub.Scalar"])
def sub(a: Any, b: Any, alpha: Any = 1) -> Any:
    """Lower ``aten.sub.Tensor``, ``aten.sub.Scalar`` to MLX: ``a - (b * alpha if alpha != 1 else
    b)``.
    """
    return a - (b * alpha if alpha != 1 else b)


@register(name=["aten.rsub.Scalar", "aten.rsub.Tensor"])
def rsub(a: Any, b: Any, alpha: Any = 1) -> Any:
    """Lower ``aten.rsub.Scalar``, ``aten.rsub.Tensor`` to MLX: ``b - (a * alpha if alpha != 1 else
    a)``.
    """
    return b - (a * alpha if alpha != 1 else a)


@register(name=["aten.mul.Tensor", "aten.mul.Scalar"])
def mul(a: Any, b: Any) -> Any:
    """Lower ``aten.mul.Tensor``, ``aten.mul.Scalar`` to MLX: ``a * b``."""
    return a * b


@register(name=["aten.div.Tensor", "aten.div.Scalar"])
def div(a: Any, b: Any) -> Any:
    """Lower ``aten.div.Tensor``, ``aten.div.Scalar`` to MLX: ``a / b``."""
    return a / b


@register(name=["aten.div.Tensor_mode", "aten.div.Scalar_mode"])
def div_mode(a: Any, b: Any, rounding_mode: Any = None) -> Any:
    """Lower ``aten.div.Tensor_mode``, ``aten.div.Scalar_mode`` to MLX."""
    if rounding_mode == "floor":
        return mx.floor_divide(a, b)
    q = a / b
    if rounding_mode == "trunc":
        return mx.where(q < 0, mx.ceil(q), mx.floor(q))
    return q


@register(name=["aten.floor_divide.default"])
def floor_divide(a: Any, b: Any) -> Any:
    """Lower ``aten.floor_divide.default`` to MLX: ``mx.floor_divide(a, b)``."""
    return mx.floor_divide(a, b)


@register(name=["aten.remainder.Tensor", "aten.remainder.Scalar"])
def remainder(a: Any, b: Any) -> Any:
    """Lower ``aten.remainder.Tensor``, ``aten.remainder.Scalar`` to MLX: ``mx.remainder(a, b)``."""
    return mx.remainder(a, b)


@register(name=["aten.pow.Tensor_Scalar", "aten.pow.Tensor_Tensor", "aten.pow.Scalar"])
def power(a: Any, b: Any) -> Any:
    """Lower ``aten.pow.Tensor_Scalar``, ``aten.pow.Tensor_Tensor``, ``aten.pow.Scalar`` to MLX."""
    if not _is_array(a):
        return mx.power(mx.array(a, dtype=b.dtype), b)
    return mx.power(a, b)


@register(name=["aten.neg.default"])
def neg(x: Any) -> Any:
    """Lower ``aten.neg.default`` to MLX: ``-x``."""
    return -x


@register(name=["aten.abs.default"])
def abs_(x: Any) -> Any:
    """Lower ``aten.abs.default`` to MLX: ``mx.abs(x)``."""
    return mx.abs(x)


@register(name=["aten.exp.default"])
def exp(x: Any) -> Any:
    """Lower ``aten.exp.default`` to MLX: ``mx.exp(x)``."""
    return mx.exp(x)


@register(name=["aten.log.default"])
def log(x: Any) -> Any:
    """Lower ``aten.log.default`` to MLX: ``mx.log(x)``."""
    return mx.log(x)


@register(name=["aten.sqrt.default"])
def sqrt(x: Any) -> Any:
    """Lower ``aten.sqrt.default`` to MLX: ``mx.sqrt(x)``."""
    return mx.sqrt(x)


@register(name=["aten.rsqrt.default"])
def rsqrt(x: Any) -> Any:
    """Lower ``aten.rsqrt.default`` to MLX: ``mx.rsqrt(x)``."""
    return mx.rsqrt(x)


@register(name=["aten.reciprocal.default"])
def reciprocal(x: Any) -> Any:
    """Lower ``aten.reciprocal.default`` to MLX: ``mx.reciprocal(x)``."""
    return mx.reciprocal(x)


@register(name=["aten.sin.default"])
def sin(x: Any) -> Any:
    """Lower ``aten.sin.default`` to MLX: ``mx.sin(x)``."""
    return mx.sin(x)


@register(name=["aten.cos.default"])
def cos(x: Any) -> Any:
    """Lower ``aten.cos.default`` to MLX: ``mx.cos(x)``."""
    return mx.cos(x)


@register(name=["aten.erf.default"])
def erf(x: Any) -> Any:
    """Lower ``aten.erf.default`` to MLX: ``mx.erf(x)``."""
    return mx.erf(x)


@register(name=["aten.floor.default"])
def floor(x: Any) -> Any:
    """Lower ``aten.floor.default`` to MLX: ``mx.floor(x)``."""
    return mx.floor(x)


@register(name=["aten.ceil.default"])
def ceil(x: Any) -> Any:
    """Lower ``aten.ceil.default`` to MLX: ``mx.ceil(x)``."""
    return mx.ceil(x)


@register(name=["aten.round.default"])
def round_(x: Any) -> Any:
    """Lower ``aten.round.default`` to MLX: ``mx.round(x)``."""
    return mx.round(x)


@register(name=["aten.tanh.default"])
def tanh(x: Any) -> Any:
    """Lower ``aten.tanh.default`` to MLX: ``mx.tanh(x)``."""
    return mx.tanh(x)


@register(name=["aten.sigmoid.default"])
def sigmoid(x: Any) -> Any:
    """Lower ``aten.sigmoid.default`` to MLX: ``mx.sigmoid(x)``."""
    return mx.sigmoid(x)


@register(name=["aten.relu.default"])
def relu(x: Any) -> Any:
    """Lower ``aten.relu.default`` to MLX: ``mx.maximum(x, 0)``."""
    return mx.maximum(x, 0)


@register(name=["aten.leaky_relu.default"])
def leaky_relu(x: Any, negative_slope: Any = 0.01) -> Any:
    """Lower ``aten.leaky_relu.default`` to MLX: ``mx.where(x >= 0, x, x * negative_slope)``."""
    return mx.where(x >= 0, x, x * negative_slope)


@register(name=["aten.silu.default"])
def silu(x: Any) -> Any:
    """Lower ``aten.silu.default`` to MLX: ``x * mx.sigmoid(x)``."""
    return x * mx.sigmoid(x)


@register(name=["aten.gelu.default"])
def gelu(x: Any, approximate: Any = "none") -> Any:
    """Lower ``aten.gelu.default`` to MLX."""
    if approximate == "tanh":
        return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


@register(name=["aten.hardsigmoid.default"])
def hardsigmoid(x: Any) -> Any:
    """Lower ``aten.hardsigmoid.default`` to MLX: ``mx.clip(x / 6 + 0.5, 0, 1)``."""
    return mx.clip(x / 6 + 0.5, 0, 1)


@register(name=["aten.hardswish.default"])
def hardswish(x: Any) -> Any:
    """Lower ``aten.hardswish.default`` to MLX: ``x * mx.clip(x / 6 + 0.5, 0, 1)``."""
    return x * mx.clip(x / 6 + 0.5, 0, 1)


@register(name=["aten.hardtanh.default"])
def hardtanh(x: Any, min_val: Any = -1.0, max_val: Any = 1.0) -> Any:
    """Lower ``aten.hardtanh.default`` to MLX: ``mx.clip(x, min_val, max_val)``."""
    return mx.clip(x, min_val, max_val)


@register(name=["aten.clamp.default", "aten.clamp.Tensor"])
def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
    """Lower ``aten.clamp.default``, ``aten.clamp.Tensor`` to MLX."""
    if min is None and max is None:
        return x
    if min is None:
        return mx.minimum(x, max)
    if max is None:
        return mx.maximum(x, min)
    return mx.clip(x, min, max)


@register(name=["aten.clamp_min.default"])
def clamp_min(x: Any, min: Any) -> Any:
    """Lower ``aten.clamp_min.default`` to MLX: ``mx.maximum(x, min)``."""
    return mx.maximum(x, min)


@register(name=["aten.clamp_max.default"])
def clamp_max(x: Any, max: Any) -> Any:
    """Lower ``aten.clamp_max.default`` to MLX: ``mx.minimum(x, max)``."""
    return mx.minimum(x, max)


@register(name=["aten.maximum.default"])
def maximum(a: Any, b: Any) -> Any:
    """Lower ``aten.maximum.default`` to MLX: ``mx.maximum(a, b)``."""
    return mx.maximum(a, b)


@register(name=["aten.minimum.default"])
def minimum(a: Any, b: Any) -> Any:
    """Lower ``aten.minimum.default`` to MLX: ``mx.minimum(a, b)``."""
    return mx.minimum(a, b)


@register(
    name=["aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther", "aten.where.Scalar"]
)
def where(cond: Any, a: Any, b: Any) -> Any:
    """Lower ``aten.where.self``, ``aten.where.ScalarSelf``, ``aten.where.ScalarOther``,
    ``aten.where.Scalar`` to MLX: ``mx.where(cond, a, b)``.
    """
    return mx.where(cond, a, b)


@register(name=["aten.masked_fill.Scalar", "aten.masked_fill.Tensor"])
def masked_fill(x: Any, mask: Any, value: Any) -> Any:
    """Lower ``aten.masked_fill.Scalar``, ``aten.masked_fill.Tensor`` to MLX: ``mx.where(mask,
    mx.array(value, dtype=x.dtype), x)``.
    """
    return mx.where(mask, mx.array(value, dtype=x.dtype), x)


@register(name=["aten.eq.Tensor", "aten.eq.Scalar"])
def eq(a: Any, b: Any) -> Any:
    """Lower ``aten.eq.Tensor``, ``aten.eq.Scalar`` to MLX: ``a == b``."""
    return a == b


@register(name=["aten.ne.Tensor", "aten.ne.Scalar"])
def ne(a: Any, b: Any) -> Any:
    """Lower ``aten.ne.Tensor``, ``aten.ne.Scalar`` to MLX: ``a != b``."""
    return a != b


@register(name=["aten.lt.Tensor", "aten.lt.Scalar"])
def lt(a: Any, b: Any) -> Any:
    """Lower ``aten.lt.Tensor``, ``aten.lt.Scalar`` to MLX: ``a < b``."""
    return a < b


@register(name=["aten.le.Tensor", "aten.le.Scalar"])
def le(a: Any, b: Any) -> Any:
    """Lower ``aten.le.Tensor``, ``aten.le.Scalar`` to MLX: ``a <= b``."""
    return a <= b


@register(name=["aten.gt.Tensor", "aten.gt.Scalar"])
def gt(a: Any, b: Any) -> Any:
    """Lower ``aten.gt.Tensor``, ``aten.gt.Scalar`` to MLX: ``a > b``."""
    return a > b


@register(name=["aten.ge.Tensor", "aten.ge.Scalar"])
def ge(a: Any, b: Any) -> Any:
    """Lower ``aten.ge.Tensor``, ``aten.ge.Scalar`` to MLX: ``a >= b``."""
    return a >= b


@register(name=["aten.logical_not.default", "aten.bitwise_not.default"])
def logical_not(x: Any) -> Any:
    """Lower ``aten.logical_not.default``, ``aten.bitwise_not.default`` to MLX:
    ``mx.logical_not(x)``.
    """
    return mx.logical_not(x)


@register(name=["aten.logical_and.default", "aten.bitwise_and.Tensor"])
def logical_and(a: Any, b: Any) -> Any:
    """Lower ``aten.logical_and.default``, ``aten.bitwise_and.Tensor`` to MLX: ``mx.logical_and(a,
    b)``.
    """
    return mx.logical_and(a, b)


@register(name=["aten.logical_or.default", "aten.bitwise_or.Tensor"])
def logical_or(a: Any, b: Any) -> Any:
    """Lower ``aten.logical_or.default``, ``aten.bitwise_or.Tensor`` to MLX: ``mx.logical_or(a,
    b)``.
    """
    return mx.logical_or(a, b)


@register(name=["aten.isnan.default"])
def isnan(x: Any) -> Any:
    """Lower ``aten.isnan.default`` to MLX: ``mx.isnan(x)``."""
    return mx.isnan(x)
