"""Element-wise ATen ops → MLX (arithmetic, activations, comparisons, logic).

Converters receive resolved arguments (mlx arrays, Python scalars, lists) in ATen's argument
order. Python scalars broadcast natively in MLX, so ``x + 1`` and ``1 + x`` need no wrapping.
"""

from __future__ import annotations

import math

import mlx.core as mx

from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _is_array(x) -> bool:
    return isinstance(x, mx.array)


@register(name=["aten.add.Tensor", "aten.add.Scalar"])
def add(a, b, alpha=1):
    return a + (b * alpha if alpha != 1 else b)


@register(name=["aten.sub.Tensor", "aten.sub.Scalar"])
def sub(a, b, alpha=1):
    return a - (b * alpha if alpha != 1 else b)


@register(name=["aten.rsub.Scalar", "aten.rsub.Tensor"])
def rsub(a, b, alpha=1):
    return b - (a * alpha if alpha != 1 else a)


@register(name=["aten.mul.Tensor", "aten.mul.Scalar"])
def mul(a, b):
    return a * b


@register(name=["aten.div.Tensor", "aten.div.Scalar"])
def div(a, b):
    return a / b


@register(name=["aten.div.Tensor_mode", "aten.div.Scalar_mode"])
def div_mode(a, b, rounding_mode=None):
    if rounding_mode == "floor":
        return mx.floor_divide(a, b)
    q = a / b
    if rounding_mode == "trunc":
        return mx.where(q < 0, mx.ceil(q), mx.floor(q))
    return q


@register(name=["aten.floor_divide.default"])
def floor_divide(a, b):
    return mx.floor_divide(a, b)


@register(name=["aten.remainder.Tensor", "aten.remainder.Scalar"])
def remainder(a, b):
    return mx.remainder(a, b)


@register(name=["aten.pow.Tensor_Scalar", "aten.pow.Tensor_Tensor", "aten.pow.Scalar"])
def power(a, b):
    if not _is_array(a):
        return mx.power(mx.array(a, dtype=b.dtype), b)
    return mx.power(a, b)


@register(name=["aten.neg.default"])
def neg(x):
    return -x


@register(name=["aten.abs.default"])
def abs_(x):
    return mx.abs(x)


@register(name=["aten.exp.default"])
def exp(x):
    return mx.exp(x)


@register(name=["aten.log.default"])
def log(x):
    return mx.log(x)


@register(name=["aten.sqrt.default"])
def sqrt(x):
    return mx.sqrt(x)


@register(name=["aten.rsqrt.default"])
def rsqrt(x):
    return mx.rsqrt(x)


@register(name=["aten.reciprocal.default"])
def reciprocal(x):
    return mx.reciprocal(x)


@register(name=["aten.sin.default"])
def sin(x):
    return mx.sin(x)


@register(name=["aten.cos.default"])
def cos(x):
    return mx.cos(x)


@register(name=["aten.erf.default"])
def erf(x):
    return mx.erf(x)


@register(name=["aten.floor.default"])
def floor(x):
    return mx.floor(x)


@register(name=["aten.ceil.default"])
def ceil(x):
    return mx.ceil(x)


@register(name=["aten.round.default"])
def round_(x):
    return mx.round(x)


@register(name=["aten.tanh.default"])
def tanh(x):
    return mx.tanh(x)


@register(name=["aten.sigmoid.default"])
def sigmoid(x):
    return mx.sigmoid(x)


@register(name=["aten.relu.default"])
def relu(x):
    return mx.maximum(x, 0)


@register(name=["aten.leaky_relu.default"])
def leaky_relu(x, negative_slope=0.01):
    return mx.where(x >= 0, x, x * negative_slope)


@register(name=["aten.silu.default"])
def silu(x):
    return x * mx.sigmoid(x)


@register(name=["aten.gelu.default"])
def gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


@register(name=["aten.hardsigmoid.default"])
def hardsigmoid(x):
    return mx.clip(x / 6 + 0.5, 0, 1)


@register(name=["aten.hardswish.default"])
def hardswish(x):
    return x * mx.clip(x / 6 + 0.5, 0, 1)


@register(name=["aten.hardtanh.default"])
def hardtanh(x, min_val=-1.0, max_val=1.0):
    return mx.clip(x, min_val, max_val)


@register(name=["aten.clamp.default", "aten.clamp.Tensor"])
def clamp(x, min=None, max=None):  # noqa: A002 - ATen argument names
    if min is None and max is None:
        return x
    if min is None:
        return mx.minimum(x, max)
    if max is None:
        return mx.maximum(x, min)
    return mx.clip(x, min, max)


@register(name=["aten.clamp_min.default"])
def clamp_min(x, min):  # noqa: A002
    return mx.maximum(x, min)


@register(name=["aten.clamp_max.default"])
def clamp_max(x, max):  # noqa: A002
    return mx.minimum(x, max)


@register(name=["aten.maximum.default"])
def maximum(a, b):
    return mx.maximum(a, b)


@register(name=["aten.minimum.default"])
def minimum(a, b):
    return mx.minimum(a, b)


@register(
    name=["aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther", "aten.where.Scalar"]
)
def where(cond, a, b):
    return mx.where(cond, a, b)


@register(name=["aten.masked_fill.Scalar", "aten.masked_fill.Tensor"])
def masked_fill(x, mask, value):
    return mx.where(mask, mx.array(value, dtype=x.dtype), x)


@register(name=["aten.eq.Tensor", "aten.eq.Scalar"])
def eq(a, b):
    return a == b


@register(name=["aten.ne.Tensor", "aten.ne.Scalar"])
def ne(a, b):
    return a != b


@register(name=["aten.lt.Tensor", "aten.lt.Scalar"])
def lt(a, b):
    return a < b


@register(name=["aten.le.Tensor", "aten.le.Scalar"])
def le(a, b):
    return a <= b


@register(name=["aten.gt.Tensor", "aten.gt.Scalar"])
def gt(a, b):
    return a > b


@register(name=["aten.ge.Tensor", "aten.ge.Scalar"])
def ge(a, b):
    return a >= b


@register(name=["aten.logical_not.default", "aten.bitwise_not.default"])
def logical_not(x):
    return mx.logical_not(x)


@register(name=["aten.logical_and.default", "aten.bitwise_and.Tensor"])
def logical_and(a, b):
    return mx.logical_and(a, b)


@register(name=["aten.logical_or.default", "aten.bitwise_or.Tensor"])
def logical_or(a, b):
    return mx.logical_or(a, b)


@register(name=["aten.isnan.default"])
def isnan(x):
    return mx.isnan(x)
