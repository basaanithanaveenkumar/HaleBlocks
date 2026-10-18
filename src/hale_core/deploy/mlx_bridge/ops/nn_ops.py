"""Normalization, convolution and pooling ATen ops → MLX.

Layout: torch convolutions are channels-first (NCHW / NCL); MLX convolutions are channels-last.
The converters transpose in and out around ``mx.conv1d`` / ``mx.conv2d``. Weights are
transposed per call; MLX's lazy graph fuses the transposes, and weights are constants, so the
cost is paid once per compiled graph.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module


def _per_dim(value, spatial: int) -> list[int]:
    """Expand an int or one-element list to ``spatial`` values."""
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    return values * spatial if len(values) == 1 else values


def _norm_axes(x, normalized_shape):
    return list(range(x.ndim - len(normalized_shape), x.ndim))


@register(name=["aten.layer_norm.default"])
def layer_norm(
    x: Any,
    normalized_shape: Any,
    weight: Any = None,
    bias: Any = None,
    eps: Any = 1e-05,
    cudnn_enable: Any = True,
) -> Any:
    """Lower ``aten.layer_norm.default`` to MLX."""
    if len(normalized_shape) == 1:
        return mx.fast.layer_norm(x, weight, bias, eps)
    axes = _norm_axes(x, normalized_shape)
    mean = mx.mean(x, axis=axes, keepdims=True)
    y = (x - mean) * mx.rsqrt(mx.var(x, axis=axes, keepdims=True) + eps)
    y = y * weight if weight is not None else y
    return y + bias if bias is not None else y


@register(name=["aten.native_layer_norm.default"])
def native_layer_norm(x: Any, normalized_shape: Any, weight: Any, bias: Any, eps: Any) -> Any:
    """Lower ``aten.native_layer_norm.default`` to MLX."""
    axes = _norm_axes(x, normalized_shape)
    mean = mx.mean(x, axis=axes, keepdims=True)
    rstd = mx.rsqrt(mx.var(x, axis=axes, keepdims=True) + eps)
    return layer_norm(x, normalized_shape, weight, bias, eps), mean, rstd


@register(name=["aten.rms_norm.default"])
def rms_norm(x: Any, normalized_shape: Any, weight: Any = None, eps: Any = None) -> Any:
    """Lower ``aten.rms_norm.default`` to MLX."""
    eps = eps if eps is not None else 1.1920928955078125e-07  # torch.finfo(float32).eps
    if len(normalized_shape) == 1:
        w = weight if weight is not None else mx.ones((x.shape[-1],), dtype=x.dtype)
        return mx.fast.rms_norm(x, w, eps)
    axes = _norm_axes(x, normalized_shape)
    y = x * mx.rsqrt(mx.mean(x * x, axis=axes, keepdims=True) + eps)
    return y * weight if weight is not None else y


@register(name=["aten.native_group_norm.default"])
def native_group_norm(
    x: Any, weight: Any, bias: Any, n: Any, c: Any, hxw: Any, group: Any, eps: Any
) -> Any:
    """Lower ``aten.native_group_norm.default`` to MLX."""
    shape = x.shape
    g = mx.reshape(x, (shape[0], group, -1))
    mean = mx.mean(g, axis=-1, keepdims=True)
    rstd = mx.rsqrt(mx.var(g, axis=-1, keepdims=True) + eps)
    y = mx.reshape((g - mean) * rstd, shape)
    affine = (1, -1) + (1,) * (len(shape) - 2)
    if weight is not None:
        y = y * mx.reshape(weight, affine)
    if bias is not None:
        y = y + mx.reshape(bias, affine)
    return y, mx.reshape(mean, (shape[0], group)), mx.reshape(rstd, (shape[0], group))


@register(name=["aten._native_batch_norm_legit_no_training.default"])
def batch_norm_eval(
    x: Any, weight: Any, bias: Any, running_mean: Any, running_var: Any, momentum: Any, eps: Any
) -> Any:
    """Lower ``aten._native_batch_norm_legit_no_training.default`` to MLX."""
    affine = (1, -1) + (1,) * (x.ndim - 2)
    y = (x - mx.reshape(running_mean, affine)) * mx.reshape(mx.rsqrt(running_var + eps), affine)
    if weight is not None:
        y = y * mx.reshape(weight, affine)
    if bias is not None:
        y = y + mx.reshape(bias, affine)
    empty = mx.zeros((0,), dtype=x.dtype)
    return y, empty, empty


@register(name=["aten.convolution.default"])
def convolution(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    transposed: Any,
    output_padding: Any,
    groups: Any,
) -> Any:
    """Lower ``aten.convolution.default`` to MLX."""
    if transposed:
        from hale_core.deploy.exceptions import ConversionError

        raise ConversionError(
            "transposed convolution is not supported by the MLX bridge",
            ["aten.convolution(transposed=True)"],
        )
    spatial = x.ndim - 2
    # ATen allows one value for every spatial dim (SigLIP's patch embedding passes padding=[0]).
    stride, padding, dilation = (_per_dim(v, spatial) for v in (stride, padding, dilation))
    if spatial == 1:
        y = mx.conv1d(
            mx.swapaxes(x, 1, 2),
            mx.swapaxes(weight, 1, 2),
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            groups=groups,
        )
        y = mx.swapaxes(y, 1, 2)
    elif spatial == 2:
        y = mx.conv2d(
            mx.transpose(x, (0, 2, 3, 1)),
            mx.transpose(weight, (0, 2, 3, 1)),
            stride=tuple(stride),
            padding=tuple(padding),
            dilation=tuple(dilation),
            groups=groups,
        )
        y = mx.transpose(y, (0, 3, 1, 2))
    else:
        from hale_core.deploy.exceptions import ConversionError

        raise ConversionError(
            f"{spatial}-d convolution is not supported by the MLX bridge", ["aten.convolution"]
        )
    if bias is not None:
        y = y + mx.reshape(bias, (1, -1) + (1,) * spatial)
    return y


@register(name=["aten.conv2d.default"])
def conv2d(
    x: Any,
    weight: Any,
    bias: Any = None,
    stride: Any = (1, 1),
    padding: Any = (0, 0),
    dilation: Any = (1, 1),
    groups: Any = 1,
) -> Any:
    """Lower ``aten.conv2d.default`` to MLX."""
    pair = lambda v: list(v) if isinstance(v, (list, tuple)) else [v, v]  # noqa: E731
    return convolution(
        x, weight, bias, pair(stride), pair(padding), pair(dilation), False, [0, 0], groups
    )


@register(name=["aten.adaptive_avg_pool2d.default", "aten._adaptive_avg_pool2d.default"])
def adaptive_avg_pool2d(x: Any, output_size: Any) -> Any:
    """Lower ``aten.adaptive_avg_pool2d.default``, ``aten._adaptive_avg_pool2d.default`` to MLX."""
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    n, c, h, w = x.shape
    if h % oh or w % ow:
        from hale_core.deploy.exceptions import ConversionError

        raise ConversionError(
            f"adaptive_avg_pool2d {h}x{w} -> {oh}x{ow} needs divisible sizes",
            ["aten.adaptive_avg_pool2d"],
        )
    return mx.mean(mx.reshape(x, (n, c, oh, h // oh, ow, w // ow)), axis=(3, 5))


@register(name=["aten.avg_pool2d.default"])
def avg_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = (0, 0),
    ceil_mode: Any = False,
    count_include_pad: Any = True,
    divisor_override: Any = None,
) -> Any:
    """Lower ``aten.avg_pool2d.default`` to MLX."""
    import mlx.nn as nn

    pool = nn.AvgPool2d(tuple(kernel_size), tuple(stride or kernel_size), tuple(padding))
    return mx.transpose(pool(mx.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))


@register(name=["aten.max_pool2d.default"])
def max_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = (0, 0),
    dilation: Any = (1, 1),
    ceil_mode: Any = False,
) -> Any:
    """Lower ``aten.max_pool2d.default`` to MLX."""
    import mlx.nn as nn

    pool = nn.MaxPool2d(tuple(kernel_size), tuple(stride or kernel_size), tuple(padding))
    return mx.transpose(pool(mx.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))


@register(name=["aten.max_pool2d_with_indices.default"])
def max_pool2d_with_indices(
    x: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = (0, 0),
    dilation: Any = (1, 1),
    ceil_mode: Any = False,
) -> Any:
    """Inference graphs only consume the values; indices (for unpooling) are not produced."""
    return max_pool2d(x, kernel_size, stride or kernel_size, padding, dilation, ceil_mode), None
