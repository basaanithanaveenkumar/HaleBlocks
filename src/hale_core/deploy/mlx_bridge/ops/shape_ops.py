"""Views, reshapes, joins, slicing and indexing: ATen → MLX."""

from __future__ import annotations

import mlx.core as mx

from hale_core.deploy.registry import MLX_OPS

register = MLX_OPS.register_module

#: torch encodes "to the end" as INT64_MAX in slice nodes.
_INT64_MAX = 2**63 - 1


@register(
    name=[
        "aten.view.default",
        "aten._unsafe_view.default",
        "aten.reshape.default",
        "aten.view_copy.default",
    ]
)
def reshape(x, shape):
    return mx.reshape(x, [int(s) for s in shape])


@register(name=["aten.flatten.using_ints"])
def flatten(x, start_dim=0, end_dim=-1):
    return mx.flatten(x, start_axis=start_dim, end_axis=end_dim)


@register(name=["aten.permute.default", "aten.permute_copy.default"])
def permute(x, dims):
    return mx.transpose(x, [int(d) for d in dims])


@register(name=["aten.transpose.int"])
def transpose(x, dim0, dim1):
    return mx.swapaxes(x, dim0, dim1)


@register(name=["aten.unsqueeze.default", "aten.unsqueeze_copy.default"])
def unsqueeze(x, dim):
    return mx.expand_dims(x, dim if dim >= 0 else dim + x.ndim + 1)


@register(name=["aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default"])
def squeeze(x, dim=None):
    if dim is None:
        return mx.squeeze(x)
    dims = [d % x.ndim for d in (dim if isinstance(dim, (list, tuple)) else [dim])]
    dims = [d for d in dims if x.shape[d] == 1]  # torch ignores non-singleton dims
    return mx.squeeze(x, dims) if dims else x


@register(name=["aten.expand.default", "aten.expand_copy.default"])
def expand(x, size, implicit=False):
    size = [int(s) for s in size]
    lead = len(size) - x.ndim
    target = [s if s != -1 else x.shape[i - lead] for i, s in enumerate(size)]
    return mx.broadcast_to(x, target)


@register(name=["aten.repeat.default"])
def repeat(x, repeats):
    reps = [int(r) for r in repeats]
    if len(reps) > x.ndim:
        x = mx.reshape(x, [1] * (len(reps) - x.ndim) + list(x.shape))
    return mx.tile(x, reps)


@register(name=["aten.cat.default", "aten.concat.default"])
def cat(tensors, dim=0):
    return mx.concatenate([t for t in tensors if t.size > 0 or len(tensors) == 1], axis=dim)


@register(name=["aten.stack.default"])
def stack(tensors, dim=0):
    return mx.stack(list(tensors), axis=dim)


@register(name=["aten.split.Tensor", "aten.split_copy.Tensor"])
def split(x, split_size, dim=0):
    n = x.shape[dim]
    bounds = list(range(split_size, n, split_size))
    return mx.split(x, bounds, axis=dim)


@register(name=["aten.split_with_sizes.default", "aten.split_with_sizes_copy.default"])
def split_with_sizes(x, split_sizes, dim=0):
    bounds, total = [], 0
    for s in split_sizes[:-1]:
        total += int(s)
        bounds.append(total)
    return mx.split(x, bounds, axis=dim)


@register(name=["aten.unbind.int"])
def unbind(x, dim=0):
    return [mx.take(x, i, axis=dim) for i in range(x.shape[dim])]


@register(name=["aten.slice.Tensor", "aten.slice_copy.Tensor"])
def slice_(x, dim=0, start=None, end=None, step=1):
    dim = dim % x.ndim
    size = x.shape[dim]
    end = size if end is None or end >= _INT64_MAX else end
    index = [slice(None)] * x.ndim
    index[dim] = slice(start, end, step)
    return x[tuple(index)]


@register(name=["aten.select.int", "aten.select_copy.int"])
def select(x, dim, index):
    dim = dim % x.ndim
    return mx.take(x, index % x.shape[dim], axis=dim)


@register(name=["aten.index_select.default"])
def index_select(x, dim, index):
    return mx.take(x, index, axis=dim)


@register(name=["aten.index.Tensor"])
def index(x, indices):
    return x[tuple(slice(None) if i is None else i for i in indices)]


@register(name=["aten.gather.default"])
def gather(x, dim, index, sparse_grad=False):
    return mx.take_along_axis(x, index, axis=dim)


@register(name=["aten.embedding.default"])
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return mx.take(weight, indices, axis=0)


@register(name=["aten.constant_pad_nd.default", "aten.pad.default"])
def constant_pad(x, pad, value=0.0, mode="constant"):
    pairs = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):  # torch pads from the last dim backwards
        pairs[x.ndim - 1 - i] = (int(pad[2 * i]), int(pad[2 * i + 1]))
    return mx.pad(x, pairs, constant_values=value)


@register(name=["aten.triu.default"])
def triu(x, diagonal=0):
    return (
        mx.triu(x, k=diagonal)
        if x.dtype != mx.bool_
        else mx.triu(x.astype(mx.int8), k=diagonal).astype(mx.bool_)
    )


@register(name=["aten.tril.default"])
def tril(x, diagonal=0):
    return (
        mx.tril(x, k=diagonal)
        if x.dtype != mx.bool_
        else mx.tril(x.astype(mx.int8), k=diagonal).astype(mx.bool_)
    )


@register(name=["aten.roll.default"])
def roll(x, shifts, dims=()):
    return mx.roll(x, list(shifts), axis=list(dims) if dims else None)


@register(name=["aten.flip.default"])
def flip(x, dims):
    index = [slice(None)] * x.ndim
    for d in dims:
        index[d % x.ndim] = slice(None, None, -1)
    return x[tuple(index)]


@register(name=["aten.as_strided.default", "aten.as_strided_copy.default"])
def as_strided(x, size, stride, storage_offset=0):
    """Only *restrides* that keep the logical shape are supported (decompositions emit them to
    change memory format, e.g. ``adaptive_avg_pool2d``); MLX arrays are logically contiguous,
    so those are identities. Genuine strided windows have no safe MLX equivalent."""
    if [int(s) for s in size] == list(x.shape) and not storage_offset:
        return x
    from hale_core.deploy.exceptions import ConversionError

    raise ConversionError(
        f"as_strided to {list(size)} from {list(x.shape)} is not supported", ["aten.as_strided"]
    )
