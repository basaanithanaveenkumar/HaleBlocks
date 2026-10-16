"""Positional-in / tuple-out wrapper that every exporter can trace."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hale_core.deploy.exceptions import DeployError

ForwardFn = Callable[[nn.Module, dict[str, torch.Tensor]], Any]


class ExportModule(nn.Module):
    """Adapt any model to the calling convention exporters agree on.

    Exporters (``torch.onnx``, ``torch.export``, ``torch.jit.trace``, AI Hub, ExecuTorch) all
    want ``forward(*tensors) -> tuple[tensor, ...]``. Hale models take dict batches and return
    dicts, dataclasses or bare tensors. This wrapper maps the positional inputs back to names,
    calls the adapter's ``forward_fn`` and returns the outputs in signature order.

    Args:
        model: the wrapped module (registered as a submodule so its parameters export).
        forward_fn: ``(model, {name: tensor}) -> outputs`` (tensor, tuple or mapping).
        input_names: positional order of the inputs.
        output_names: order of the returned tuple; mapping outputs are selected by these names.
    """

    def __init__(
        self,
        model: nn.Module,
        forward_fn: ForwardFn,
        input_names: Sequence[str],
        output_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.forward_fn = forward_fn
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(args) != len(self.input_names):
            raise DeployError(
                f"expected {len(self.input_names)} inputs {self.input_names}, got {len(args)}"
            )
        outputs = self.forward_fn(self.model, dict(zip(self.input_names, args, strict=True)))
        return self._select(outputs)

    def _select(self, outputs: Any) -> tuple[torch.Tensor, ...]:
        if isinstance(outputs, torch.Tensor):
            return (outputs,)
        if isinstance(outputs, Mapping):
            missing = [n for n in self.output_names if n not in outputs]
            if missing:
                raise DeployError(f"model outputs {sorted(outputs)} lack {missing}")
            return tuple(outputs[n] for n in self.output_names)
        if hasattr(outputs, "__dataclass_fields__") or hasattr(outputs, "_fields"):
            return tuple(getattr(outputs, n) for n in self.output_names)
        if isinstance(outputs, (tuple, list)):
            return tuple(outputs[: len(self.output_names)])
        raise DeployError(
            f"cannot map model output of type {type(outputs).__name__} to {self.output_names}"
        )
