"""``LossFn``: Structural type of a registered loss: ``(model, batch) -> scalar tensor``."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class LossFn(Protocol):
    """Structural type of a registered loss: ``(model, batch) -> scalar tensor``."""

    def __call__(self, model: nn.Module, batch: dict[str, Any]) -> torch.Tensor: ...
