"""``SamplerFn``: Structural type of a registered sampler: generates token ids from a prompt."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class SamplerFn(Protocol):
    """Structural type of a registered sampler: generates token ids from a prompt."""
    def __call__(
        self, model: nn.Module, tokenizer: Any, prompt_ids: torch.Tensor, **kwargs: Any
    ) -> Any: ...
