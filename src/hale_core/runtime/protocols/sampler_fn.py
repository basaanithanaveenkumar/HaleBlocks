"""``SamplerFn``: SamplerFn (Structural types so training/eval depend on interfaces, not concrete variants)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class SamplerFn(Protocol):
    def __call__(
        self, model: nn.Module, tokenizer: Any, prompt_ids: torch.Tensor, **kwargs: Any
    ) -> Any: ...
