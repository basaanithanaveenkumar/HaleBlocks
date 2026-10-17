"""Adam optimizer wrapper (registered as ``adam``)."""

from __future__ import annotations

from typing import Any

import torch

from hale_core.nn.optim.base import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    """``torch.optim.Adam`` behind the :class:`BaseOptimizer` interface."""
    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        return torch.optim.Adam(
            params,
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )
