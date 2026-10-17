"""Abstract optimizer interface used by the trainer (wraps a ``torch.optim`` optimizer)."""

from __future__ import annotations

from typing import Any


class BaseOptimizer:
    """Uniform (params, config) constructor over a torch.optim.Optimizer."""

    def __init__(self, params: Any, config: dict[str, Any]) -> None:
        self.params = params
        self.config = config
        self.optimizer = self._make_optimizer(params, config)

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Reset gradients (``set_to_none`` frees them instead of zeroing)."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Apply one optimisation step."""
        self.optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        """Optimizer state for checkpoints."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state from a checkpoint."""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """The wrapped optimizer's parameter groups (learning-rate schedulers edit these)."""
        return self.optimizer.param_groups
