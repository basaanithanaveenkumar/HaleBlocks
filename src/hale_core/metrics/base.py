"""
Base metric interface for HaleBlocks.

All metrics follow the accumulate-then-compute pattern:
  1. Call ``reset()`` before each evaluation run.
  2. Call ``update(**kwargs)`` once per batch / model call.
  3. Call ``compute()`` at the end to get a ``dict[str, float]`` of results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base for all HaleBlocks metrics.

    Subclasses accumulate state in ``update()`` and produce a results dict
    in ``compute()``.  ``reset()`` clears all accumulators so the same
    instance can be reused across multiple evaluation runs.
    """

    # Human-readable name used for logging / display.
    metric_name: str = ""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state.  Call before each evaluation run."""
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Ingest one batch of data.

        Concrete signatures vary per subclass — keyword arguments are used
        to make call-sites self-documenting.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return accumulated results as a flat ``{metric_name: value}`` dict.

        The dict is suitable for direct logging to TensorBoard / W&B.
        Values are plain Python floats (not tensors).
        """
        raise NotImplementedError
