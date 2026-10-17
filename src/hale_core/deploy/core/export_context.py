"""Shared state of one pipeline run, passed to every stage."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch import nn

from hale_core.deploy.core.artifact_store import ArtifactStore
from hale_core.deploy.core.io_signature import IOSignature

if TYPE_CHECKING:
    from hale_core.deploy.adapters.export_module import ExportModule
    from hale_core.deploy.adapters.model_adapter import ModelAdapter
    from hale_core.deploy.calibration.calibration_source import CalibrationSource


class ExportContext:
    """Lazily built model, signature, reference inputs/outputs and the artifact store.

    Expensive objects (the model, reference outputs) are built on first access and cached, so
    a pipeline that only runs e.g. an MLX-LM conversion never traces the full graph, and every
    parity stage compares against the *same* reference run.

    Args:
        adapter: the model adapter.
        calibration: calibration source for quantizing stages.
        work_dir: output directory of this run (``work_dir/name``).
        device: device profile dict from the config (vendor, chip, soc ...).
        seed: seed of the sampled reference inputs.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        calibration: CalibrationSource,
        work_dir: Path,
        device: dict[str, Any] | None = None,
        seed: int = 0,
        name: str = "model",
    ) -> None:
        self.adapter = adapter
        self.calibration = calibration
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.device = dict(device or {})
        self.seed = seed
        self.name = name
        self.store = ArtifactStore()
        self._inputs: dict[str, dict[str, torch.Tensor]] = {}
        self._references: dict[str, tuple[torch.Tensor, ...]] = {}

    @cached_property
    def model(self) -> nn.Module:
        """The adapter's model, built and loaded on first access."""
        return self.adapter.load()

    @cached_property
    def signature(self) -> IOSignature:
        """The adapter's effective I/O signature."""
        return self.adapter.signature()

    @cached_property
    def export_module(self) -> ExportModule:
        """The model wrapped for export (cached)."""
        return self.adapter.export_module(self.model)

    def sample_inputs(self, which: str = "opt") -> dict[str, torch.Tensor]:
        """Deterministic sampled inputs at a signature corner (cached per corner)."""
        if which not in self._inputs:
            generator = torch.Generator().manual_seed(
                self.seed + {"min": 1, "opt": 0, "max": 2, "trace": 3}[which]
            )
            self._inputs[which] = self.signature.sample(which, generator)
        return self._inputs[which]

    def example_args(self, which: str = "trace") -> tuple[torch.Tensor, ...]:
        """Positional inputs; the default ``trace`` corner keeps dynamic dims symbolic (see
        :attr:`DimSpec.trace`)."""
        return self.signature.as_args(self.sample_inputs(which))

    def reference_outputs(self, which: str = "opt") -> tuple[torch.Tensor, ...]:
        """PyTorch outputs of the export module on :meth:`sample_inputs` (cached)."""
        if which not in self._references:
            with torch.no_grad():
                self._references[which] = tuple(
                    t.detach() for t in self.export_module(*self.example_args(which))
                )
            logger.debug(
                "reference outputs ({}): {}",
                which,
                [tuple(t.shape) for t in self._references[which]],
            )
        return self._references[which]

    def stage_dir(self, stage_name: str) -> Path:
        """Per-stage output directory under the work directory (created)."""
        path = self.work_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path
