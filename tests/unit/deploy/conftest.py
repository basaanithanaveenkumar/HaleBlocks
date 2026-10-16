"""Fixtures shared by the deployment unit tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.calibration.random_calibration_source import RandomCalibrationSource
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.io_signature import IOSignature

SIGNATURE = {
    "inputs": [{"name": "x", "shape": ["batch", 8], "dtype": "float32"}],
    "outputs": ["y"],
    "dims": {"batch": [1, 1, 4]},
}


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"y": self.fc2(torch.relu(self.fc1(x)))}


class TinyAdapter(ModelAdapter):
    """Adapter over :class:`TinyMLP` used across the unit tests."""

    def build_model(self) -> nn.Module:
        torch.manual_seed(0)
        return TinyMLP()

    def default_signature(self) -> IOSignature:
        return IOSignature.from_config(SIGNATURE)


@pytest.fixture
def tiny_adapter() -> TinyAdapter:
    return TinyAdapter()


@pytest.fixture
def tiny_context(tmp_path, tiny_adapter) -> ExportContext:
    return ExportContext(tiny_adapter, RandomCalibrationSource(num_batches=2), tmp_path / "run")
