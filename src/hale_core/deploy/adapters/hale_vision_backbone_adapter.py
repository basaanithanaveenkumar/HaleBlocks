"""Adapter for the HaleBlocks vision backbone zoo (ResNet, ConvNeXt, ViT, DINO, EVA ...)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class HaleVisionBackboneAdapter(ModelAdapter):
    """Export ``pixel_values [B, 3, H, W] → pooled_output [B, D]`` (plus ``patch_tokens`` for ViTs).

    Vision towers are the part of the VLM / VLA stacks that runs on every frame, so they are
    the main TensorRT / NPU target; this adapter is reused by Hale-VLM and HALE-WAM.

    Args:
        name: registered backbone name (``list_vision_backbones()``).
        kwargs: constructor kwargs (``img_size`` ...).
        outputs: ``BackboneOutput`` fields to export.
        max_batch: largest batch (cameras × frames) the engine must accept.
    """

    def __init__(
        self,
        name: str,
        kwargs: Mapping[str, Any] | None = None,
        outputs: Sequence[str] = ("pooled_output",),
        max_batch: int = 8,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.name = name
        self.kwargs = dict(kwargs or {})
        self.outputs = list(outputs)
        self.max_batch = max_batch

    def build_model(self) -> nn.Module:
        from hale_core.nn.vision.registry import build_vision_backbone

        return build_vision_backbone(self.name, **self.kwargs)

    def default_signature(self) -> IOSignature:
        size = int(self.kwargs.get("img_size", 224))
        channels = int(self.kwargs.get("in_chans", 3))
        return IOSignature.from_config(
            {
                "inputs": [
                    {
                        "name": "pixel_values",
                        "shape": ["batch", channels, size, size],
                        "dtype": "float32",
                        "value_range": [-2.0, 2.0],
                    }
                ],
                "outputs": self.outputs,
                "dims": {"batch": [1, 1, self.max_batch]},
            }
        )

    def forward_for_export(self, model: nn.Module, inputs: dict) -> Any:
        return model(inputs["pixel_values"])
