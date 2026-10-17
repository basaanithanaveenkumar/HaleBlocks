"""Adapter for HaleBlocks sequence backbones (GPT, LGT, DiT, looped) built by ``build_backbone``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class HaleSequenceBackboneAdapter(ModelAdapter):
    """Export ``input_ids [B, T] (+ t [B]) → logits [B, T, vocab]``.

    Sequence backbones carry a tied LM head, so their output is already vocabulary logits.

    Args:
        vocab_size: vocabulary size.
        backbone: kwargs of :func:`hale_core.nn.backbones.factory.build_backbone`
            (``arch``, ``d_model``, ``n_heads``, ``n_layers``, ``max_length`` ...).
        max_batch: largest batch the exported graph must accept.
        opt_seq: sequence length used for tracing / TensorRT kernel selection.
    """

    def __init__(
        self,
        vocab_size: int,
        backbone: Mapping[str, Any],
        max_batch: int = 8,
        opt_seq: int = 64,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.vocab_size = vocab_size
        self.backbone_kwargs = dict(backbone)
        self.max_batch = max_batch
        self.opt_seq = opt_seq

    def build_model(self) -> nn.Module:
        """Build the backbone from the adapter's config (weights loaded by :meth:`load`)."""
        from hale_core.nn.backbones.factory import build_backbone

        return build_backbone(self.vocab_size, **self.backbone_kwargs)

    @property
    def time_conditioned(self) -> bool:
        """Whether the backbone takes a diffusion time ``t`` input."""
        return bool(self.backbone_kwargs.get("use_time_cond"))

    def default_signature(self) -> IOSignature:
        """Inputs ``input_ids`` (and ``t`` when time-conditioned) with dynamic batch/sequence; output ``logits``."""
        max_len = int(self.backbone_kwargs.get("max_length", 512))
        inputs = [
            {
                "name": "input_ids",
                "shape": ["batch", "seq"],
                "dtype": "int64",
                "value_range": [0, self.vocab_size],
            }
        ]
        if self.time_conditioned:
            inputs.append(
                {"name": "t", "shape": ["batch"], "dtype": "float32", "value_range": [0.0, 1.0]}
            )
        return IOSignature.from_config(
            {
                "inputs": inputs,
                "outputs": ["logits"],
                "dims": {
                    "batch": [1, 1, self.max_batch],
                    "seq": [1, min(self.opt_seq, max_len), max_len],
                },
            }
        )

    def forward_for_export(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Call the backbone with the exported inputs and return its logits."""
        return model(inputs["input_ids"], t=inputs.get("t"))
