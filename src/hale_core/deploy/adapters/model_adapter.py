"""Contract between a repo's model and the deployment pipeline (Adapter pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from hale_core.deploy.adapters.export_module import ExportModule
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError, DeployError


class ModelAdapter(ABC):
    """Everything the pipeline needs to know about one exportable model.

    The pipeline never imports a repo's model classes; a repo registers an adapter that

    * builds the model (and loads its checkpoint),
    * declares the :class:`IOSignature` (names, dtypes, dynamic dims),
    * maps named inputs to a forward call (:meth:`forward_for_export`),
    * optionally provides a Hugging Face checkpoint (TensorRT-LLM / mlx-lm paths) and
      realistic calibration batches (quantizing stages).

    Subclasses implement :meth:`build_model` and usually :meth:`default_signature`.

    Args:
        signature: signature config; overrides :meth:`default_signature` when given.
        checkpoint: optional state-dict file (``.pt``/``.pth``/``.safetensors``) loaded after build.
        strict: strict state-dict loading.
    """

    def __init__(
        self,
        signature: Mapping[str, Any] | None = None,
        checkpoint: str | None = None,
        strict: bool = True,
    ) -> None:
        self._signature_cfg = signature
        self.checkpoint = checkpoint
        self.strict = strict

    # ------------------------------------------------------------------ required
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct the model with its architecture config (weights are loaded separately)."""

    # ------------------------------------------------------------------ overridable
    def default_signature(self) -> IOSignature:
        raise ConfigError(
            f"{type(self).__name__} has no default signature; set adapter.signature in the config"
        )

    def forward_for_export(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> Any:
        """Call the model with named inputs; default ``model(**inputs)``."""
        return model(**inputs)

    def hf_checkpoint(self, model: nn.Module) -> tuple[Any, Any] | None:
        """``(transformers.PreTrainedModel, tokenizer)`` when the model has an HF form."""
        return None

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]] | None:
        """Realistic inputs for quantization; ``None`` lets the pipeline sample randomly."""
        return None

    # ------------------------------------------------------------------ template methods
    def signature(self) -> IOSignature:
        if self._signature_cfg is not None:
            return IOSignature.from_config(self._signature_cfg)
        return self.default_signature()

    def load(self) -> nn.Module:
        """Build, load weights, switch to eval mode (Template Method)."""
        model = self.build_model()
        if self.checkpoint:
            self.load_checkpoint(model, Path(self.checkpoint))
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        logger.info(
            "adapter {} built {} ({:,} parameters)", type(self).__name__, type(model).__name__, n
        )
        return model

    def export_module(self, model: nn.Module) -> ExportModule:
        sig = self.signature()
        return ExportModule(model, self.forward_for_export, sig.input_names, sig.outputs).eval()

    def load_checkpoint(self, model: nn.Module, path: Path) -> None:
        """Load ``.safetensors`` or a torch file holding a state dict (optionally nested)."""
        if not path.is_file():
            raise DeployError(f"checkpoint not found: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
                    state = state[key]
                    break
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
        result = model.load_state_dict(state, strict=self.strict)
        logger.info(
            "loaded checkpoint {} (missing={}, unexpected={})",
            path,
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
