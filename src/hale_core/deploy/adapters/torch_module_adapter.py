"""Generic adapter: build any ``nn.Module`` from an import path and keyword arguments."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from torch import nn

from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.exceptions import ConfigError
from hale_core.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class TorchModuleAdapter(ModelAdapter):
    """``factory: "package.module:ClassOrFunction"`` called with ``kwargs``.

    Covers models that need no custom input mapping (their ``forward`` takes the signature's
    inputs as keyword arguments). The signature must be given in the config.

    Example::

        adapter:
          type: TorchModuleAdapter
          factory: torchvision.models:resnet18
          kwargs: {num_classes: 10}
          signature:
            inputs: [{name: x, shape: [batch, 3, 224, 224]}]
            outputs: [logits]
            dims: {batch: [1, 1, 8]}
    """

    def __init__(
        self,
        factory: str,
        kwargs: Mapping[str, Any] | None = None,
        positional: bool = False,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        if ":" not in factory:
            raise ConfigError(f"factory must look like 'package.module:Name', got {factory!r}")
        self.factory = factory
        self.kwargs = dict(kwargs or {})
        self.positional = positional

    def build_model(self) -> nn.Module:
        module_name, _, attr = self.factory.partition(":")
        try:
            target = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"cannot import factory {self.factory!r}: {exc}") from exc
        try:
            model = target(**self.kwargs)
        except TypeError as exc:
            raise ConfigError(
                f"factory {self.factory!r} rejected kwargs {sorted(self.kwargs)}: {exc}"
            ) from exc
        if not isinstance(model, nn.Module):
            raise ConfigError(
                f"factory {self.factory!r} returned {type(model).__name__}, not an nn.Module"
            )
        return model

    def forward_for_export(self, model: nn.Module, inputs: dict) -> Any:
        return model(*inputs.values()) if self.positional else model(**inputs)
