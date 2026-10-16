"""``LoopInputInjection``: Combine the recurrent state with the loop input ``e`` before each pass."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.layers.loop._helpers import InjectionKind


class LoopInputInjection(nn.Module):
    """Combine the recurrent state with the loop input ``e`` before each pass.

    ``add``: ``h + e``. ``concat``: ``W [h; e]`` (Huginn's adapter), initialised so that it starts
    as ``h + e``. ``none``: ``h`` (the input enters only through ``h_0 = e``).
    """

    def __init__(self, kind: InjectionKind, d_model: int) -> None:
        super().__init__()
        if kind not in ("none", "add", "concat"):
            raise ValueError(f"unknown input injection {kind!r}")
        self.kind = kind
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False) if kind == "concat" else None
        if self.adapter is not None:
            with torch.no_grad():
                eye = torch.eye(d_model)
                self.adapter.weight.copy_(torch.cat([eye, eye], dim=1))

    @property
    def starts_from_zero(self) -> bool:
        """Injected loops start from ``h_0 = 0``; the input arrives through the injection."""
        return self.kind != "none"

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return state
        if self.kind == "add":
            return state + inputs
        return self.adapter(torch.cat([state, inputs], dim=-1))
