"""FFN sub-config: typed parameters for every registered feed-forward implementation."""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.common import StrictModel


class FFNConfig(StrictModel):
    """Fine-grained FFN configuration.

    YAML example::

        model:
          block:
            ffn:
              type: geglu
              dropout: 0.1

        # MoE example:
        model:
          block:
            ffn:
              type: moe
              moe_num_experts: 8
              moe_top_k: 2
              moe_num_shared: 1
    """

    type: Literal["mlp", "geglu", "moe"] = "mlp"
    dropout: float = 0.0
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1
