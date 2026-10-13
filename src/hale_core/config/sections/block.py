"""Transformer block sub-config: bundles attention + FFN + norm settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.attention import AttentionConfig
from hale_core.config.sections.common import StrictModel
from hale_core.config.sections.ffn import FFNConfig
from hale_core.config.sections.norm import NormConfig


class BlockConfig(StrictModel):
    """A single transformer block's full configuration.

    Controls norm placement, time conditioning (DiT), and delegates all
    attention / FFN parameters to their typed sub-configs.

    YAML example::

        model:
          block:
            norm_type: rms
            post_norm: false
            use_time_cond: false
            attention:
              impl: gqa
              n_kv_heads: 4
              qk_norm: true
            ffn:
              type: geglu
              dropout: 0.05
            norm:
              type: rms
              eps: 1e-6
    """

    norm_type: Literal["layer", "rms"] = "layer"
    post_norm: bool = False
    use_time_cond: bool = False
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    ffn: FFNConfig = Field(default_factory=FFNConfig)
    norm: NormConfig = Field(default_factory=NormConfig)
