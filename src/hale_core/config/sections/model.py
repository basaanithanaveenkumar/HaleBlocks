"""``model:`` section of a run config: architecture, widths and block options."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.block import BlockConfig
from hale_core.config.sections.common import StrictModel
from hale_core.config.sections.loop import LoopConfig


class ModelConfig(StrictModel):
    """Architecture hyper-parameters.

    **Flat fields** (shorthands, always present — backwards-compatible)::

        model:
          d_model: 512
          n_heads: 8
          attn_impl: gqa
          ffn_type: geglu

    **Nested overrides** (fine-grained, take precedence when present)::

        model:
          d_model: 512
          n_heads: 8
          block:                # overrides attn_impl / ffn_type / norm / dropout at block level
            attention:
              impl: gqa
              n_kv_heads: 2
              qk_norm: true
            ffn:
              type: geglu
            norm:
              type: rms
    """

    # ── Backbone ───────────────────────────────────────────────────────────────
    arch: Literal["default", "transformer", "lgt", "dit", "looped"] = "default"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64

    # ── Attention (flat shorthands) ────────────────────────────────────────────
    attn_type: Literal["causal", "bidirectional", "block_causal"] = "causal"
    attn_impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    qk_norm: bool = False

    # ── FFN (flat shorthands) ──────────────────────────────────────────────────
    ffn_type: Literal["mlp", "geglu", "moe"] = "mlp"
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1

    # ── DiT / time-conditional ─────────────────────────────────────────────────
    use_time_cond: bool = False

    # ── LGT (local-global transformer) ────────────────────────────────────────
    block_size: int | None = None
    n_mtp_heads: int = 2
    sliding_window: int = 512
    local_global_ratio: int = 5
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    p_rope: float = 0.25

    # ── Looped transformer (arch: looped) ─────────────────────────────────────
    loop: LoopConfig = Field(default_factory=LoopConfig)

    # ── Fine-grained nested override (optional, takes precedence) ─────────────
    block: BlockConfig | None = Field(default=None)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def resolved_attn_impl(self) -> str:
        """Attention implementation, from ``block.attention`` when a block is configured."""
        return self.block.attention.impl if self.block else self.attn_impl

    def resolved_ffn_type(self) -> str:
        """Feed-forward type, from ``block.ffn`` when a block is configured."""
        return self.block.ffn.type if self.block else self.ffn_type

    def resolved_norm_type(self) -> str:
        """Normalisation type: ``block.norm_type`` when a block is configured, else ``"layer"``."""
        if self.block:
            return self.block.norm_type
        return "layer"

    def resolved_n_kv_heads(self) -> int | None:
        """Key/value heads (GQA), from ``block.attention`` when a block is configured."""
        if self.block:
            return self.block.attention.n_kv_heads
        return self.n_kv_heads

    def resolved_qk_norm(self) -> bool:
        """Whether queries/keys are normalised, from ``block.attention`` when configured."""
        if self.block:
            return self.block.attention.qk_norm
        return self.qk_norm
