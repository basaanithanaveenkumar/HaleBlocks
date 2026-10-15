"""Looped token LM: learned positions + LoopedStack (one tied block run T times) + LM head."""

from __future__ import annotations

import torch.nn as nn

from hale_core.nn.backbones.sequence import SequenceBackbone
from hale_core.nn.layers.attention import AttnType
from hale_core.nn.stacks.looped import LoopedStack, LoopSpec


class LoopedBackbone(SequenceBackbone):
    """``arch: looped`` - a GPT-style backbone whose ``n_layers`` form the shared, looped block.

    Works for every paradigm the other backbones serve (causal, bidirectional, block-causal
    attention, diffusion / flow time conditioning). ``loop`` configures the iteration
    (:class:`~hale_core.nn.stacks.looped.LoopSpec`); ``self.stack.aux_loss`` holds the ACT
    ponder loss of the last forward and ``self.stack.last_stats`` what the loop did.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 512,
        max_length: int = 64,
        dropout: float = 0.0,
        attn_type: AttnType = "causal",
        use_time_cond: bool = False,
        block_size: int | None = None,
        tie_embeddings: bool = True,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        arch: str = "looped",
        loop: LoopSpec | dict | None = None,
        **_ignored,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            attn_type=attn_type,
            block_size=block_size,
            use_time_cond=use_time_cond,
            learned_pos=True,
            tie_embeddings=tie_embeddings,
            arch=arch,
        )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        self.stack = LoopedStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn_impl=attn_impl,
            n_kv_heads=n_kv_heads,
            ffn_type=ffn_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared=moe_num_shared,
            use_time_cond=use_time_cond,
            loop=loop,
        )
        self.loop = self.stack.spec
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_head(self.lm_head)
        self.layers = self.stack.layers
        self.ln_out = self.stack.ln_out
        self.head = self.lm_head
