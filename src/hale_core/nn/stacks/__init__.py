"""Hidden-state stacks. Operate on (batch, seq, d_model) — no tokenizer or LM head.

Use these from a VLM or world model: you supply embeddings, this module runs the decoder.
"""

from hale_core.nn.stacks.dit import DiTStack
from hale_core.nn.stacks.gpt import GPTStack
from hale_core.nn.stacks.lgt import LGTStack, is_global_layer
from hale_core.nn.stacks.looped import (
    LOOP_PRESETS,
    LoopedStack,
    LoopPreset,
    LoopSpec,
    LoopStats,
    loop_aux_loss,
    loop_budget,
    loop_stats,
    looped_stacks,
    set_loop_budget,
)

__all__ = [
    "GPTStack",
    "LGTStack",
    "DiTStack",
    "LOOP_PRESETS",
    "LoopedStack",
    "LoopPreset",
    "LoopSpec",
    "LoopStats",
    "is_global_layer",
    "loop_aux_loss",
    "loop_budget",
    "loop_stats",
    "looped_stacks",
    "set_loop_budget",
]
