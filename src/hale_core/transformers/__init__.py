"""Hidden-state stacks. Operate on (batch, seq, d_model) — no tokenizer or LM head.

Use these from a VLM or world model: you supply embeddings, this module runs the decoder.
"""

from hale_core.transformers.dit import DiTStack
from hale_core.transformers.gpt import GPTStack
from hale_core.transformers.lgt import LGTStack, is_global_layer

__all__ = ["GPTStack", "LGTStack", "DiTStack", "is_global_layer"]
