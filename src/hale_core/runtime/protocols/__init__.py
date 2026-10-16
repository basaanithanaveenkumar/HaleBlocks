"""Structural types so training/eval depend on interfaces, not concrete variants."""

from hale_core.runtime.protocols.attention_block import AttentionBlock
from hale_core.runtime.protocols.feed_forward_block import FeedForwardBlock
from hale_core.runtime.protocols.loss_fn import LossFn
from hale_core.runtime.protocols.sampler_fn import SamplerFn

__all__ = ["AttentionBlock", "FeedForwardBlock", "LossFn", "SamplerFn"]
