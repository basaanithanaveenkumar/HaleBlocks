"""Token-sequence backbones: embeddings + transformer stack + optional LM head.

Swap via `model.arch` or `build_backbone`. Stacks are in `hale_core.transformers`
so a VLM can reuse GPTStack / LGTStack / DiTStack with its own embeddings.
"""

from hale_core.backbones.dit import DiTBackbone
from hale_core.backbones.factory import BACKBONES, backbone_kwargs, build_backbone, register_backbone
from hale_core.backbones.gpt import GPTBackbone, TransformerBackbone
from hale_core.backbones.lgt import LGTBackbone
from hale_core.backbones.sequence import SequenceBackbone

__all__ = [
    "BACKBONES",
    "SequenceBackbone",
    "GPTBackbone",
    "TransformerBackbone",
    "LGTBackbone",
    "DiTBackbone",
    "build_backbone",
    "backbone_kwargs",
    "register_backbone",
]
