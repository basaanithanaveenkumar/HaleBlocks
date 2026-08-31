"""hale-llm-core: components, transformer stacks, token backbones, and registries."""

from hale_core.backbones import (
    BACKBONES,
    DiTBackbone,
    GPTBackbone,
    LGTBackbone,
    TransformerBackbone,
    build_backbone,
)
from hale_core.checkpoint import CheckpointStore, load_checkpoint, save_checkpoint
from hale_core.components import build_attn_mask
from hale_core.registry import (
    NamedRegistry,
    VariantRegistry,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_variant,
)
from hale_core.transformers import DiTStack, GPTStack, LGTStack

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BACKBONES",
    "build_backbone",
    "GPTBackbone",
    "LGTBackbone",
    "TransformerBackbone",
    "DiTBackbone",
    "GPTStack",
    "LGTStack",
    "DiTStack",
    "build_attn_mask",
    "NamedRegistry",
    "VariantRegistry",
    "CheckpointStore",
    "get_model",
    "get_variant",
    "get_loss",
    "get_sampler",
    "get_optimizer",
    "save_checkpoint",
    "load_checkpoint",
]
