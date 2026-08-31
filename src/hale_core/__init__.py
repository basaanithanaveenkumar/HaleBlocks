"""hale-blocks: reusable transformer core, config, training, and plugin registries."""

from hale_core.nn import (
    BACKBONES,
    DiTBackbone,
    DiTStack,
    GPTBackbone,
    GPTStack,
    LGTBackbone,
    LGTStack,
    TransformerBackbone,
    build_attn_mask,
    build_backbone,
)
from hale_core.registry import (
    NamedRegistry,
    VariantRegistry,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_variant,
)
from hale_core.runtime import CheckpointStore, load_checkpoint, save_checkpoint

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
