"""Config-driven builders: wire typed config objects → registered nn components.

All nn imports are lazy so this module can be imported before the nn package
without creating circular dependencies.

Public surface::

    from hale_core.registry.nn import (
        build_attention_from_cfg,   # AttentionConfig → nn.Module
        build_ffn_from_cfg,         # FFNConfig       → nn.Module
        build_norm_from_cfg,        # NormConfig      → nn.Module
        build_block_from_cfg,       # BlockConfig     → nn.Module
        build_scheduler,            # SchedulerConfig → LRScheduler
        build_dataloader,           # DataLoaderConfig, dataset → DataLoader
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    import torch.optim as optim
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset

    from hale_core.config.sections.attention import AttentionConfig
    from hale_core.config.sections.block import BlockConfig
    from hale_core.config.sections.dataloader import DataLoaderConfig
    from hale_core.config.sections.ffn import FFNConfig
    from hale_core.config.sections.model import ModelConfig
    from hale_core.config.sections.norm import NormConfig
    from hale_core.config.sections.schedule import SchedulerConfig


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def build_attention_from_cfg(
    cfg: AttentionConfig,
    *,
    d_model: int,
    n_heads: int,
) -> nn.Module:
    """Build a registered attention module from :class:`AttentionConfig`.

    Example::

        from hale_core.config.sections.attention import AttentionConfig
        from hale_core.registry.nn import build_attention_from_cfg

        attn = build_attention_from_cfg(
            AttentionConfig(impl="gqa", n_kv_heads=2, qk_norm=True),
            d_model=512, n_heads=8,
        )
    """
    from hale_core.nn.layers.attention.factory import build_attention  # lazy

    return build_attention(
        cfg.impl,
        d_model=d_model,
        n_heads=n_heads,
        dropout=cfg.dropout,
        n_kv_heads=cfg.n_kv_heads,
        rope_theta=cfg.rope_theta,
        rotary_dim=cfg.rotary_dim,
        qk_norm=cfg.qk_norm,
    )


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

def build_ffn_from_cfg(
    cfg: FFNConfig,
    *,
    d_model: int,
    d_ff: int,
) -> nn.Module:
    """Build a registered FFN module from :class:`FFNConfig`.

    Example::

        ffn = build_ffn_from_cfg(FFNConfig(type="geglu"), d_model=512, d_ff=2048)
    """
    from hale_core.nn.layers.ffn.factory import build_ffn  # lazy

    return build_ffn(
        cfg.type,
        d_model=d_model,
        d_ff=d_ff,
        dropout=cfg.dropout,
        moe_num_experts=cfg.moe_num_experts,
        moe_top_k=cfg.moe_top_k,
        moe_num_shared=cfg.moe_num_shared,
    )


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------

def build_norm_from_cfg(cfg: NormConfig, *, d_model: int) -> nn.Module:
    """Build a registered norm layer from :class:`NormConfig`.

    Example::

        norm = build_norm_from_cfg(NormConfig(type="rms"), d_model=512)
    """
    from hale_core.registry.plugins import NORMS_REGISTRY  # lazy

    cls = NORMS_REGISTRY.get(cfg.type)
    if cfg.type == "layer":
        return cls(d_model, eps=cfg.eps, elementwise_affine=cfg.affine)
    return cls(d_model, eps=cfg.eps)


def build_norm(name: str, d_model: int, **kwargs: Any) -> nn.Module:
    """Build a norm by registry name (string shorthand).

    Example::

        norm = build_norm("rms", 512)
    """
    from hale_core.registry.plugins import NORMS_REGISTRY  # lazy

    return NORMS_REGISTRY.get(name)(d_model, **kwargs)


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

def build_block_from_cfg(
    cfg: BlockConfig,
    *,
    d_model: int,
    n_heads: int,
    d_ff: int,
) -> nn.Module:
    """Build a full transformer block from :class:`BlockConfig`.

    Returns the registered class for *transformer* or *dit* blocks, wired up
    with the nested attention, FFN and norm configs.

    Example::

        from hale_core.config.sections.block import BlockConfig
        from hale_core.config.sections.attention import AttentionConfig

        blk = build_block_from_cfg(
            BlockConfig(
                norm_type="rms",
                attention=AttentionConfig(impl="gqa", n_kv_heads=2),
            ),
            d_model=512, n_heads=8, d_ff=2048,
        )
    """
    attn = build_attention_from_cfg(cfg.attention, d_model=d_model, n_heads=n_heads)
    ffn  = build_ffn_from_cfg(cfg.ffn, d_model=d_model, d_ff=d_ff)

    block_name = "dit" if cfg.use_time_cond else "transformer"

    from hale_core.registry.plugins import BLOCKS_REGISTRY  # lazy

    cls = BLOCKS_REGISTRY.get(block_name)
    return cls(
        d_model,
        attention=attn,
        ffn=ffn,
        use_time_cond=cfg.use_time_cond,
        norm=cfg.norm_type,
        post_norm=cfg.post_norm,
    )


# ---------------------------------------------------------------------------
# Backbone (from full ModelConfig)
# ---------------------------------------------------------------------------

def build_backbone_from_cfg(vocab_size: int, cfg: ModelConfig) -> nn.Module:
    """Build a registered backbone from :class:`ModelConfig`.

    Nested ``cfg.block`` overrides the flat ``attn_impl`` / ``ffn_type`` /
    ``norm_type`` shorthands when present.

    Example::

        backbone = build_backbone_from_cfg(vocab_size=32000, cfg=run_cfg.model)
    """
    from hale_core.nn.backbones.factory import backbone_kwargs, build_backbone  # lazy

    kwargs = backbone_kwargs(vocab_size, cfg)

    # Apply nested block overrides
    if cfg.block is not None:
        kwargs["attn_impl"]      = cfg.block.attention.impl
        kwargs["n_kv_heads"]     = cfg.block.attention.n_kv_heads
        kwargs["qk_norm"]        = cfg.block.attention.qk_norm
        kwargs["ffn_type"]       = cfg.block.ffn.type
        kwargs["moe_num_experts"]= cfg.block.ffn.moe_num_experts
        kwargs["moe_top_k"]      = cfg.block.ffn.moe_top_k
        kwargs["moe_num_shared"] = cfg.block.ffn.moe_num_shared

    return build_backbone(vocab_size, **{k: v for k, v in kwargs.items() if k != "vocab_size"})


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    cfg: SchedulerConfig,
    *,
    optimizer: optim.Optimizer,
    total_steps: int,
) -> LRScheduler:
    """Build a registered LR scheduler from :class:`SchedulerConfig`.

    Example::

        sched = build_scheduler(
            SchedulerConfig(type="warmup_cosine", warmup_steps=100),
            optimizer=opt,
            total_steps=10_000,
        )
    """
    from hale_core.registry.plugins import SCHEDULERS_REGISTRY  # lazy

    factory = SCHEDULERS_REGISTRY.get(cfg.type)
    return factory(optimizer, cfg, total_steps)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def build_dataloader(
    cfg: DataLoaderConfig,
    *,
    dataset: Dataset,
    batch_size: int,
    collate_fn: Any = None,
) -> DataLoader:
    """Build a DataLoader from :class:`DataLoaderConfig`.

    If ``cfg.collator`` is set to something other than ``"default"`` and
    *collate_fn* is not provided, the registered collator is looked up.

    Example::

        loader = build_dataloader(
            DataLoaderConfig(num_workers=4, pin_memory=True),
            dataset=train_ds,
            batch_size=32,
        )
    """
    from hale_core.registry.plugins import COLLATORS_REGISTRY, get_dataloader_factory  # lazy

    if collate_fn is None and cfg.collator != "default":
        collate_fn = COLLATORS_REGISTRY.get(cfg.collator)

    factory = get_dataloader_factory("default")
    return factory(dataset, cfg, batch_size, collate_fn=collate_fn)
