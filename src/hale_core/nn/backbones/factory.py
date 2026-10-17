"""Open/closed backbone factory. New stacks register here; callers use build_backbone."""

from __future__ import annotations

from hale_core.registry import NamedRegistry

BACKBONES = NamedRegistry("backbone")


def register_backbone(name: str):
    """Decorator registering a backbone class under ``name`` in ``BACKBONES``."""
    def deco(cls):
        BACKBONES.add(name, cls)
        return cls

    return deco


def backbone_kwargs(vocab_size: int, model_cfg, **overrides) -> dict:
    """Constructor kwargs of a backbone from a ``ModelConfig`` (``overrides`` win)."""
    kwargs = {
        "vocab_size": vocab_size,
        "d_model": model_cfg.d_model,
        "n_heads": model_cfg.n_heads,
        "n_layers": model_cfg.n_layers,
        "d_ff": model_cfg.d_ff,
        "max_length": model_cfg.max_length,
        "dropout": model_cfg.dropout,
        "attn_type": model_cfg.attn_type,
        "use_time_cond": model_cfg.use_time_cond,
        "block_size": model_cfg.block_size,
        "attn_impl": model_cfg.attn_impl,
        "n_kv_heads": model_cfg.n_kv_heads,
        "ffn_type": model_cfg.ffn_type,
        "moe_num_experts": model_cfg.moe_num_experts,
        "moe_top_k": model_cfg.moe_top_k,
        "moe_num_shared": model_cfg.moe_num_shared,
        "arch": model_cfg.arch,
        "sliding_window": model_cfg.sliding_window,
        "local_global_ratio": model_cfg.local_global_ratio,
        "rope_theta_local": model_cfg.rope_theta_local,
        "rope_theta_global": model_cfg.rope_theta_global,
        "p_rope": model_cfg.p_rope,
        "qk_norm": model_cfg.qk_norm,
        # Only the looped backbone reads it; configs without a `loop` section get the defaults.
        "loop": getattr(model_cfg, "loop", None),
    }
    kwargs.update(overrides)
    return kwargs


def build_backbone(vocab_size: int, model_cfg=None, **overrides):
    """Build the backbone named by ``model_cfg.arch`` for a vocabulary of ``vocab_size``.

    Args:
        vocab_size: tokenizer vocabulary size.
        model_cfg: the ``model:`` config section (defaults when ``None``).
        **overrides: constructor arguments that replace config values.
    """
    if model_cfg is not None:
        kwargs = backbone_kwargs(vocab_size, model_cfg, **overrides)
    else:
        kwargs = {"vocab_size": vocab_size, **overrides}
        kwargs.setdefault("arch", "transformer")
    return BACKBONES.get(kwargs["arch"])(**kwargs)


def _register_builtins() -> None:
    from hale_core.nn.backbones.dit import DiTBackbone
    from hale_core.nn.backbones.gpt import GPTBackbone
    from hale_core.nn.backbones.lgt import LGTBackbone
    from hale_core.nn.backbones.looped import LoopedBackbone

    if "transformer" in BACKBONES:
        return
    register_backbone("default")(GPTBackbone)
    register_backbone("transformer")(GPTBackbone)
    register_backbone("lgt")(LGTBackbone)
    register_backbone("dit")(DiTBackbone)
    register_backbone("looped")(LoopedBackbone)


_register_builtins()
