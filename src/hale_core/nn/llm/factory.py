"""Build a configured, fine-tuning-ready language backbone from :class:`LLMBackboneConfig`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger

from hale_core.config.sections.llm_backbone import LLMBackboneConfig
from hale_core.nn.llm.base import LanguageBackbone
from hale_core.nn.llm.registry import get_llm_backbone_spec
from hale_core.nn.llm.spec import LLMBackboneSpec


def resolve_spec(cfg: LLMBackboneConfig) -> LLMBackboneSpec:
    """The registered spec for ``cfg.name``, or a generic Hugging Face spec for ``custom``."""
    if cfg.name == "custom":
        return LLMBackboneSpec("custom", "custom", "hf", str(cfg.model_id), 0.0)
    return get_llm_backbone_spec(cfg.name)


def load_backbone_tokenizer(cfg: LLMBackboneConfig) -> Any:
    """The tokenizer of the backbone ``cfg`` selects, without loading model weights.

    Data pipelines use this so their token ids match the model's vocabulary.
    """
    from transformers import AutoTokenizer  # lazy: optional dependency

    spec = resolve_spec(cfg)
    if spec.source == "hf":
        name = cfg.model_id or spec.model_id
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg.trust_remote_code)
    else:
        state = torch.load(str(cfg.checkpoint), map_location="cpu", weights_only=False, mmap=True)
        tokenizer = AutoTokenizer.from_pretrained(state["config"]["data"]["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_language_backbone(cfg: LLMBackboneConfig) -> LanguageBackbone:
    """Load the backbone named by ``cfg`` with all weights trainable (no fine-tuning policy yet)."""
    spec = resolve_spec(cfg)
    if spec.source == "hf":
        from hale_core.nn.llm.hf_backbone import HFCausalLMBackbone

        return HFCausalLMBackbone.from_pretrained(
            spec, cfg.model_id, cfg.dtype, cfg.attn_implementation, cfg.trust_remote_code
        )
    from hale_core.nn.llm.hale_backbone import HaleSequenceLMBackbone

    if cfg.checkpoint is None:
        raise ValueError(f"{cfg.name} needs a HaloDiffusionLLM checkpoint")
    return HaleSequenceLMBackbone.from_checkpoint(spec, cfg.checkpoint, cfg.dtype)


def configure_finetuning(backbone: LanguageBackbone, cfg: LLMBackboneConfig) -> LanguageBackbone:
    """Apply ``cfg.finetune`` (``frozen`` / ``full`` / ``lora``) to an already built backbone.

    Call this **after** adding special tokens: resizing the vocabulary creates new embedding
    parameters, which would otherwise escape the freezing done here.
    """
    if cfg.finetune == "frozen":
        backbone.requires_grad_(False)
    elif cfg.finetune == "full":
        backbone.requires_grad_(True)
    else:
        targets = cfg.lora.target_modules or backbone.default_lora_targets()
        if not targets:
            raise ValueError(f"{cfg.name}: no LoRA target modules found; set lora.target_modules")
        backbone.apply_lora(cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, targets)
    if cfg.finetune != "frozen" and cfg.lora.train_embeddings:
        for param in backbone.embedding_parameters():
            param.requires_grad = True
    if cfg.gradient_checkpointing and hasattr(backbone, "enable_gradient_checkpointing"):
        backbone.enable_gradient_checkpointing()
    trainable, total = backbone.parameter_counts()
    logger.info(
        "llm backbone {} finetune={} trainable {:,} / {:,} ({:.2%})",
        cfg.name,
        cfg.finetune,
        trainable,
        total,
        trainable / max(total, 1),
    )
    return backbone


def build_language_backbone(
    cfg: LLMBackboneConfig, special_tokens: Sequence[str] = ()
) -> tuple[LanguageBackbone, dict[str, int]]:
    """Load, add ``special_tokens`` (e.g. ``<image>``, ``<halo_action>``), then fine-tune-configure.

    Returns:
        The backbone and ``{token: id}`` for every special token.
    """
    backbone = load_language_backbone(cfg)
    ids = backbone.add_special_tokens(special_tokens) if special_tokens else []
    configure_finetuning(backbone, cfg)
    return backbone, dict(zip(special_tokens, ids, strict=True))
