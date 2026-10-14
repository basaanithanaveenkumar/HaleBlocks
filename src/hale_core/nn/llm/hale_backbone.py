"""A HaloDiffusionLLM causal checkpoint (``hale_core.nn.backbones``) as a :class:`LanguageBackbone`."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn

from hale_core.nn.backbones import SequenceBackbone, build_backbone
from hale_core.nn.layers import build_attn_mask
from hale_core.nn.llm.base import LanguageBackbone, LanguageBackboneOutput
from hale_core.nn.llm.catalog import UNSUPPORTED_HALE_VARIANTS
from hale_core.nn.llm.hf_backbone import DTYPES
from hale_core.nn.llm.spec import LLMBackboneSpec

#: Linear layers that must never get adapters (MoE routing would change expert selection).
_NO_LORA = ("router", "gate", "noise", "lm_head")


class HaleSequenceLMBackbone(LanguageBackbone):
    """A ``SequenceBackbone`` (GPT / LGT stack) trained by HaloDiffusionLLM.

    Only causal variants qualify (``autoregressive``, ``mtp``); the checkpoint's attention must be
    ``causal`` and time conditioning off. The VLM / VLA path feeds ``inputs_embeds``: the
    backbone's learned positions are added here, then its stack and LM head run unchanged.

    LoRA is injected in place with PEFT's ``inject_adapter_in_model`` (the module is not a
    Hugging Face model); default targets are every linear layer of the transformer blocks except
    MoE routers. With the default ``mha`` attention the fused input projection is a
    ``nn.MultiheadAttention`` parameter, so only its output projection and the MLP get adapters;
    checkpoints trained with ``attn_impl: gqa`` expose separate ``wq/wk/wv/wo``.

    Args:
        spec: registry entry.
        backbone: the sequence LM.
        tokenizer: tokenizer it was trained with.
    """

    def __init__(self, spec: LLMBackboneSpec, backbone: SequenceBackbone, tokenizer: Any) -> None:
        super().__init__(spec, tokenizer)
        if backbone.attn_type != "causal" or backbone.use_time_cond:
            raise ValueError(
                f"{spec.name}: the checkpoint uses attn_type={backbone.attn_type!r}, "
                f"use_time_cond={backbone.use_time_cond}; a VLM/VLA backbone must be causal and "
                "unconditioned"
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.backbone = backbone

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls, spec: LLMBackboneSpec, path: str, dtype: str = "float32"
    ) -> HaleSequenceLMBackbone:
        """Rebuild the backbone from a HaloDiffusionLLM training checkpoint.

        The checkpoint holds ``variant``, ``vocab_size``, ``config`` (the full run config, whose
        ``model`` section sizes the backbone and ``data.tokenizer_name`` names the tokenizer)
        and ``model_state_dict``.
        """
        from transformers import AutoTokenizer  # lazy: optional dependency

        state = torch.load(path, map_location="cpu", weights_only=False)
        variant = state.get("variant")
        if variant in UNSUPPORTED_HALE_VARIANTS:
            raise ValueError(f"{path}: variant {variant!r} cannot be a backbone - "
                             f"{UNSUPPORTED_HALE_VARIANTS[variant]}")
        if variant != spec.model_id:
            raise ValueError(f"{path} holds variant {variant!r}, but {spec.name} expects {spec.model_id!r}")
        config = state["config"]
        overrides = {"tie_embeddings": False} if variant == "mtp" else {}
        backbone = build_backbone(int(state["vocab_size"]), SimpleNamespace(**config["model"]), **overrides)
        weights = {k.removeprefix("backbone."): v for k, v in state["model_state_dict"].items()
                   if k.startswith("backbone.")}  # MTP's extra heads are not part of the backbone
        backbone.load_state_dict(weights)
        tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])
        logger.info("llm backbone {} loaded from {} ({} params)", spec.name, path,
                    sum(p.numel() for p in backbone.parameters()))
        return cls(spec, backbone.to(DTYPES[dtype]), tokenizer)

    # ------------------------------------------------------------------ geometry
    @property
    def hidden_size(self) -> int:
        return int(self.backbone.d_model)

    @property
    def vocab_size(self) -> int:
        return int(self.backbone.token_emb.weight.size(0))

    @property
    def max_length(self) -> int:
        """Longest sequence the learned positions cover."""
        return int(self.backbone.max_length)

    # ------------------------------------------------------------------ compute
    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.backbone.token_emb(input_ids)

    def forward(
        self,
        *,
        inputs_embeds: Tensor | None = None,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> LanguageBackboneOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("pass inputs_embeds or input_ids")
            inputs_embeds = self.embed_tokens(input_ids)
        bb = self.backbone
        length = inputs_embeds.size(1)
        if length > bb.max_length:
            raise ValueError(f"sequence of {length} tokens exceeds the backbone's max_length {bb.max_length}")
        hidden = inputs_embeds if bb.pos_emb is None else inputs_embeds + bb.pos_emb[:, :length]
        device = inputs_embeds.device
        causal = build_attn_mask(bb.attn_type, length, device, block_size=bb.block_size)
        padding = None if attention_mask is None else attention_mask == 0
        hidden = bb.encode(hidden, None, causal, padding, torch.arange(length, device=device))
        logits = bb.lm_head(hidden)
        loss = None if labels is None else self.causal_lm_loss(logits, labels)
        return LanguageBackboneOutput(logits, hidden, loss)

    # ------------------------------------------------------------------ vocabulary
    def resize_token_embeddings(self, num_tokens: int) -> None:
        bb = self.backbone
        tied = bb.lm_head.weight is bb.token_emb.weight
        old = bb.token_emb.weight
        bb.token_emb = nn.Embedding(num_tokens, old.size(1), device=old.device, dtype=old.dtype)
        with torch.no_grad():
            bb.token_emb.weight.copy_(self._grown_rows(old.detach(), num_tokens))
        head = bb.lm_head
        new_head = nn.Linear(head.in_features, num_tokens, bias=False, device=old.device, dtype=old.dtype)
        if tied:
            new_head.weight = bb.token_emb.weight
        else:
            with torch.no_grad():
                new_head.weight.copy_(self._grown_rows(head.weight.detach(), num_tokens))
        bb.lm_head = bb.head = new_head
        bb.vocab_size = num_tokens

    # ------------------------------------------------------------------ fine-tuning
    def default_lora_targets(self) -> list[str]:
        names = []
        for name, module in self.backbone.named_modules():
            leaf = name.split(".")
            if isinstance(module, nn.Linear) and ".layers." in f".{name}" and not set(leaf) & set(_NO_LORA):
                names.append(f"backbone.{name}")
        return names

    def apply_lora(self, r: int, alpha: int, dropout: float, targets: Sequence[str]) -> None:
        from peft import LoraConfig, inject_adapter_in_model  # lazy: optional dependency

        self.requires_grad_(False)
        inject_adapter_in_model(
            LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets), bias="none"),
            self,
        )
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def embedding_parameters(self) -> list[nn.Parameter]:
        params = [self.backbone.token_emb.weight]
        if self.backbone.lm_head.weight is not params[0]:
            params.append(self.backbone.lm_head.weight)
        return params
