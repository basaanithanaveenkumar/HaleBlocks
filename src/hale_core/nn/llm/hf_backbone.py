"""Hugging Face ``AutoModelForCausalLM`` as a :class:`LanguageBackbone`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn

from hale_core.nn.llm.base import LanguageBackbone, LanguageBackboneOutput
from hale_core.nn.llm.spec import LLMBackboneSpec

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class HFCausalLMBackbone(LanguageBackbone):
    """A pretrained decoder-only LM from the Hub (Qwen, Llama, SmolLM2, Gemma, Phi, Mistral ...).

    LoRA uses PEFT's ``get_peft_model`` (``task_type=CAUSAL_LM``), so adapters can be saved
    with ``backbone.model.save_pretrained(dir)`` and merged for deployment with
    ``merge_and_unload()``.

    Args:
        spec: registry entry.
        model: the loaded ``*ForCausalLM`` module.
        tokenizer: its tokenizer (a pad token is added from EOS when missing).
    """

    def __init__(self, spec: LLMBackboneSpec, model: nn.Module, tokenizer: Any) -> None:
        super().__init__(spec, tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model = model

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls,
        spec: LLMBackboneSpec,
        model_id: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        trust_remote_code: bool = False,
    ) -> HFCausalLMBackbone:
        """Download / load ``model_id`` (default: the spec's repo) and its tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: optional dependency

        repo = model_id or spec.model_id
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:  # transformers >= 4.56 renamed torch_dtype -> dtype
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPES[dtype], **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=DTYPES[dtype], **kwargs)
        logger.info("llm backbone {} loaded from {} ({})", spec.name, repo, dtype)
        return cls(spec, model, tokenizer)

    # ------------------------------------------------------------------ geometry
    @property
    def base(self) -> nn.Module:
        """The underlying ``*ForCausalLM`` (unwrapped from PEFT)."""
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    @property
    def hidden_size(self) -> int:
        return int(self.base.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.base.get_input_embeddings().weight.size(0))

    # ------------------------------------------------------------------ compute
    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.base.get_input_embeddings()(input_ids)

    def forward(
        self,
        *,
        inputs_embeds: Tensor | None = None,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> LanguageBackboneOutput:
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[-1] is the decoder output after the final norm (what lm_head reads).
        return LanguageBackboneOutput(out.logits, out.hidden_states[-1], out.loss)

    def generate(self, **kwargs: Any) -> Tensor:
        """Hugging Face generation (``inputs_embeds`` / ``input_ids`` + sampling kwargs)."""
        return self.model.generate(**kwargs)

    # ------------------------------------------------------------------ vocabulary
    def resize_token_embeddings(self, num_tokens: int) -> None:
        # mean_resizing initialises new rows from the old embeddings' distribution.
        self.base.resize_token_embeddings(num_tokens, mean_resizing=True)

    # ------------------------------------------------------------------ fine-tuning
    def default_lora_targets(self) -> list[str]:
        return list(self.spec.lora_targets)

    def apply_lora(self, r: int, alpha: int, dropout: float, targets: Sequence[str]) -> None:
        from peft import LoraConfig, get_peft_model  # lazy: optional dependency

        self.model = get_peft_model(
            self.model,
            LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                       task_type="CAUSAL_LM", bias="none"),
        )

    def embedding_parameters(self) -> list[nn.Parameter]:
        params = [self.base.get_input_embeddings().weight]
        head = self.base.get_output_embeddings()
        if head is not None and head.weight is not params[0]:
            params.append(head.weight)
        return params

    def enable_gradient_checkpointing(self) -> None:
        self.base.gradient_checkpointing_enable()
        # With frozen embeddings (LoRA), checkpointed blocks need an input that requires grad.
        self.base.enable_input_require_grads()
