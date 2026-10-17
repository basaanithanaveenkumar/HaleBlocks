"""Hugging Face causal LMs (or the text tower of a multimodal checkpoint) as a :class:`LanguageBackbone`."""

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
    """A pretrained decoder-only LM from the Hub (Qwen, Llama, SmolLM, Gemma, Phi, Mistral ...).

    ``vision-language`` specs (Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2, LFM2.5-VL,
    SmolVLM2) load the multimodal checkpoint and keep only its text tower
    (:mod:`hale_core.nn.llm.hf_text_tower`).

    LoRA uses PEFT's ``get_peft_model`` (``task_type=CAUSAL_LM``), so adapters can be saved
    with ``backbone.model.save_pretrained(dir)`` and merged for deployment with
    ``merge_and_unload()``.

    Gemma 4's "E" models add a per-layer embedding looked up from the **token id** of every
    position. A VLM / VLA splices image / state embeddings into ``inputs_embeds``, so the ids
    cannot be recovered from the embeddings; pass them as ``token_ids`` to :meth:`forward`
    (positions holding non-text embeddings may carry any id - ids outside the per-layer table,
    such as newly added special tokens, use the padding id).

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
        decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
        #: Rows of Gemma 4's per-layer embedding table (``None``: the model has none).
        self.per_layer_vocab: int | None = None
        if decoder is not None and getattr(decoder.config, "hidden_size_per_layer_input", None):
            self.per_layer_vocab = int(decoder.config.vocab_size_per_layer_input)

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
        if spec.is_multimodal_checkpoint:
            from hale_core.nn.llm.hf_text_tower import load_text_tower

            loader = load_text_tower
        else:
            loader = AutoModelForCausalLM.from_pretrained
        try:
            model = cls._load(loader, repo, dtype, kwargs)
        except ValueError as exc:
            # Some checkpoints contain sub-models without an SDPA kernel (Gemma 4's audio tower
            # in transformers >= 5.20); the text tower we keep runs fine with eager attention.
            if (
                "attention implementation" not in str(exc)
                or kwargs.get("attn_implementation") == "eager"
            ):
                raise
            logger.warning(
                "{}: {!r} attention unsupported here ({}); falling back to eager",
                spec.name,
                kwargs.get("attn_implementation"),
                exc,
            )
            model = cls._load(loader, repo, dtype, {**kwargs, "attn_implementation": "eager"})
        logger.info(
            "llm backbone {} loaded from {} ({}{})",
            spec.name,
            repo,
            dtype,
            ", text tower of a multimodal checkpoint" if spec.is_multimodal_checkpoint else "",
        )
        return cls(spec, model, tokenizer)

    @staticmethod
    def _load(loader: Any, repo: str, dtype: str, kwargs: dict[str, Any]) -> Any:
        """Call ``loader`` with the dtype keyword this transformers version understands."""
        try:  # transformers >= 4.56 renamed torch_dtype -> dtype
            return loader(repo, dtype=DTYPES[dtype], **kwargs)
        except TypeError:
            return loader(repo, torch_dtype=DTYPES[dtype], **kwargs)

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
        token_ids: Tensor | None = None,
    ) -> LanguageBackboneOutput:
        """Forward pass of :class:`HFCausalLMBackbone`.

        Args:
            inputs_embeds: input (``Tensor | None``).
            input_ids: input (``Tensor | None``).
            attention_mask: input (``Tensor | None``).
            labels: input (``Tensor | None``).
            token_ids: input (``Tensor | None``).

        Returns:
            ``LanguageBackboneOutput``.
        """
        extra: dict[str, Any] = {}
        if self.per_layer_vocab is not None and inputs_embeds is not None and token_ids is not None:
            extra["per_layer_inputs"] = self.per_layer_inputs(token_ids)
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **extra,
        )
        # hidden_states[-1] is the decoder output after the final norm (what lm_head reads).
        return LanguageBackboneOutput(out.logits, out.hidden_states[-1], out.loss)

    def per_layer_inputs(self, token_ids: Tensor) -> Tensor:
        """Gemma 4 per-layer token embeddings for ``token_ids`` ``(B, L)``.

        Ids beyond the per-layer table (special tokens added for the VLM / VLA) map to the
        padding id; those positions hold spliced embeddings whose identity is not a text token.
        """
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.where(
            token_ids < self.per_layer_vocab, token_ids, torch.full_like(token_ids, pad)
        )
        return self.base.get_decoder().get_per_layer_inputs(ids, None)

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
            LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=list(targets),
                task_type="CAUSAL_LM",
                bias="none",
            ),
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
