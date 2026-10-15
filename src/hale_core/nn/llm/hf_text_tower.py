"""The text tower of a natively multimodal Hugging Face checkpoint as a causal LM.

Natively multimodal models (Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2 / Qwen3-VL, LFM2.5-VL,
SmolVLM2) are pre-trained on interleaved image-text data, so their language decoder carries a
multimodal prior that a text-only LLM of the same size lacks. A VLM / VLA that brings its own
vision encoder and projector only needs that decoder and its LM head: :func:`extract_text_tower`
lifts them out of the ``*ForConditionalGeneration`` model into an ordinary causal LM - the
architecture's own ``*ForCausalLM`` class when ``transformers`` has one (so model-specific
details such as Gemma's final-logit soft-capping are kept), else :class:`TextTowerForCausalLM`.
The vision / audio towers and the checkpoint's own projector are dropped.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Imported lazily by ``hf_backbone`` (``transformers`` is the optional ``finetune`` extra).


class TextTowerForCausalLM(PreTrainedModel, GenerationMixin):
    """A multimodal checkpoint's decoder + LM head as a generic causal LM.

    Used for text configurations without a ``*ForCausalLM`` class in ``transformers``
    (e.g. ``qwen3_vl_text``). Supports ``inputs_embeds``, labels (shifted cross-entropy), hidden
    states, KV-cached generation, PEFT and gradient checkpointing.

    Args:
        config: the text configuration (``full.get_decoder().config``).
        decoder: the text model (``full.get_decoder()``).
        lm_head: the output projection (``full.get_output_embeddings()``).
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: Any, decoder: nn.Module, lm_head: nn.Module) -> None:
        super().__init__(config)
        self.model = decoder
        self.lm_head = lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden = out.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, keep, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


def extract_text_tower(full: nn.Module) -> nn.Module:
    """``*ForConditionalGeneration`` -> causal LM over its decoder and LM head (vision dropped).

    The decoder and head modules are moved, not copied, so tied input / output embeddings stay
    tied and no weight is duplicated.

    Raises:
        ValueError: ``full`` has no separable text decoder / LM head.
    """
    decoder = full.get_decoder()
    head = full.get_output_embeddings()
    if decoder is None or head is None or decoder is full:
        raise ValueError(f"{type(full).__name__} has no separable text decoder and LM head")
    text_config = decoder.config
    # The text config of a multimodal checkpoint often leaves tying to the outer config; record
    # what the weights actually do so vocabulary resizing keeps them tied (or untied).
    text_config.tie_word_embeddings = head.weight is decoder.get_input_embeddings().weight

    lm: nn.Module | None = None
    try:
        with torch.device("meta"):
            candidate = AutoModelForCausalLM.from_config(text_config)
        if type(candidate).__name__.endswith("ForCausalLM") and {n for n, _ in candidate.named_children()} == {
            "model", "lm_head"
        }:
            candidate.model, candidate.lm_head = decoder, head
            lm = candidate
    except ValueError:  # no *ForCausalLM registered for this text config
        lm = None
    if lm is None:
        lm = TextTowerForCausalLM(text_config, decoder, head)
    if any(t.is_meta for t in (*lm.parameters(), *lm.buffers())):
        raise RuntimeError(f"text tower of {type(full).__name__} still has uninitialised (meta) tensors")
    if getattr(full, "generation_config", None) is not None:
        lm.generation_config = full.generation_config
    return lm


def load_text_tower(repo: str, **from_pretrained_kwargs: Any) -> nn.Module:
    """Load the multimodal checkpoint ``repo`` and return only its text tower (see module doc)."""
    full = AutoModelForImageTextToText.from_pretrained(repo, **from_pretrained_kwargs)
    lm = extract_text_tower(full)
    del full  # releases the vision / audio towers
    gc.collect()
    return lm
