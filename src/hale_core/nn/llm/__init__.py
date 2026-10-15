"""Fine-tunable language-model backbones for VLMs and VLAs.

One registry (:func:`list_llm_backbones`) of causal LLMs - Hugging Face text families (SmolLM2/3,
Qwen2.5, Qwen3, Llama 3.x, Gemma 2/3, Phi-3.5/4-mini, Mistral, Granite 4.2, DeepSeek-R1-Distill),
the text towers of natively multimodal models (Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2,
LFM2.5-VL, SmolVLM2; see :mod:`hale_core.nn.llm.hf_text_tower`) and HaloDiffusionLLM's causal
checkpoints - behind one interface, :class:`LanguageBackbone`, with
LoRA / full / frozen fine-tuning chosen by :class:`~hale_core.config.sections.LLMBackboneConfig`::

    from hale_core.config.sections import LLMBackboneConfig
    from hale_core.nn.llm import build_language_backbone

    llm, ids = build_language_backbone(LLMBackboneConfig(name="qwen3-1.7b", finetune="lora"),
                                       special_tokens=["<image>"])
    out = llm(inputs_embeds=embeds, attention_mask=mask, labels=labels)
    out.loss, out.logits, out.hidden_states

``transformers`` and ``peft`` are imported lazily (only when a backbone is built).
"""

from hale_core.nn.llm.base import LanguageBackbone, LanguageBackboneOutput
from hale_core.nn.llm.factory import (
    build_language_backbone,
    configure_finetuning,
    load_backbone_tokenizer,
    load_language_backbone,
    resolve_spec,
)
from hale_core.nn.llm.registry import (
    LLM_BACKBONES,
    get_llm_backbone_spec,
    is_registered,
    list_llm_backbones,
    register_llm_backbone,
)
from hale_core.nn.llm.spec import LLMBackboneSpec

__all__ = [
    "LLM_BACKBONES",
    "LLMBackboneSpec",
    "LanguageBackbone",
    "LanguageBackboneOutput",
    "build_language_backbone",
    "configure_finetuning",
    "get_llm_backbone_spec",
    "is_registered",
    "list_llm_backbones",
    "load_backbone_tokenizer",
    "load_language_backbone",
    "register_llm_backbone",
    "resolve_spec",
]
