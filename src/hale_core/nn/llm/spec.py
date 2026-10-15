"""What the registry knows about one fine-tunable language-model backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: Attention + MLP projections of Llama-style decoders (Llama, Qwen2/2.5/3, Mistral / Ministral 3,
#: Gemma 2/3/4, SmolLM2/3, Granite, Qwen3-VL text, DeepSeek-R1-Distill-Qwen). Adapting all seven
#: is the QLoRA recipe (Dettmers et al., 2023, arXiv:2305.14314): adapters on every linear layer
#: match full fine-tuning far better than attention-only adapters.
LLAMA_STYLE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
#: Phi-3 / Phi-4-mini fuse Q/K/V and gate/up into single projections.
PHI3_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
#: Qwen3.5 hybrid decoder: full-attention layers use q/k/v/o, Gated DeltaNet (linear-attention)
#: layers use ``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``; the per-head ``in_proj_a`` /
#: ``in_proj_b`` decay / gate projections are tiny and left frozen.
QWEN3_5_TARGETS = (*LLAMA_STYLE_TARGETS, "in_proj_qkv", "in_proj_z", "out_proj")
#: LFM2 hybrid decoder: attention layers (q/k/v/``out_proj``), gated short-convolution layers
#: (``in_proj`` / ``out_proj``) and SwiGLU feed-forwards (``w1`` / ``w2`` / ``w3``).
LFM2_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")

#: How the decoder mixes tokens, which matters to VLA action experts that read per-layer
#: key/value caches (pi0-style): only ``full`` layers carry a KV cache over the whole prefix.
AttentionLayout = Literal["full", "sliding_hybrid", "linear_hybrid", "conv_hybrid"]


@dataclass(frozen=True)
class LLMBackboneSpec:
    """A registered, fine-tunable causal language model.

    Attributes:
        name: registry key used in configs (``qwen3-1.7b``).
        family: model family (``qwen3``, ``llama3``, ``hale`` ...), for grouping and docs.
        source: ``hf`` = Hugging Face checkpoint; ``hale`` = HaloDiffusionLLM checkpoint built
            from ``hale_core.nn.backbones``.
        model_id: Hugging Face repo id (``hf``) or variant name (``hale``).
        params_b: approximate parameter count in billions of the **language model** that is
            loaded (for a ``vision-language`` checkpoint, without its vision / audio towers).
        lora_targets: default LoRA target module names (empty = derive from the module tree).
        gated: the repo requires accepting a licence on Hugging Face before download.
        reasoning: the model was trained to emit chain-of-thought (``<think>``) by default.
        modality: ``text`` = a ``*ForCausalLM`` checkpoint; ``vision-language`` = a natively
            multimodal checkpoint (``*ForConditionalGeneration``) whose **text tower** - decoder
            and LM head, with its multimodal pre-training - is extracted and used as the
            language backbone. Its own vision / audio towers are discarded: the VLM / VLA brings
            its configured vision encoder and projector.
        attention: token-mixing layout of the decoder (:data:`AttentionLayout`).
        license: licence of the weights when it is not a standard open licence.
        paper: reference for the model family.
        notes: anything a user should know when picking it.
    """

    name: str
    family: str
    source: Literal["hf", "hale"]
    model_id: str
    params_b: float
    lora_targets: tuple[str, ...] = LLAMA_STYLE_TARGETS
    gated: bool = False
    reasoning: bool = False
    modality: Literal["text", "vision-language"] = "text"
    attention: AttentionLayout = "full"
    license: str = ""
    paper: str = ""
    notes: str = field(default="")

    @property
    def is_multimodal_checkpoint(self) -> bool:
        """The weights come from a multimodal model; only its text tower is loaded."""
        return self.modality == "vision-language"
