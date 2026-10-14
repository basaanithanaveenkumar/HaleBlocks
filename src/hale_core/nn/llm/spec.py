"""What the registry knows about one fine-tunable language-model backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: Attention + MLP projections of Llama-style decoders (Llama, Qwen2/2.5/3, Mistral, Gemma 2/3,
#: SmolLM2, DeepSeek-R1-Distill-Qwen). Adapting all seven is the QLoRA recipe
#: (Dettmers et al., 2023, arXiv:2305.14314): adapters on every linear layer match full
#: fine-tuning far better than attention-only adapters.
LLAMA_STYLE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
#: Phi-3 / Phi-4-mini fuse Q/K/V and gate/up into single projections.
PHI3_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


@dataclass(frozen=True)
class LLMBackboneSpec:
    """A registered, fine-tunable causal language model.

    Attributes:
        name: registry key used in configs (``qwen3-1.7b``).
        family: model family (``qwen3``, ``llama3``, ``hale`` ...), for grouping and docs.
        source: ``hf`` = Hugging Face ``AutoModelForCausalLM``; ``hale`` = HaloDiffusionLLM
            checkpoint built from ``hale_core.nn.backbones``.
        model_id: Hugging Face repo id (``hf``) or variant name (``hale``).
        params_b: approximate parameter count in billions (docs / sanity checks).
        lora_targets: default LoRA target module names (empty = derive from the module tree).
        gated: the repo requires accepting a licence on Hugging Face before download.
        reasoning: the model was trained to emit chain-of-thought (``<think>``) by default.
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
    paper: str = ""
    notes: str = field(default="")
