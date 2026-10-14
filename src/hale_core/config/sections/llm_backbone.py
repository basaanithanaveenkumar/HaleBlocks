"""Language-model backbone sub-config shared by the VLM and VLA repos."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from hale_core.config.sections.common import StrictModel


class LoRASettings(StrictModel):
    """Low-rank adapters (Hu et al., 2021, arXiv:2106.09685) on the backbone's linear layers.

    ``W x`` becomes ``W x + (alpha / r) * B A x`` with ``A: (r, in)`` and ``B: (out, r)``; only
    ``A`` / ``B`` train, ``W`` stays frozen. ``target_modules`` left empty uses the backbone's
    registered defaults (attention and MLP projections).
    """

    r: int = Field(default=16, ge=1)
    """Adapter rank."""
    alpha: int = Field(default=32, ge=1)
    """Scaling numerator: the update is multiplied by ``alpha / r``."""
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    """Dropout on the adapter input."""
    target_modules: list[str] | None = None
    """Module names (suffix match) to adapt; ``None`` = the registered defaults."""
    train_embeddings: bool = False
    """Also train the input embeddings and LM head (needed when new tokens are added and
    their embeddings must be learned, e.g. ``<image>`` / ``<halo_action>``)."""


class LLMBackboneConfig(StrictModel):
    """Which language model backs a VLM / VLA, and how it is fine-tuned.

    ``name`` selects an entry of :func:`hale_core.nn.llm.list_llm_backbones` (Hugging Face
    families such as ``qwen3-1.7b`` / ``smollm2-360m``, or ``hale-autoregressive`` /
    ``hale-mtp`` for HaloDiffusionLLM checkpoints). Only causal, fine-tunable language models
    are registered.

    YAML example::

        llm_backbone:
          name: qwen2.5-1.5b
          finetune: lora          # lora | full | frozen
          dtype: bfloat16
          lora: {r: 16, alpha: 32, dropout: 0.05, train_embeddings: true}

        llm_backbone:
          name: hale-autoregressive
          checkpoint: checkpoints/ar_small/last.pt   # a HaloDiffusionLLM training checkpoint
    """

    name: str = "smollm2-360m"
    """Registered backbone name, or ``custom`` for any other Hugging Face causal LM (``model_id``
    required; LoRA defaults to the Llama-style q/k/v/o + MLP projections)."""
    model_id: str | None = None
    """Override the registered Hugging Face repo id / local path (same architecture family)."""
    checkpoint: str | None = None
    """HaloDiffusionLLM checkpoint (``*.pt``); required for ``hale-*`` backbones."""
    finetune: Literal["lora", "full", "frozen"] = "lora"
    """``lora``: train adapters only; ``full``: train every weight; ``frozen``: no LLM gradients."""
    lora: LoRASettings = Field(default_factory=LoRASettings)
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    """Weight dtype of the loaded backbone."""
    attn_implementation: str | None = "sdpa"
    """Hugging Face attention kernel (``sdpa``, ``eager``, ``flash_attention_2``)."""
    trust_remote_code: bool = False
    """Allow repository code; registered backbones never need it."""
    gradient_checkpointing: bool = False
    """Recompute activations in backward to save memory (Hugging Face backbones)."""

    @model_validator(mode="after")
    def _source_requirements(self) -> LLMBackboneConfig:
        if self.name.startswith("hale-") and self.checkpoint is None:
            raise ValueError(f"llm_backbone {self.name!r} needs `checkpoint:` (a HaloDiffusionLLM .pt)")
        if self.name == "custom" and self.model_id is None:
            raise ValueError("llm_backbone 'custom' needs `model_id:` (a Hugging Face causal LM repo or path)")
        return self
