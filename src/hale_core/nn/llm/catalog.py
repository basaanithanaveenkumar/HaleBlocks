"""The fine-tunable language backbones available to the VLM and VLA.

Selection rule - a model is registered only if it is:

* a **causal decoder** (the VLM / VLA objectives are next-token prediction over a sequence
  whose image / state tokens are prepended or interleaved);
* **dense** (no MoE: LoRA on routed experts is memory-hungry and poorly supported);
* loadable with ``transformers`` alone (no ``trust_remote_code``) and **8B-class** - at most 9B
  parameters in the language model that is loaded - so LoRA fine-tuning fits on one 24-80 GB GPU.

Text-only ``*ForCausalLM`` checkpoints load as they are. **Natively multimodal** checkpoints
(``modality="vision-language"``: Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2, LFM2.5-VL,
SmolVLM2) contribute their **text tower** - decoder and LM head with their multimodal
pre-training; their vision / audio towers are discarded because the VLM / VLA brings its own
encoder and projector. Their ``params_b`` counts the language model only.

Hybrid decoders are allowed and marked by ``attention``: Qwen3.5 interleaves Gated DeltaNet
linear attention with full attention 3:1, LFM2.5 interleaves gated short convolutions with
attention, Gemma 4 interleaves sliding-window with global attention. They work as backbones
(the VLM / VLA reads hidden states), but a pi0-style action expert that attends to per-layer KV
caches can only use the ``full``-attention layers of a ``linear_hybrid`` / ``conv_hybrid`` model.

HaloDiffusionLLM contributes its two causal variants. Its masked-diffusion, block-diffusion and
flow-matching variants are not registered: they need noise / time inputs and bidirectional or
doubled-stream attention, so they cannot serve as a next-token backbone.

Every Hugging Face id below was checked against the Hub (architecture, gating, licence) and every
architecture was instantiated with the installed ``transformers`` from the repo's own config;
parameter counts come from those instantiations. :data:`NOT_REGISTERED` lists models that were
considered and why they are left out.
"""

from __future__ import annotations

from hale_core.nn.llm.registry import register_llm_backbone
from hale_core.nn.llm.spec import LFM2_TARGETS, PHI3_TARGETS, QWEN3_5_TARGETS, LLMBackboneSpec

_SMOLLM2 = "Allal et al., SmolLM2, arXiv:2502.02737"
_QWEN25 = "Qwen Team, Qwen2.5 Technical Report, arXiv:2412.15115"
_QWEN3 = "Qwen Team, Qwen3 Technical Report, arXiv:2505.09388"
_LLAMA3 = "Grattafiori et al., The Llama 3 Herd of Models, arXiv:2407.21783"
_GEMMA2 = "Gemma Team, Gemma 2, arXiv:2408.00118"
_GEMMA3 = "Gemma Team, Gemma 3 Technical Report, arXiv:2503.19786"
_PHI3 = "Abdin et al., Phi-3 Technical Report, arXiv:2404.14219"
_PHI4 = "Microsoft, Phi-4-Mini Technical Report, arXiv:2503.01743"
_MISTRAL = "Jiang et al., Mistral 7B, arXiv:2310.06825"
_R1 = "DeepSeek-AI, DeepSeek-R1, arXiv:2501.12948"
_QWEN35 = "Qwen Team, Qwen3.5 small models (0.8B-9B), Mar 2026"
_GEMMA4 = "Gemma Team, Gemma 4, 2026"
_MINISTRAL3 = "Mistral AI, Ministral 3 (3B / 8B / 14B), Dec 2025"
_COSMOS_R2 = "NVIDIA, Cosmos-Reason2 (physical-AI reasoning VLM; GR00T N1.7 backbone)"
_LFM25 = "Liquid AI, LFM2.5-VL (450M / 1.6B / 3B)"
_SMOLVLM = "Marafioti et al., SmolVLM, arXiv:2504.05299"
_SMOLLM3 = "Hugging Face, SmolLM3 (fully open 3B), 2025"
_GRANITE42 = "IBM, Granite 4.2, Aug 2026"
_PHI4R = "Microsoft, Phi-4-mini-reasoning, arXiv:2504.21233"

_HF_BACKBONES = (
    # SmolLM2 - Llama architecture, tiny: fastest iteration; 135M shares the VLA's cosmo2 tokenizer.
    LLMBackboneSpec(
        "smollm2-135m",
        "smollm2",
        "hf",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        0.135,
        paper=_SMOLLM2,
        notes="same cosmo2 tokenizer as the built-in Halo-VLA decoder",
    ),
    LLMBackboneSpec(
        "smollm2-360m",
        "smollm2",
        "hf",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        0.36,
        paper=_SMOLLM2,
        notes="SmolVLA / SmolVLM language backbone size",
    ),
    LLMBackboneSpec(
        "smollm2-1.7b", "smollm2", "hf", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, paper=_SMOLLM2
    ),
    # Qwen2.5 instruct.
    LLMBackboneSpec(
        "qwen2.5-0.5b", "qwen2.5", "hf", "Qwen/Qwen2.5-0.5B-Instruct", 0.49, paper=_QWEN25
    ),
    LLMBackboneSpec(
        "qwen2.5-1.5b", "qwen2.5", "hf", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, paper=_QWEN25
    ),
    LLMBackboneSpec("qwen2.5-3b", "qwen2.5", "hf", "Qwen/Qwen2.5-3B-Instruct", 3.1, paper=_QWEN25),
    LLMBackboneSpec("qwen2.5-7b", "qwen2.5", "hf", "Qwen/Qwen2.5-7B-Instruct", 7.6, paper=_QWEN25),
    # Qwen3 dense (hybrid thinking / non-thinking chat template).
    LLMBackboneSpec("qwen3-0.6b", "qwen3", "hf", "Qwen/Qwen3-0.6B", 0.6, paper=_QWEN3),
    LLMBackboneSpec("qwen3-1.7b", "qwen3", "hf", "Qwen/Qwen3-1.7B", 1.7, paper=_QWEN3),
    LLMBackboneSpec("qwen3-4b", "qwen3", "hf", "Qwen/Qwen3-4B", 4.0, paper=_QWEN3),
    LLMBackboneSpec("qwen3-8b", "qwen3", "hf", "Qwen/Qwen3-8B", 8.2, paper=_QWEN3),
    # Llama 3.x instruct (gated: accept the Meta licence on the Hub first).
    LLMBackboneSpec(
        "llama3.2-1b",
        "llama3",
        "hf",
        "meta-llama/Llama-3.2-1B-Instruct",
        1.2,
        gated=True,
        paper=_LLAMA3,
    ),
    LLMBackboneSpec(
        "llama3.2-3b",
        "llama3",
        "hf",
        "meta-llama/Llama-3.2-3B-Instruct",
        3.2,
        gated=True,
        paper=_LLAMA3,
    ),
    LLMBackboneSpec(
        "llama3.1-8b",
        "llama3",
        "hf",
        "meta-llama/Llama-3.1-8B-Instruct",
        8.0,
        gated=True,
        paper=_LLAMA3,
    ),
    # Gemma (gated: accept the Gemma terms on the Hub first).
    LLMBackboneSpec(
        "gemma2-2b",
        "gemma2",
        "hf",
        "google/gemma-2-2b-it",
        2.6,
        gated=True,
        paper=_GEMMA2,
        notes="use attn_implementation: eager (logit soft-capping)",
    ),
    LLMBackboneSpec(
        "gemma3-1b",
        "gemma3",
        "hf",
        "google/gemma-3-1b-it",
        1.0,
        gated=True,
        paper=_GEMMA3,
        notes="text-only Gemma 3; larger Gemma 3 sizes are multimodal and not registered",
    ),
    # Phi (fused QKV / gate-up projections).
    LLMBackboneSpec(
        "phi3.5-mini",
        "phi3",
        "hf",
        "microsoft/Phi-3.5-mini-instruct",
        3.8,
        lora_targets=PHI3_TARGETS,
        paper=_PHI3,
    ),
    LLMBackboneSpec(
        "phi4-mini",
        "phi4",
        "hf",
        "microsoft/Phi-4-mini-instruct",
        3.8,
        lora_targets=PHI3_TARGETS,
        paper=_PHI4,
    ),
    # Mistral.
    LLMBackboneSpec(
        "mistral-7b", "mistral", "hf", "mistralai/Mistral-7B-Instruct-v0.3", 7.2, paper=_MISTRAL
    ),
    # DeepSeek-R1 distilled into Qwen2.5 (reasoning traces).
    LLMBackboneSpec(
        "deepseek-r1-distill-qwen-1.5b",
        "deepseek-r1",
        "hf",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        1.8,
        reasoning=True,
        paper=_R1,
    ),
    LLMBackboneSpec(
        "deepseek-r1-distill-qwen-7b",
        "deepseek-r1",
        "hf",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        7.6,
        reasoning=True,
        paper=_R1,
    ),
    # SmolLM3 - fully open (weights, data, recipe): clean ablations where you must know what the LM saw.
    LLMBackboneSpec(
        "smollm3-3b",
        "smollm3",
        "hf",
        "HuggingFaceTB/SmolLM3-3B",
        3.08,
        paper=_SMOLLM3,
        notes="hybrid /think and /no_think chat template; 3 of 4 layers use NoPE",
    ),
    LLMBackboneSpec(
        "smollm3-3b-base",
        "smollm3",
        "hf",
        "HuggingFaceTB/SmolLM3-3B-Base",
        3.08,
        paper=_SMOLLM3,
        notes="base checkpoint, preferred for VLM / VLA pre-training",
    ),
    # Granite 4.2 dense (Apache 2.0); embeddings scaled by embedding_multiplier inside the model.
    LLMBackboneSpec(
        "granite4.2-3b", "granite", "hf", "ibm-granite/granite-4.2-3b", 3.66, paper=_GRANITE42
    ),
    LLMBackboneSpec(
        "granite4.2-8b",
        "granite",
        "hf",
        "ibm-granite/granite-4.2-8b",
        8.79,
        paper=_GRANITE42,
        notes="128K context; 8.8B with its untied output embeddings",
    ),
    LLMBackboneSpec(
        "phi4-mini-reasoning",
        "phi4",
        "hf",
        "microsoft/Phi-4-mini-reasoning",
        3.84,
        lora_targets=PHI3_TARGETS,
        reasoning=True,
        paper=_PHI4R,
    ),
)

#: Text towers of natively multimodal checkpoints (see the module docstring).
_MULTIMODAL_TEXT_TOWERS = (
    # Qwen3.5 small: hybrid Gated DeltaNet (linear) + full attention 3:1. Base = pre-training checkpoints.
    *(
        LLMBackboneSpec(
            f"qwen3.5-{size}{suffix}",
            "qwen3.5",
            "hf",
            f"Qwen/Qwen3.5-{size.upper()}{repo_suffix}",
            params,
            lora_targets=QWEN3_5_TARGETS,
            modality="vision-language",
            attention="linear_hybrid",
            paper=_QWEN35,
            notes=("base checkpoint, preferred for VLA pre-training; " if suffix else "")
            + "install flash-linear-attention + causal-conv1d for fast linear-attention kernels",
        )
        for size, params in (("0.8b", 0.75), ("2b", 1.88), ("4b", 4.21))
        for suffix, repo_suffix in (("", ""), ("-base", "-Base"))
    ),
    # Gemma 4 "E" models: per-layer embeddings (PLE) make the raw count larger than the effective size.
    LLMBackboneSpec(
        "gemma4-e2b",
        "gemma4",
        "hf",
        "google/gemma-4-E2B",
        4.63,
        modality="vision-language",
        attention="sliding_hybrid",
        paper=_GEMMA4,
        notes="base; 2.4B of the 4.6B are per-layer embedding tables (~2B effective); pass token_ids",
    ),
    LLMBackboneSpec(
        "gemma4-e2b-it",
        "gemma4",
        "hf",
        "google/gemma-4-E2B-it",
        4.63,
        modality="vision-language",
        attention="sliding_hybrid",
        paper=_GEMMA4,
        notes="instruction-tuned E2B; pass token_ids",
    ),
    LLMBackboneSpec(
        "gemma4-e4b",
        "gemma4",
        "hf",
        "google/gemma-4-E4B",
        7.46,
        modality="vision-language",
        attention="sliding_hybrid",
        paper=_GEMMA4,
        notes="base; 2.9B of the 7.5B are per-layer embedding tables (~4B effective); pass token_ids",
    ),
    LLMBackboneSpec(
        "gemma4-e4b-it",
        "gemma4",
        "hf",
        "google/gemma-4-E4B-it",
        7.46,
        modality="vision-language",
        attention="sliding_hybrid",
        paper=_GEMMA4,
        notes="instruction-tuned E4B; pass token_ids",
    ),
    # Ministral 3: dense Llama-style decoder. The default Instruct repos are FP8; the -BF16 ones are trainable.
    LLMBackboneSpec(
        "ministral3-3b",
        "ministral3",
        "hf",
        "mistralai/Ministral-3-3B-Base-2512",
        3.43,
        modality="vision-language",
        paper=_MINISTRAL3,
        notes="base checkpoint",
    ),
    LLMBackboneSpec(
        "ministral3-3b-instruct",
        "ministral3",
        "hf",
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        3.43,
        modality="vision-language",
        paper=_MINISTRAL3,
    ),
    LLMBackboneSpec(
        "ministral3-8b",
        "ministral3",
        "hf",
        "mistralai/Ministral-3-8B-Base-2512",
        7.95,
        modality="vision-language",
        paper=_MINISTRAL3,
        notes="base checkpoint",
    ),
    LLMBackboneSpec(
        "ministral3-8b-instruct",
        "ministral3",
        "hf",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        8.49,
        modality="vision-language",
        paper=_MINISTRAL3,
    ),
    # Cosmos-Reason2: Qwen3-VL architecture post-trained for physical reasoning (GR00T N1.7's backbone).
    LLMBackboneSpec(
        "cosmos-reason2-2b",
        "cosmos-reason2",
        "hf",
        "nvidia/Cosmos-Reason2-2B",
        1.72,
        modality="vision-language",
        gated=True,
        license="NVIDIA Open Model License",
        paper=_COSMOS_R2,
        notes="Qwen3-VL-2B architecture; accept the licence on the Hub",
    ),
    # LFM2.5-VL: hybrid gated short-convolution + attention decoder, built for edge latency.
    *(
        LLMBackboneSpec(
            f"lfm2.5-vl-{size}",
            "lfm2.5-vl",
            "hf",
            f"LiquidAI/LFM2.5-VL-{size.upper()}",
            params,
            lora_targets=LFM2_TARGETS,
            modality="vision-language",
            attention="conv_hybrid",
            license="LFM Open License",
            paper=_LFM25,
        )
        for size, params in (("450m", 0.35), ("1.6b", 1.17), ("3b", 2.70))
    ),
    # SmolVLM2: Llama (SmolLM2) text tower; very token-frugal images in the original model.
    LLMBackboneSpec(
        "smolvlm2-256m",
        "smolvlm2",
        "hf",
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        0.16,
        modality="vision-language",
        paper=_SMOLVLM,
    ),
    LLMBackboneSpec(
        "smolvlm2-500m",
        "smolvlm2",
        "hf",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        0.41,
        modality="vision-language",
        paper=_SMOLVLM,
    ),
    LLMBackboneSpec(
        "smolvlm2-2.2b",
        "smolvlm2",
        "hf",
        "HuggingFaceTB/SmolVLM2-2.2B-Base",
        1.81,
        modality="vision-language",
        paper=_SMOLVLM,
        notes="base checkpoint",
    ),
)

_HALE_BACKBONES = (
    LLMBackboneSpec(
        "hale-autoregressive",
        "hale",
        "hale",
        "autoregressive",
        0.0,
        lora_targets=(),
        paper="HaloDiffusionLLM autoregressive variant (GPT / LGT / looped stacks from hale_core)",
        notes="needs `checkpoint:`; size comes from the checkpoint",
    ),
    LLMBackboneSpec(
        "hale-mtp",
        "hale",
        "hale",
        "mtp",
        0.0,
        lora_targets=(),
        paper="Gloeckle et al., Better & Faster LLMs via Multi-token Prediction, arXiv:2404.19737",
        notes="needs `checkpoint:`; the main next-token head is used, extra heads are dropped",
    ),
    LLMBackboneSpec(
        "hale-looped",
        "hale",
        "hale",
        "looped",
        0.0,
        lora_targets=(),
        paper="Dehghani et al., Universal Transformers, arXiv:1807.03819; Geiping et al., "
        "recurrent depth, arXiv:2502.05171",
        notes="needs `checkpoint:`; weight-tied looped block, ACT ponder loss not applied "
        "while fine-tuning",
    ),
)

for _spec in _HF_BACKBONES + _MULTIMODAL_TEXT_TOWERS + _HALE_BACKBONES:
    register_llm_backbone(_spec)

#: HaloDiffusionLLM variants that are deliberately not backbones, and why (used in errors).
UNSUPPORTED_HALE_VARIANTS: dict[str, str] = {
    "diffusion": "masked diffusion: bidirectional attention and a noise-level input, no next-token head",
    "block_diffusion": "block diffusion: doubled clean/noised stream with block-causal attention",
    "flow_matching": "flow matching: continuous vector field, no token logits",
}

#: Models from the 2026 small-backbone survey that were considered and are not registered.
NOT_REGISTERED: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "9.4B language model: above the 8B-class limit",
    "IFM/K2-Horizon-0.9B/3.7B/7B": "needs trust_remote_code (k2_horizon is not in transformers)",
    "nvidia/Cosmos3-Edge": "omnimodal Mixture-of-Transformers world-action model (diffusion tower for "
    "video / actions): a WAM backbone, not a causal LM",
    "Wan-AI/Wan2.2-TI2V-5B": "video diffusion transformer: a WAM backbone, not a causal LM",
    "google/gemma-4-12B / 26B-A4B / 31B": "above 8B, or MoE",
    "mistralai/Ministral-3-*-Instruct-2512": "FP8-quantised weights; the -BF16 repos are registered instead",
}
