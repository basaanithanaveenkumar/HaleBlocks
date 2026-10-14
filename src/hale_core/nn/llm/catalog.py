"""The fine-tunable language backbones available to the VLM and VLA.

Selection rule - a model is registered only if it is:

* a **causal decoder** (the VLM / VLA objectives are next-token prediction over a sequence
  whose image / state tokens are prepended or interleaved);
* **dense** (no MoE: LoRA on routed experts is memory-hungry and poorly supported);
* loadable with ``transformers`` alone (no ``trust_remote_code``) and **≤ 8B** parameters, so
  LoRA fine-tuning fits on one 24-80 GB GPU;
* text-only (multimodal checkpoints already carry their own vision tower).

HaloDiffusionLLM contributes its two causal variants. Its masked-diffusion, block-diffusion and
flow-matching variants are not registered: they need noise / time inputs and bidirectional or
doubled-stream attention, so they cannot serve as a next-token backbone.

Every Hugging Face id below was checked against the Hub (architecture and gating).
"""

from __future__ import annotations

from hale_core.nn.llm.registry import register_llm_backbone
from hale_core.nn.llm.spec import PHI3_TARGETS, LLMBackboneSpec

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

_HF_BACKBONES = (
    # SmolLM2 - Llama architecture, tiny: fastest iteration; 135M shares the VLA's cosmo2 tokenizer.
    LLMBackboneSpec("smollm2-135m", "smollm2", "hf", "HuggingFaceTB/SmolLM2-135M-Instruct", 0.135, paper=_SMOLLM2,
                    notes="same cosmo2 tokenizer as the built-in Halo-VLA decoder"),
    LLMBackboneSpec("smollm2-360m", "smollm2", "hf", "HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, paper=_SMOLLM2,
                    notes="SmolVLA / SmolVLM language backbone size"),
    LLMBackboneSpec("smollm2-1.7b", "smollm2", "hf", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, paper=_SMOLLM2),
    # Qwen2.5 instruct.
    LLMBackboneSpec("qwen2.5-0.5b", "qwen2.5", "hf", "Qwen/Qwen2.5-0.5B-Instruct", 0.49, paper=_QWEN25),
    LLMBackboneSpec("qwen2.5-1.5b", "qwen2.5", "hf", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, paper=_QWEN25),
    LLMBackboneSpec("qwen2.5-3b", "qwen2.5", "hf", "Qwen/Qwen2.5-3B-Instruct", 3.1, paper=_QWEN25),
    LLMBackboneSpec("qwen2.5-7b", "qwen2.5", "hf", "Qwen/Qwen2.5-7B-Instruct", 7.6, paper=_QWEN25),
    # Qwen3 dense (hybrid thinking / non-thinking chat template).
    LLMBackboneSpec("qwen3-0.6b", "qwen3", "hf", "Qwen/Qwen3-0.6B", 0.6, paper=_QWEN3),
    LLMBackboneSpec("qwen3-1.7b", "qwen3", "hf", "Qwen/Qwen3-1.7B", 1.7, paper=_QWEN3),
    LLMBackboneSpec("qwen3-4b", "qwen3", "hf", "Qwen/Qwen3-4B", 4.0, paper=_QWEN3),
    LLMBackboneSpec("qwen3-8b", "qwen3", "hf", "Qwen/Qwen3-8B", 8.2, paper=_QWEN3),
    # Llama 3.x instruct (gated: accept the Meta licence on the Hub first).
    LLMBackboneSpec("llama3.2-1b", "llama3", "hf", "meta-llama/Llama-3.2-1B-Instruct", 1.2, gated=True, paper=_LLAMA3),
    LLMBackboneSpec("llama3.2-3b", "llama3", "hf", "meta-llama/Llama-3.2-3B-Instruct", 3.2, gated=True, paper=_LLAMA3),
    LLMBackboneSpec("llama3.1-8b", "llama3", "hf", "meta-llama/Llama-3.1-8B-Instruct", 8.0, gated=True, paper=_LLAMA3),
    # Gemma (gated: accept the Gemma terms on the Hub first).
    LLMBackboneSpec("gemma2-2b", "gemma2", "hf", "google/gemma-2-2b-it", 2.6, gated=True, paper=_GEMMA2,
                    notes="use attn_implementation: eager (logit soft-capping)"),
    LLMBackboneSpec("gemma3-1b", "gemma3", "hf", "google/gemma-3-1b-it", 1.0, gated=True, paper=_GEMMA3,
                    notes="text-only Gemma 3; larger Gemma 3 sizes are multimodal and not registered"),
    # Phi (fused QKV / gate-up projections).
    LLMBackboneSpec("phi3.5-mini", "phi3", "hf", "microsoft/Phi-3.5-mini-instruct", 3.8,
                    lora_targets=PHI3_TARGETS, paper=_PHI3),
    LLMBackboneSpec("phi4-mini", "phi4", "hf", "microsoft/Phi-4-mini-instruct", 3.8,
                    lora_targets=PHI3_TARGETS, paper=_PHI4),
    # Mistral.
    LLMBackboneSpec("mistral-7b", "mistral", "hf", "mistralai/Mistral-7B-Instruct-v0.3", 7.2, paper=_MISTRAL),
    # DeepSeek-R1 distilled into Qwen2.5 (reasoning traces).
    LLMBackboneSpec("deepseek-r1-distill-qwen-1.5b", "deepseek-r1", "hf",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1.8, reasoning=True, paper=_R1),
    LLMBackboneSpec("deepseek-r1-distill-qwen-7b", "deepseek-r1", "hf",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.6, reasoning=True, paper=_R1),
)

_HALE_BACKBONES = (
    LLMBackboneSpec("hale-autoregressive", "hale", "hale", "autoregressive", 0.0, lora_targets=(),
                    paper="HaloDiffusionLLM autoregressive variant (GPT / LGT stacks from hale_core)",
                    notes="needs `checkpoint:`; size comes from the checkpoint"),
    LLMBackboneSpec("hale-mtp", "hale", "hale", "mtp", 0.0, lora_targets=(),
                    paper="Gloeckle et al., Better & Faster LLMs via Multi-token Prediction, arXiv:2404.19737",
                    notes="needs `checkpoint:`; the main next-token head is used, extra heads are dropped"),
)

for _spec in _HF_BACKBONES + _HALE_BACKBONES:
    register_llm_backbone(_spec)

#: HaloDiffusionLLM variants that are deliberately not backbones, and why (used in errors).
UNSUPPORTED_HALE_VARIANTS: dict[str, str] = {
    "diffusion": "masked diffusion: bidirectional attention and a noise-level input, no next-token head",
    "block_diffusion": "block diffusion: doubled clean/noised stream with block-causal attention",
    "flow_matching": "flow matching: continuous vector field, no token logits",
}
