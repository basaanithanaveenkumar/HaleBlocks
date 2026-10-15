# Fine-tunable LLM backbones for VLMs and VLAs

`hale_core.nn.llm` gives the vision-language model (Hale-VLM) and the vision-language-action model
(HALE-WAM) one registry of language backbones, one interface (`LanguageBackbone`) and one
fine-tuning policy (`LLMBackboneConfig`: LoRA, full or frozen). Install the extra with
`pip install -e '.[finetune]'` (`transformers` + `peft`, imported lazily).

```python
from hale_core.config.sections import LLMBackboneConfig
from hale_core.nn.llm import build_language_backbone, list_llm_backbones

llm, ids = build_language_backbone(
    LLMBackboneConfig(name="qwen3-1.7b", finetune="lora", lora={"r": 16, "alpha": 32}),
    special_tokens=["<image>"],       # added BEFORE fine-tuning is configured (see below)
)
out = llm(inputs_embeds=embeds, attention_mask=mask, labels=labels)
out.loss, out.logits, out.hidden_states   # hidden_states: what VLA action / video heads read
```

## Which LLMs are registered

Only **fine-tunable causal decoders** qualify: the VLM / VLA objectives are next-token
prediction over a sequence with image / state tokens spliced in. A model is registered only if it
is a dense (non-MoE) causal decoder in the **8B class** (at most 9B parameters in the language
model that is loaded) that loads with `transformers` alone (no `trust_remote_code`). Every Hub id
was checked for existence, architecture, gating and licence, and every architecture was
instantiated from the repo's own config with the installed `transformers`. The parameter counts
below come from those instantiations.

### Natively multimodal checkpoints (text towers)

Qwen3.5, Gemma 4, Ministral 3, Cosmos-Reason2, LFM2.5-VL and SmolVLM2 are published as
image-text models (`*ForConditionalGeneration`). They are pre-trained on interleaved image-text
data, so their language decoder carries a multimodal prior that a text-only LLM of the same size
lacks. For these `modality: vision-language` entries, `hale_core.nn.llm.hf_text_tower` loads the
checkpoint and lifts out the **text tower** (decoder + LM head, moved rather than copied, so tied
embeddings stay tied). It uses the architecture's own `*ForCausalLM` class where `transformers`
has one, which keeps details such as Gemma's logit soft-capping, and otherwise
`TextTowerForCausalLM` (Qwen3-VL / Cosmos-Reason2). The checkpoint's vision / audio towers and
projector are discarded: Hale-VLM and HALE-WAM attach their configured vision encoder and
projector instead. Tests check that the extracted tower reproduces the multimodal model's logits
on text exactly, for all six architectures.

### Hybrid attention

| layout | models | what it means |
|---|---|---|
| `full` | most | every layer attends over the whole sequence |
| `sliding_hybrid` | Gemma 4 | sliding-window layers interleaved with global layers |
| `linear_hybrid` | Qwen3.5 | Gated DeltaNet linear-attention layers interleaved with full attention 3:1 |
| `conv_hybrid` | LFM2.5-VL | gated short-convolution layers interleaved with attention |

All of them work as VLM / VLA backbones, because the models read hidden states. The default LoRA
targets reach every layer, including the linear-attention and convolution layers (tested). A
π0-style action expert that attends to the VLM's per-layer KV cache, however, only finds a KV
cache in the full-attention layers of `linear_hybrid` / `conv_hybrid` models. Such an expert must
attend to those layers only, or read pooled hidden states (as Halo-VLA's dual-system latents do).
Qwen3.5 runs a slow PyTorch fallback unless `flash-linear-attention` and `causal-conv1d` are
installed.

### Gemma 4: pass `token_ids`

Gemma 4's "E" models add a per-layer embedding looked up from each position's token id
(2.4B / 2.9B of E2B's / E4B's parameters). Called with `inputs_embeds`, Gemma reverse-looks-up
the ids from the embeddings, which fails once image / state embeddings are spliced in. Pass the
ids aligned with `inputs_embeds` as `forward(..., token_ids=ids)`. Ids beyond the per-layer
table (special tokens added for the VLM / VLA) use the padding id. Hale-VLM and HALE-WAM pass
`token_ids` for every backbone; the others ignore it.

### Picks

| goal | backbone |
|---|---|
| VLA, most capability at ≤ 5B total | `qwen3.5-4b-base` or `gemma4-e4b` + a flow-matching action expert |
| VLA, edge / high control rate | `cosmos-reason2-2b` (GR00T N1.7's backbone) or `qwen3.5-2b-base`; `lfm2.5-vl-1.6b` |
| drop-in for π0-style per-layer KV experts | `gemma4-*`, `ministral3-*` (no linear-attention layers) |
| fully reproducible research VLM | `smollm3-3b-base` (open data) |
| tiny / fast iteration | `smollm2-135m`, `smolvlm2-256m`, `qwen3.5-0.8b-base`, `lfm2.5-vl-450m` |

From HaloDiffusionLLM only the causal variants qualify (`autoregressive`, `mtp`). Masked diffusion,
block diffusion and flow matching need noise / time inputs and bidirectional or doubled-stream
attention, so they are rejected with an explanation. Any other Hub causal LM can still be used as
`name: custom` with `model_id:`.

| name | family | source / repo | LM params | modality | attention | LoRA targets | notes |
|---|---|---|---|---|---|---|---|
| `deepseek-r1-distill-qwen-1.5b` | deepseek-r1 | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.8B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | reasoning |
| `deepseek-r1-distill-qwen-7b` | deepseek-r1 | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.6B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | reasoning |
| `gemma2-2b` | gemma2 | [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | 2.6B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence, use attn_implementation: eager (logit soft-capping) |
| `gemma3-1b` | gemma3 | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) | 1B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence, text-only Gemma 3; larger Gemma 3 sizes are multimodal and not registered |
| `granite4.2-3b` | granite | [`ibm-granite/granite-4.2-3b`](https://huggingface.co/ibm-granite/granite-4.2-3b) | 3.66B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `granite4.2-8b` | granite | [`ibm-granite/granite-4.2-8b`](https://huggingface.co/ibm-granite/granite-4.2-8b) | 8.79B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | 128K context; 8.8B with its untied output embeddings |
| `llama3.2-1b` | llama3 | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1.2B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `llama3.2-3b` | llama3 | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `llama3.1-8b` | llama3 | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `mistral-7b` | mistral | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.2B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `phi3.5-mini` | phi3 | [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | text | full | `qkv_proj o_proj gate_up_proj down_proj` |  |
| `phi4-mini` | phi4 | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B | text | full | `qkv_proj o_proj gate_up_proj down_proj` |  |
| `phi4-mini-reasoning` | phi4 | [`microsoft/Phi-4-mini-reasoning`](https://huggingface.co/microsoft/Phi-4-mini-reasoning) | 3.84B | text | full | `qkv_proj o_proj gate_up_proj down_proj` | reasoning |
| `qwen2.5-0.5b` | qwen2.5 | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.49B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-1.5b` | qwen2.5 | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-3b` | qwen2.5 | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3.1B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-7b` | qwen2.5 | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.6B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-0.6b` | qwen3 | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-1.7b` | qwen3 | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-4b` | qwen3 | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | 4B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-8b` | qwen3 | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smollm2-135m` | smollm2 | [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 0.135B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | same cosmo2 tokenizer as the built-in Halo-VLA decoder |
| `smollm2-360m` | smollm2 | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 0.36B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | SmolVLA / SmolVLM language backbone size |
| `smollm2-1.7b` | smollm2 | [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smollm3-3b` | smollm3 | [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.08B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | hybrid /think and /no_think chat template; 3 of 4 layers use NoPE |
| `smollm3-3b-base` | smollm3 | [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) | 3.08B | text | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base checkpoint, preferred for VLM / VLA pre-training |
| `cosmos-reason2-2b` | cosmos-reason2 | [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | 1.72B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence, NVIDIA Open Model License, Qwen3-VL-2B architecture; accept the licence on the Hub |
| `gemma4-e2b` | gemma4 | [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) | 4.63B | text tower of a VLM | sliding_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base; 2.4B of the 4.6B are per-layer embedding tables (~2B effective); pass token_ids |
| `gemma4-e2b-it` | gemma4 | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) | 4.63B | text tower of a VLM | sliding_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | instruction-tuned E2B; pass token_ids |
| `gemma4-e4b` | gemma4 | [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) | 7.46B | text tower of a VLM | sliding_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base; 2.9B of the 7.5B are per-layer embedding tables (~4B effective); pass token_ids |
| `gemma4-e4b-it` | gemma4 | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | 7.46B | text tower of a VLM | sliding_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | instruction-tuned E4B; pass token_ids |
| `lfm2.5-vl-450m` | lfm2.5-vl | [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) | 0.35B | text tower of a VLM | conv_hybrid | `q_proj k_proj v_proj out_proj in_proj w1 w2 w3` | LFM Open License |
| `lfm2.5-vl-1.6b` | lfm2.5-vl | [`LiquidAI/LFM2.5-VL-1.6B`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | 1.17B | text tower of a VLM | conv_hybrid | `q_proj k_proj v_proj out_proj in_proj w1 w2 w3` | LFM Open License |
| `lfm2.5-vl-3b` | lfm2.5-vl | [`LiquidAI/LFM2.5-VL-3B`](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) | 2.7B | text tower of a VLM | conv_hybrid | `q_proj k_proj v_proj out_proj in_proj w1 w2 w3` | LFM Open License |
| `ministral3-3b` | ministral3 | [`mistralai/Ministral-3-3B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512) | 3.43B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base checkpoint |
| `ministral3-3b-instruct` | ministral3 | [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) | 3.43B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `ministral3-8b` | ministral3 | [`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512) | 7.95B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base checkpoint |
| `ministral3-8b-instruct` | ministral3 | [`mistralai/Ministral-3-8B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | 8.49B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3.5-0.8b` | qwen3.5 | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.75B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `qwen3.5-0.8b-base` | qwen3.5 | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | 0.75B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | base checkpoint, preferred for VLA pre-training; install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `qwen3.5-2b` | qwen3.5 | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | 1.88B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `qwen3.5-2b-base` | qwen3.5 | [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | 1.88B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | base checkpoint, preferred for VLA pre-training; install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `qwen3.5-4b` | qwen3.5 | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.21B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `qwen3.5-4b-base` | qwen3.5 | [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | 4.21B | text tower of a VLM | linear_hybrid | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z out_proj` | base checkpoint, preferred for VLA pre-training; install flash-linear-attention + causal-conv1d for fast linear-attention kernels |
| `smolvlm2-256m` | smolvlm2 | [`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | 0.16B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smolvlm2-500m` | smolvlm2 | [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | 0.41B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smolvlm2-2.2b` | smolvlm2 | [`HuggingFaceTB/SmolVLM2-2.2B-Base`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Base) | 1.81B | text tower of a VLM | full | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | base checkpoint |
| `hale-autoregressive` | hale | HaloDiffusionLLM `autoregressive` checkpoint | from checkpoint | text | full | all block linears (auto) | needs `checkpoint:`; size comes from the checkpoint |
| `hale-mtp` | hale | HaloDiffusionLLM `mtp` checkpoint | from checkpoint | text | full | all block linears (auto) | needs `checkpoint:`; the main next-token head is used, extra heads are dropped |

Gated models (Llama, Gemma 2/3, Cosmos-Reason2) need their licence accepted on the Hub and `HF_TOKEN` set.

### Considered and not registered

| model | reason |
|---|---|
| `Qwen/Qwen3.5-9B` | 9.4B language model: above the 8B-class limit |
| `IFM/K2-Horizon-0.9B/3.7B/7B` | needs trust_remote_code (k2_horizon is not in transformers) |
| `nvidia/Cosmos3-Edge` | omnimodal Mixture-of-Transformers world-action model (diffusion tower for video / actions): a WAM backbone, not a causal LM |
| `Wan-AI/Wan2.2-TI2V-5B` | video diffusion transformer: a WAM backbone, not a causal LM |
| `google/gemma-4-12B / 26B-A4B / 31B` | above 8B, or MoE |
| `mistralai/Ministral-3-*-Instruct-2512` | FP8-quantised weights; the -BF16 repos are registered instead |

World-action models (Cosmos3-Edge, Wan2.2) are video-generation backbones. In HALE-WAM they would replace the DiT world-model head, not the language model.

## Fine-tuning policy

| `finetune` | trains | typical use |
|---|---|---|
| `lora` (default) | LoRA adapters `W + (α/r)·B·A` on the target projections (`B = 0` at init, so training starts from the pretrained model exactly) | most runs: 1-5 % of the LLM's parameters |
| `full` | every LLM weight | small backbones (SmolLM2, Qwen 0.5B) with enough memory |
| `frozen` | nothing in the LLM | train only the connector / heads |

`lora.train_embeddings: true` also trains the token embeddings and LM head. This is needed when new
special tokens must be learned: the VLA's `<halo_action>` / `<halo_world_video>` keep their
embedding in the sequence, so HALE-WAM turns it on by default.

**Order matters.** `build_language_backbone` adds special tokens first and applies the policy
second. Resizing the vocabulary creates new embedding parameters, so resizing *after* LoRA would
leave the whole embedding table trainable (≈ 620 M parameters for Qwen3-8B). Hale-VLM used to do
this when adding `<image>`; it now goes through this factory. New rows are initialised from the
existing embeddings' distribution rather than at random.

## Interface

| member | meaning |
|---|---|
| `tokenizer`, `hidden_size`, `vocab_size` | the backbone's vocabulary and width |
| `embed_tokens(ids)` | input embeddings (positions are added inside `forward`) |
| `forward(inputs_embeds=…, attention_mask=…, labels=…, token_ids=…)` | `LanguageBackboneOutput(logits, hidden_states, loss)`; `token_ids` only matters for Gemma 4 |
| `add_special_tokens(tokens)` | add + resize, returns ids |
| `default_lora_targets()`, `apply_lora(...)`, `embedding_parameters()` | fine-tuning hooks |

* `HFCausalLMBackbone` wraps `AutoModelForCausalLM` (or a multimodal checkpoint's text tower); LoRA via PEFT `get_peft_model`, so adapters
  save with `save_pretrained` and merge with `merge_and_unload()`.
* `HaleSequenceLMBackbone` rebuilds a HaloDiffusionLLM `SequenceBackbone` from its training
  checkpoint (config, weights, tokenizer name) and runs it over `inputs_embeds`. LoRA is injected
  in place on every block linear except MoE routers. Tests check that its logits match the native
  forward exactly.

## Where it is used

* **Hale-VLM** - `model.llm.backbone: <name>` (+ `use_lora`, `freeze_llm`, `lora_*`,
  `checkpoint`, `lora_train_embeddings`); the data module uses `load_backbone_tokenizer`.
* **HALE-WAM** - `--llm_backbone <name>` (or `HaloVLMConfig.llm_backbone`) replaces the built-in
  decoder; see HALE-WAM `docs/llm_backbone.md`.
