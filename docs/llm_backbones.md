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
is a dense (non-MoE), text-only causal LM of at most 8B parameters that loads with `transformers`
alone (no `trust_remote_code`). Every Hub id was checked for existence, architecture and gating.

From HaloDiffusionLLM only the causal variants qualify (`autoregressive`, `mtp`). Masked diffusion,
block diffusion and flow matching need noise / time inputs and bidirectional or doubled-stream
attention, so they are rejected with an explanation. Any other Hub causal LM can still be used as
`name: custom` with `model_id:`.

| name | family | source / repo | params | LoRA targets | notes |
|---|---|---|---|---|---|
| `deepseek-r1-distill-qwen-1.5b` | deepseek-r1 | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.8B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | reasoning |
| `deepseek-r1-distill-qwen-7b` | deepseek-r1 | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7.6B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | reasoning |
| `gemma2-2b` | gemma2 | [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | 2.6B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence, use attn_implementation: eager (logit soft-capping) |
| `gemma3-1b` | gemma3 | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) | 1B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence, text-only Gemma 3; larger Gemma 3 sizes are multimodal and not registered |
| `llama3.1-8b` | llama3 | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `llama3.2-1b` | llama3 | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1.2B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `llama3.2-3b` | llama3 | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | gated licence |
| `mistral-7b` | mistral | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.2B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `phi3.5-mini` | phi3 | [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | `qkv_proj o_proj gate_up_proj down_proj` |  |
| `phi4-mini` | phi4 | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B | `qkv_proj o_proj gate_up_proj down_proj` |  |
| `qwen2.5-0.5b` | qwen2.5 | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.49B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-1.5b` | qwen2.5 | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-3b` | qwen2.5 | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3.1B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen2.5-7b` | qwen2.5 | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.6B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-0.6b` | qwen3 | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-1.7b` | qwen3 | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-4b` | qwen3 | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | 4B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `qwen3-8b` | qwen3 | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smollm2-1.7b` | smollm2 | [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |  |
| `smollm2-135m` | smollm2 | [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 0.135B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | same cosmo2 tokenizer as the built-in Halo-VLA decoder |
| `smollm2-360m` | smollm2 | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 0.36B | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | SmolVLA / SmolVLM language backbone size |
| `hale-autoregressive` | hale | HaloDiffusionLLM `autoregressive` checkpoint | from checkpoint | all block linears (auto) | needs `checkpoint:`; size comes from the checkpoint |
| `hale-mtp` | hale | HaloDiffusionLLM `mtp` checkpoint | from checkpoint | all block linears (auto) | needs `checkpoint:`; the main next-token head is used, extra heads are dropped |

Gated models (Llama, Gemma) need their licence accepted on the Hub and `HF_TOKEN` set.

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
| `forward(inputs_embeds=…, attention_mask=…, labels=…)` | `LanguageBackboneOutput(logits, hidden_states, loss)` |
| `add_special_tokens(tokens)` | add + resize, returns ids |
| `default_lora_targets()`, `apply_lora(...)`, `embedding_parameters()` | fine-tuning hooks |

* `HFCausalLMBackbone` wraps `AutoModelForCausalLM`; LoRA via PEFT `get_peft_model`, so adapters
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
