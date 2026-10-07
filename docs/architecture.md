# Architecture

Mermaid diagrams (render on GitHub). See also the [project page](../project-page/index.html)
and the [README](../README.md) for a laymen overview.

---

## 1. Registry system

HaleBlocks uses two registry types. `NamedRegistry` maps a string key to a class or function. `VariantRegistry` also stores the config schema for each variant, enabling strict per-variant validation.

```mermaid
flowchart TB
  subgraph NAMED["NamedRegistry — simple name → class"]
    NR_DEC["@register_model('gpt_small')\nclass GPTSmall(nn.Module)"]
    NR_LOSS["@register_loss('ce')\ndef cross_entropy_loss(…)"]
    NR_SAMP["@register_sampler('autoregressive')\nclass ARSampler"]
    NR_MET["@register_metric('perplexity')\nclass PerplexityMetric"]
  end

  subgraph VARIANT["VariantRegistry — name + config schema"]
    VR_VLM["@register_vlm_model('qwen3_8b_vlm')\nclass HaleVLM + VLMModelConfig schema"]
    VR_DS["@register_vlm_dataset('laion_coco')\nclass LAIONCOCOAdapter + DatasetSpec schema"]
  end

  subgraph LOOKUP["At runtime (from YAML key)"]
    BUILD_M["MODELS.build('gpt_small', cfg)"]
    BUILD_L["LOSSES.build('ce')"]
    BUILD_D["VLM_DATASETS.build('laion_coco', spec)"]
  end

  NR_DEC & NR_LOSS --> NAMED
  VR_VLM & VR_DS --> VARIANT
  NAMED & VARIANT --> LOOKUP
```

---

## 2. Config system — strict pydantic with inheritance

```mermaid
flowchart LR
  BASE["base.yaml\nd_model: 384\nn_layers: 16\nlr: 3e-4\nbatch_size: 32\n…"]

  EXP["experiment.yaml\ninherits: base.yaml\nn_layers: 24\n← overrides just this field"]

  subgraph LOAD["load_config(path)"]
    MERGE["deep merge:\nbase fields + override fields\n(later keys win)"]
    PARSE["pydantic RunConfig\n(unknown fields → ValidationError)"]
    PATH["resolve experiment paths\n(data_dir, checkpoint_dir, log_dir)"]
    MERGE --> PARSE --> PATH
  end

  BASE & EXP --> LOAD --> CFG["RunConfig object\nall fields validated\nno dynamic access needed"]
```

---

## 3. Transformer building blocks

```mermaid
flowchart TB
  subgraph LAYER["One transformer layer"]
    H_IN["hidden states h\n[B, S, D]"]

    subgraph ATTN["Attention sub-layer"]
      NORM1["RMSNorm or LayerNorm"]
      ATTN_MOD["build_attention(variant, …)\nMHA: all heads attend full sequence\nGQA: Q heads >> KV heads (grouped)\nMQA: single shared KV head"]
      ROPE["RoPE position encoding\nθᵢ = base⁻²ⁱ/ᵈ  (applied to Q,K)"]
      MASK["build_attn_mask(type)\ncausal / bidirectional / block-causal"]
      ATT_OUT["attended output [B, S, D]"]
      NORM1 --> ATTN_MOD
      ROPE --> ATTN_MOD
      MASK --> ATTN_MOD
      ATTN_MOD --> ATT_OUT
    end

    subgraph FFN_S["FFN sub-layer"]
      NORM2["RMSNorm or LayerNorm"]
      FFN_MOD["FeedForward(variant, D, D_ff)\nmlp:   Linear → GELU → Linear\ngeglu: gate · up, split activation\nmoe:   DeepSeekMoE routing"]
      FFN_OUT["[B, S, D]"]
      NORM2 --> FFN_MOD --> FFN_OUT
    end

    RESID1["residual add"]
    RESID2["residual add"]
    H_OUT["h_out [B, S, D]"]

    H_IN --> NORM1
    ATT_OUT --> RESID1
    RESID1 --> NORM2
    FFN_OUT --> RESID2 --> H_OUT
  end
```

---

## 4. Transformer stacks

```mermaid
flowchart LR
  subgraph GPT["GPTStack — autoregressive"]
    GP_IN["token ids [B, S]"]
    GP_EMB["embedding → [B, S, D]\n+ learnable position emb"]
    GP_LAYERS["N × GPT layer\ncausal mask, single RoPE"]
    GP_OUT["hidden states [B, S, D]"]
    GP_IN --> GP_EMB --> GP_LAYERS --> GP_OUT
  end

  subgraph LGT["LGTStack — local-global (for diffusion)"]
    LG_IN["hidden states or embeddings [B, S, D]"]
    LG_LYR["N × LGT layer:\n  local heads: window w, local RoPE\n  global heads: full sequence, global RoPE\n  outputs concatenated → projection\n→ [B, S, D]"]
    LG_OUT["hidden states [B, S, D]"]
    LG_IN --> LG_LYR --> LG_OUT
  end

  subgraph DIT_S["DiTStack — for world models"]
    DT_IN["noisy input x_t [B, S, D]\nconditioning c [B, D]"]
    DT_LYR["N × DiT layer:\n  adaLN-Zero(c): predict\n  (γ,β,α) from c via Linear\n  self-attn → FFN with gating"]
    DT_OUT["velocity prediction v [B, S, D]"]
    DT_IN --> DT_LYR --> DT_OUT
  end
```

---

## 5. Sequential streaming data loader

```mermaid
sequenceDiagram
  participant C as Caller (Trainer)
  participant S as SequentialMultiDatasetStream
  participant P as Prefetch thread
  participant H as HF Hub / local disk

  Note over S: dataset_list = [ds1, ds2, ds3, …]

  S->>P: start warm_up(ds2)
  P->>H: resolve metadata, open iterable stream for ds2
  loop stream ds1 (up to max_samples_per_dataset)
    S->>H: yield next row from ds1
    S-->>C: sample (dict)
  end
  S->>P: wait for ds2 warm-up to finish
  S->>P: start warm_up(ds3)
  loop stream ds2 (up to max_samples_per_dataset)
    S->>H: yield next row from ds2
    S-->>C: sample (dict)
  end
  Note over S: never holds more than 1 dataset in memory
```

---

## 6. LLM dataset registry (SmolLM2 recipe)

```mermaid
flowchart TB
  subgraph PRETRAIN_LLM["Pretrain — 15 datasets (stages 1-4)"]
    PT1["Stage 1: FineWeb-Edu, DCLM-Baseline\n(web text, filtered for quality)"]
    PT2["Stage 2: + Stack-v2, The-Stack\n(code corpora)"]
    PT3["Stage 3: + OpenWebMath, Proof-Pile-2\n(math corpora)"]
    PT4["Stage 4: + Wikipedia, Wikibooks, arXiv\n(high-quality reference text)"]
  end

  subgraph SFT_LLM["SFT — 14 datasets"]
    SFT1["SmolTalk (disabled — see rejected_datasets.py)"]
    SFT2["OpenAssistant, Alpaca, FLAN\n(instruction following)"]
    SFT3["CodeAlpaca, MagicCoder\n(code instructions)"]
    SFT4["MetaMath, NuminaMath\n(math reasoning)"]
  end

  subgraph RL_LLM["RL/DPO — 4 datasets"]
    RL1["UltraFeedback (preference pairs)"]
    RL2["Argilla-Capybara-DPO"]
    RL3["HH-RLHF (helpful + harmless)"]
    RL4["SHP (StackExchange human preferences)"]
  end

  subgraph ACCESS["Access via registry"]
    LIST["list_datasets(kind='llm', stage='pretrain')"]
    LOADER["build_llm_dataloader(stage, samples_per_dataset)"]
    STAGE["build_smollm2_stage_loader(stage_number)"]
  end

  PRETRAIN_LLM & SFT_LLM & RL_LLM --> ACCESS
```

---

## 7. Trainer loop

```mermaid
sequenceDiagram
  participant D as DataLoader
  participant M as Model (from registry)
  participant L as Loss fn (from registry)
  participant O as Optimizer (AdamW)
  participant AMP as torch.cuda.amp.GradScaler

  loop every micro-batch
    D->>M: batch (device)
    M-->>L: logits / hidden states / predictions
    D->>L: labels / targets
    L-->>O: loss / grad_accum_steps (scaled)
    AMP->>O: unscale gradients
    O->>M: clip grad norm ≤ max_grad_norm
    O->>M: optimizer.step()  (every grad_accum_steps)
    O->>O: scheduler.step()
  end

  Note over O: Checkpoint: model + optimizer + scheduler + scaler state
```
