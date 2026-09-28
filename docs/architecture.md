# Architecture

Mermaid diagrams (they render on GitHub). The same figures appear on the
[project page](../project-page/index.html) and in the [paper](../paper/main.tex).

## 1. Package map

```mermaid
flowchart TB
  subgraph nn["hale_core.nn"]
    L["layers<br/>attention · masks · ffn · norm · embeddings · TransformerLayer · DiTBlock"]
    S["stacks<br/>GPTStack · LGTStack · DiTStack"]
    B["backbones<br/>SequenceBackbone → GPT / LGT / DiT"]
    LO["losses<br/>ce · focal · kl · label smoothing"]
    O["optim<br/>sgd · adam · adamw · muon"]
    L --> S --> B
  end
  C["config<br/>strict RunConfig · YAML inherits · experiment layout"]
  R["registry<br/>NamedRegistry · VariantRegistry · register_*"]
  T["training<br/>Trainer · schedule · DDP/FSDP"]
  G["logging<br/>loguru · TensorBoard · W&B"]
  D["data<br/>SmolLM2 LLM registry · VLM registry · MediaCache"]
  RT["runtime<br/>checkpoint · device · tensors · protocols"]
  C --> R --> T
  B --> R
  LO --> R
  O --> R
  D --> R
  G --> T
  RT --> T
```

## 2. Forward contract

```mermaid
sequenceDiagram
  participant U as caller
  participant BB as SequenceBackbone
  participant ST as Stack
  participant LY as Layer / DiTBlock
  participant AT as Attention
  participant FF as FFN
  U->>BB: token_ids, attention_mask, t
  BB->>BB: embed (+ learned positions), time embedding
  BB->>BB: attn_mask (causal / bidirectional / block_causal), key_padding_mask
  BB->>ST: h, t_emb, attn_mask, key_padding_mask, positions
  loop n_layers
    ST->>LY: same arguments
    LY->>AT: norm(h), attn_mask, key_padding_mask, positions
    AT-->>LY: residual update
    LY->>FF: norm(h)
    FF-->>LY: residual update
  end
  ST-->>BB: h
  BB-->>U: logits = tied LM head(h)
```

## 3. The three stacks

```mermaid
flowchart LR
  subgraph GPT["GPTStack"]
    g1["LayerNorm → MHA/GQA → +"] --> g2["LayerNorm → MLP/GeGLU/MoE → +"] --> g3["× n, final LayerNorm"]
  end
  subgraph LGT["LGTStack (Gemma-3 style)"]
    l1["5 local layers<br/>sliding window, RoPE θ=10⁴<br/>GQA n_kv = n_kv_heads"] --> l2["1 global layer<br/>full attention, RoPE θ=10⁶, p-RoPE 25%<br/>GQA n_kv = n_heads/8"] --> l3["repeat · RMSNorm · GeGLU"]
  end
  subgraph DiT["DiTStack"]
    d1["t → sinusoidal MLP → t_emb"] --> d2["AdaLN-Zero: shift/scale/gate<br/>for attention and FFN"] --> d3["× n (zero-init = identity at start)"]
  end
```

## 4. Block-causal mask (block diffusion)

Clean copy `x^c` and noisy copy `x^n` are concatenated (length `2n`). Block index `b(i)`:

```mermaid
flowchart LR
  C1["clean block k"] -->|"b(j) ≤ b(i)"| C0["clean blocks ≤ k"]
  N1["noisy block k"] -->|"b(j) < b(i)"| C2["clean blocks &lt; k"]
  N1 -->|"b(j) = b(i)"| N2["noisy block k (itself)"]
```

Example (`n = 4`, `block_size = 2`; `1` = blocked):

```
            c0 c1 c2 c3 | n0 n1 n2 n3
clean  0 :   0  0  1  1 |  1  1  1  1
clean  2 :   0  0  0  0 |  1  1  1  1
noisy  0 :   1  1  1  1 |  0  0  1  1
noisy  2 :   0  0  1  1 |  1  1  0  0
```

## 5. From YAML to a trained model

```mermaid
flowchart LR
  Y["child.yaml<br/>inherits: base.yaml"] --> M["deep merge"] --> V["validate against<br/>registered schema (strict)"]
  V --> RC["RunConfig"]
  RC --> RM["get_model(variant)"]
  RC --> RL["get_loss(variant)"]
  RC --> RO["build_optimizer(type)"]
  RC --> RD["get_dataset / data module"]
  RM & RL & RO & RD --> TR["Trainer.fit()<br/>resume · epochs · eval · viz · checkpoints"]
  TR --> EX["experiments/&lt;variant&gt;/&lt;run&gt;/<br/>config dump · logs · ckpts · latest →"]
```
