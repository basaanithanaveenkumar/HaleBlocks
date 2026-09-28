# Components

All factories live in `hale_core.nn.*` and raise a `KeyError` listing the registered names
if you ask for an unknown one.

## Attention — `build_attention(impl, *, d_model, n_heads, dropout, n_kv_heads, rope_theta, rotary_dim, qk_norm)`

| `impl` | Class | Notes |
|---|---|---|
| `mha` | `TorchMultiHeadAttention` | standard multi-head |
| `gqa` | `GroupedQueryAttention` | `n_kv_heads` (default `n_heads // 4`), optional RoPE (`rope_theta`, `rotary_dim`) and QK-norm |
| `mqa` | `GroupedQueryAttention` | `n_kv_heads = 1` |

Forward: `attn(x, attn_mask=None, key_padding_mask=None, positions=None)`.

## Masks — `build_attn_mask(attn_type, seq_len, device, block_size=None)`

Bool, `True` = blocked. `causal` → upper-triangular `[T, T]`. `bidirectional` → `None`.
`block_causal` → `[2T, 2T]` over `[clean ; noisy]` (see [architecture](architecture.md#4-block-causal-mask-block-diffusion)).
`sliding_window_mask` and `or_masks` compose local attention.

## FFN — `build_ffn(kind, *, d_model, d_ff, dropout, moe_num_experts, moe_top_k, moe_num_shared)`

| `kind` | Class | Notes |
|---|---|---|
| `mlp` | `DenseMLP` | Linear → GELU → Linear |
| `geglu` | `GeGLU` | gated GELU |
| `moe` | `DeepseekMoE` | `moe_num_shared` shared + `moe_num_experts` routed SwiGLU experts, noisy top-k router; no aux loss |

## Norms and embeddings

`RMSNorm`, `nn.LayerNorm`; `SinusoidalTimeEmbedding` for diffusion time; learned absolute
positions on `SequenceBackbone` (`learned_pos=True`), RoPE inside GQA.

## Layers

- `TransformerLayer(d_model, attention, ffn, use_time_cond, norm="layer"|"rms", post_norm)`:
  pre-norm residual block, with optional timestep scale/shift of both norms (zero-init).
- `DiTBlock(d_model, attention, ffn)`: AdaLN-Zero (shift, scale and gate × 2, zero-init).
- `DiTFinalLayer(d_model, vocab_size)`: modulated LayerNorm + head.

## Stacks

| Stack | Norm | Attention | FFN default | Positions |
|---|---|---|---|---|
| `GPTStack` | LayerNorm | `attn_impl` | `mlp` | learned (backbone) |
| `LGTStack` | RMSNorm | GQA; 5 local (window 512, θ=10⁴) : 1 global (θ=10⁶, p-RoPE 0.25, n_kv = n_heads/8) | `geglu` | RoPE |
| `DiTStack` | LayerNorm + AdaLN-Zero | `attn_impl` | `mlp` | learned (backbone) |

## Backbones — `build_backbone(vocab_size, model_cfg=None, **overrides)`

`arch`: `default`/`transformer` → `GPTBackbone`, `lgt` → `LGTBackbone`, `dit` → `DiTBackbone`.
Forward: `backbone(token_ids, attention_mask=None, t=None, attn_mask=None, pos_ids_len=None) -> logits`.
The LM head is tied to the token embedding (`tie_embeddings=True`).

## Losses — `hale_core.nn.losses`

`token_nll`, `model_token_nll`, `per_token_cross_entropy`, `reduce_per_token`;
registered token losses: `ce`, `focal`, `kl`, `label_smoothing`; extend with
`@register_token_loss(name)`.

## Optimisers — `hale_core.nn.optim`

`sgd`, `adam`, `adamw`, `muon` (requires `torch.optim.Muon`; defaults lr 1e-3, wd 0.1,
momentum 0.95, Nesterov, 5 Newton–Schulz steps).
