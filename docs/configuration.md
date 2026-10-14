# Configuration

`RunConfig` is assembled from strict sections (`extra="forbid"`): `model`, `train`, `data`,
`eval`, `sample`, `viz`, `logging`, `experiment`. Downstream packages subclass sections and
register their schema with `@register_config(name)`.

## YAML inheritance

```yaml
# child.yaml
inherits: minimal.yaml       # deep-merged first; child keys win
variant: mock_ar
model:
  d_model: 48
```

`load_config(path)` → validated `RunConfig`. The experiment layout helper
(`config/experiment.py`) creates a run directory, copies the source YAMLs, dumps the resolved
config and updates a `latest` pointer.

## `ModelConfig`

| Field | Default | Meaning |
|---|---|---|
| `d_model`, `n_heads`, `n_layers`, `d_ff` | 128, 4, 4, 512 | size |
| `dropout`, `max_length` | 0.0, 64 | |
| `arch` | `default` | `default`, `transformer`, `lgt`, `dit` |
| `attn_type` | `causal` | `causal`, `bidirectional`, `block_causal` |
| `block_size` | null | required for `block_causal` |
| `attn_impl`, `n_kv_heads` | `mha`, null | `mha`, `gqa`, `mqa` |
| `ffn_type` | `mlp` | `mlp`, `geglu`, `moe` |
| `moe_num_experts`, `moe_top_k`, `moe_num_shared` | 4, 2, 1 | |
| `use_time_cond` | false | timestep conditioning (diffusion / flow) |
| `n_mtp_heads` | 2 | multi-token-prediction heads (downstream) |
| `sliding_window`, `local_global_ratio` | 512, 5 | LGT |
| `rope_theta_local`, `rope_theta_global`, `p_rope` | 1e4, 1e6, 0.25 | LGT |
| `qk_norm` | false | |
