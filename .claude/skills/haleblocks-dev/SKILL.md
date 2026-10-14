---
name: haleblocks-dev
description: Set up, navigate, test and release HaleBlocks (`hale_core`), the shared transformer-components, config, registry and training library used by Hale-VLM and Hale-LLM. Use when starting work in this repo, changing a public API that downstream repos import, or running the pre-commit/test suite.
---

# HaleBlocks development

Distribution `hale-blocks`, import name `hale_core`. Downstream repos (Hale-VLM, Hale-LLM)
install it from git, so **public import paths are an API**. Keep re-exports in the
`__init__.py` files stable, or update the dependants in the same change.

## Environment

```bash
uv sync --extra dev                 # + --extra llm / --extra vlm for dataset streaming
uv run pre-commit install           # ruff lint + format, unit + smoke tests on commit
uv run pytest -q                    # 114 tests, about 4 s on CPU
uv run pytest tests/unit -q ; uv run pytest tests/smoke -q -m smoke ; uv run pytest tests/integration -q -m integration
```

Python ≥ 3.12 (the code uses PEP 695 generics such as `def f[T: BaseModel](...)`).

## Layout

| Package | Contents |
|---|---|
| `nn/layers/` | attention (`mha`, `gqa`, `mqa`; RoPE, qk-norm, sliding window), masks, FFN (`mlp`, `geglu`, `moe`), `RMSNorm`, embeddings, `TransformerLayer`, `DiTBlock`/`DiTFinalLayer` |
| `nn/stacks/` | `GPTStack` (pre-LN), `LGTStack` (RMSNorm, GQA, 5:1 local/global, dual RoPE θ), `DiTStack` (AdaLN-Zero) |
| `nn/backbones/` | `SequenceBackbone` (embed → stack → tied head), `GPTBackbone`, `LGTBackbone`, `DiTBackbone`, `build_backbone` |
| `nn/losses/` | per-token CE, focal, KL, label smoothing; `token_nll`, `register_token_loss` |
| `nn/optim/` | SGD, Adam, AdamW, Muon (`torch.optim.Muon`) wrappers |
| `registry/` | `NamedRegistry`, `VariantRegistry`, `register_model/loss/sampler/optimizer/metric/config/logger/trainer/dataset` |
| `config/` | strict pydantic `RunConfig` sections, YAML `inherits:` merge, experiment directory layout |
| `training/` | `Trainer`, schedule, DDP/FSDP helpers |
| `logging/` | loguru setup, TensorBoard / W&B experiment logger |
| `data/` | LLM (SmolLM2, 33 datasets) and VLM registries, sequential streaming, LRU media cache |
| `runtime/` | checkpoint, device, tensor helpers, protocols |

## Conventions

- **Masks:** `True` = blocked (same as `nn.MultiheadAttention`). `build_attn_mask` supports
  `causal`, `bidirectional` (returns `None`) and `block_causal` (returns a `2n × 2n` mask
  over `[clean ; noisy]` for block diffusion; `seq_len % block_size == 0`).
- **Open/closed factories:** new implementations register by name (`register_attention`,
  `register_ffn`, `register_backbone`). Callers only use `build_*`.
- **Heads are tied** to the token embedding by default (`tie_embeddings=True`).
- **Config sections are strict:** a new YAML key needs a schema field first.

## Quick check

```python
from hale_core.nn.backbones import build_backbone
m = build_backbone(32000, arch="lgt", d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                   max_length=128, n_kv_heads=2)
m(torch.randint(0, 32000, (2, 64))).shape   # (2, 64, 32000); 13.9M params
```
