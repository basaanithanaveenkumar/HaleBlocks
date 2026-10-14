# Getting started

```bash
git clone https://github.com/basaanithanaveenkumar/HaleBlocks && cd HaleBlocks
uv sync --extra dev
uv run pytest -q                     # 114 passed
```

Use it from another project:

```bash
uv add "hale-blocks @ git+https://github.com/basaanithanaveenkumar/HaleBlocks.git"
```

## Build and run a backbone

```python
import torch
from hale_core.nn.backbones import build_backbone

gpt = build_backbone(32000, arch="transformer", d_model=256, n_heads=8, n_layers=6,
                     d_ff=1024, max_length=128)                                   # 12.96M
moe = build_backbone(32000, arch="transformer", d_model=256, n_heads=8, n_layers=6,
                     d_ff=1024, max_length=128, ffn_type="moe",
                     moe_num_experts=8, moe_top_k=2, moe_num_shared=1)            # 52.30M
lgt = build_backbone(32000, arch="lgt", d_model=256, n_heads=8, n_layers=6,
                     d_ff=1024, max_length=128, n_kv_heads=2)                     # 13.88M
dit = build_backbone(32000, arch="dit", d_model=256, n_heads=8, n_layers=6,
                     d_ff=1024, max_length=128, attn_type="bidirectional",
                     use_time_cond=True)                                          # 15.60M

ids = torch.randint(0, 32000, (2, 64))
gpt(ids).shape                     # (2, 64, 32000)
dit(ids, t=torch.rand(2)).shape    # (2, 64, 32000)
```

## From a YAML config

```python
from hale_core.config import load_config
from hale_core.nn.backbones import build_backbone

cfg = load_config("tests/fixtures/child.yaml")      # child inherits minimal.yaml
model = build_backbone(vocab_size=32000, model_cfg=cfg.model)
```

## Develop

```bash
uv run pre-commit install          # ruff + unit/smoke tests on each commit
uv run pytest tests/unit -q
uv run pytest tests/smoke -q -m smoke
uv run pytest tests/integration -q -m integration
```
