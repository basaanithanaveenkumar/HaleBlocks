# HaleBlocks

Reusable transformer building blocks for language models, VLMs, and world models.

Import as `hale_core`:

```python
from hale_core.nn.layers import build_attention, build_attn_mask
from hale_core.nn.stacks import GPTStack, LGTStack
from hale_core.nn.backbones import build_backbone
from hale_core.nn.losses import token_nll, register_token_loss
from hale_core.config import load_config
from hale_core.registry import register_model, register_loss, build_logger, get_trainer
```

## Install

```bash
uv add "hale-blocks @ git+https://github.com/basaanithanaveenkumar/HaleBlocks.git"
uv add --editable /path/to/HaleBlocks
```

## Layout

```text
hale_core/
  registry/       plugin infrastructure (NamedRegistry, VariantRegistry)
  config/         RunConfig, YAML load, experiment paths
  logging/        loguru + experiment backends (tensorboard, wandb)
  training/       Trainer, schedule, distributed
  nn/
    layers/       attention, FFN, norm, embeddings
    stacks/       GPT, LGT, DiT depth modules
    backbones/    token id -> logits
    losses/       token losses (ce, focal, …)
    optim/        optimizer wrappers
  runtime/        checkpoint, tensors, device, protocols
```

Legacy imports (`hale_core.components`, `hale_core.transformers`, …) remain as thin shims.

## Develop

```bash
uv sync --extra dev
uv run pytest -q
```
