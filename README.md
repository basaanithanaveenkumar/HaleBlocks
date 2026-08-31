# HaleBlocks

Reusable transformer building blocks for language models, VLMs, and world models.

Import as `hale_core`:

```python
from hale_core import GPTStack, LGTStack, build_backbone, NamedRegistry
from hale_core.components import build_attention, build_attn_mask
from hale_core.registry import register_model, register_loss
```

## Install

```bash
# from GitHub (recommended)
uv add "hale-blocks @ git+https://github.com/basaanithanaveenkumar/HaleBlocks.git"

# local editable
uv add --editable /path/to/HaleBlocks
```

## Layout

```
src/hale_core/
  components/     attention, FFN, norms, embeddings
  transformers/   GPT, LGT, DiT stacks (hidden states in/out)
  backbones/      token id -> logits wrappers
  registry/       NamedRegistry + VariantRegistry plugin system
  optimizers/     Adam, AdamW, SGD, Muon wrappers
```

## Develop

```bash
uv sync --extra dev
uv run pytest -q
```
