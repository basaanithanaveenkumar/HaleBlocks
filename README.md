# HaleBlocks

Reusable transformer building blocks for language models, VLMs, and world models.

Import as `hale_core`:

```python
from hale_core.nn.layers import build_attention, build_attn_mask
from hale_core.nn.stacks import GPTStack, LGTStack
from hale_core.nn.backbones import build_backbone
from hale_core.nn.losses import token_nll, register_token_loss
from hale_core.config import load_config
from hale_core.registry import register_model, register_loss, build_logger, get_trainer, get_dataset, build_vlm_dataloader
from hale_core.data import MixedVLMDataModule
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
  data/           VLM dataset registry + sequential streaming loader
  training/       Trainer, schedule, distributed
  nn/
    layers/       attention, FFN, norm, embeddings
    stacks/       GPT, LGT, DiT depth modules
    backbones/    token id -> logits
    losses/       token losses (ce, focal, …)
    optim/        optimizer wrappers
  runtime/        checkpoint, tensors, device, protocols
```

## Develop

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest tests/unit -q          # unit tests
uv run pytest tests/smoke -q -m smoke
uv run pytest tests/integration -q -m integration
uv run pytest -q                     # all tests
```

### LLM datasets (SmolLM2)

Install optional streaming dependencies:

```bash
uv sync --extra llm
```

```python
from hale_core.registry import get_dataset, list_datasets
from hale_core.data import build_llm_dataloader, build_smollm2_stage_loader, SMOLLM2_PRETRAIN_STAGES

# Pretrain (15), SFT (14), RL/DPO (4) — 33 LLM datasets total
print(list_datasets(kind="llm", stage="pretrain"))
print(list_datasets(kind="llm", stage="finetune"))
print(list_datasets(kind="llm", stage="rl"))

# Stream all SFT datasets sequentially (SmolTalk mixture)
loader = build_llm_dataloader(stage="finetune", samples_per_dataset=1000)
for sample in loader:
    print(sample.dataset, sample.prompt, sample.response)

# SmolLM2 paper stage mixtures (stages 1-4)
stage1 = build_smollm2_stage_loader(1, samples_per_dataset=500)
print(SMOLLM2_PRETRAIN_STAGES[4].description)
```

Registered pretrain datasets: `fineweb_edu`, `dclm`, `openwebmath`, `infimm_webmath`,
`finemath4_plus`, `finemath3_plus`, `infi_webmath4_plus`, `infi_webmath3_plus`, `aug_gsm8k`,
`starcoderdata`, `stack_edu`, `starcoder2data`, `starcoder2_jupyter`, `cosmopedia_v2`, `dolma_books`.

SFT: `smoltalk`, `magpie_ultra`, `smol_constraint`, `smol_rewrite`, `smol_summarization`,
`numinamath_cot`, `metamathqa`, `self_oss_starcoder2_instruct`, `apigen_function_calling`,
`systemchats2`, `longalign`, `everyday_conversations`, `explore_instruct`, `openhermes25`.

RL: `ultrafeedback`, `ultrainteract`, `capybara`, `orca_dpo`.

### VLM datasets

Install optional streaming dependencies:

```bash
uv sync --extra vlm
```

```python
from hale_core.registry import get_dataset, list_datasets
from hale_core.data import build_vlm_dataloader, MixedVLMDataModule

print(list_datasets())  # llava_onevision, m4_instruct, mammoth, ...

loader = build_vlm_dataloader(
    samples_per_dataset=100,      # cap per dataset
    cache_max_bytes=2_000_000_000 # 2 GB LRU media cache
)
for sample in loader:
    # sequential: onevision -> m4 -> mammoth -> ... -> sharegpt4video
    print(sample.dataset, sample.modality, sample.text)

data = MixedVLMDataModule(samples_per_dataset=64, batch_size=4)
```

### Pre-commit

Hooks run on every commit:

- trailing whitespace / EOF / YAML-TOML checks
- `ruff` lint + format
- unit and smoke tests
