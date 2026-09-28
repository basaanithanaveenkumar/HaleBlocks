---
name: haleblocks-data
description: Use or extend the HaleBlocks data registries — 33 SmolLM2 LLM datasets (pretrain/SFT/RL, stage mixtures 1–4) and the VLM dataset registry with sequential streaming and an LRU media cache. Use when building an LLM/VLM dataloader, adding a dataset, or debugging streaming or cache behaviour.
---

# HaleBlocks data

Install streaming extras first: `uv sync --extra llm` or `--extra vlm` (or `--extra data`
for both).

## LLM (SmolLM2)

```python
from hale_core.registry import list_datasets, get_dataset
from hale_core.data import build_llm_dataloader, build_smollm2_stage_loader, SMOLLM2_PRETRAIN_STAGES

list_datasets(kind="llm", stage="pretrain")   # 15
list_datasets(kind="llm", stage="finetune")   # 14 (SmolTalk mixture)
list_datasets(kind="llm", stage="rl")         # 4 (DPO)
loader = build_llm_dataloader(stage="finetune", samples_per_dataset=1000)
stage4 = build_smollm2_stage_loader(4, samples_per_dataset=500)   # annealing mixture
```

Definitions live in `data/llm/builtins/{pretrain_web,pretrain_math,pretrain_code,pretrain_synthetic,sft,rl}.py`.
Each registered spec carries the HF path, stage and a description including its mixture
share per SmolLM2 stage.

## VLM

```python
from hale_core.data import build_vlm_dataloader, MixedVLMDataModule
loader = build_vlm_dataloader(samples_per_dataset=100, cache_max_bytes=2_000_000_000)
data = MixedVLMDataModule(samples_per_dataset=64, batch_size=4)
```

Datasets are consumed **sequentially** (onevision → m4 → mammoth → … → sharegpt4video),
each capped at `samples_per_dataset`. Remote images and videos are downloaded into
`data/vlm/cache.py::MediaCache`, an on-disk LRU cache that evicts the least recently used
files when `max_bytes` (default 2 GiB) is exceeded.

## Adding a dataset

1. Add a spec + parser in the right `builtins` module and register it with
   `@register_dataset("name")` (`hale_core.registry`). Set `kind` and `stage` so
   `list_datasets(kind=..., stage=...)` finds it.
2. Parse rows into the typed sample (`data/llm/types.py` or `data/vlm/types.py`). Return
   `None` to skip malformed rows.
3. Add a test in `tests/unit/test_llm_datasets.py` / `test_vlm_datasets.py` that uses an
   in-memory row, with no network.
4. Update the dataset lists in `README.md` and `docs/data.md`.
