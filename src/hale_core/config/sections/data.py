from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.common import StrictModel
from hale_core.config.sections.dataloader import DataLoaderConfig


class DataConfig(StrictModel):
    """Dataset source and DataLoader configuration.

    YAML example::

        data:
          source: huggingface
          dataset: "Salesforce/wikitext"
          subset: "wikitext-2-raw-v1"
          loader:
            num_workers: 4
            pin_memory: true
            drop_last: true
    """

    # ── Source ────────────────────────────────────────────────────────────────
    source: Literal["huggingface", "overfit"] = "huggingface"
    dataset: str = "Salesforce/wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    cache_dir: str | None = None
    tokenizer_name: str = "gpt2"

    # ── Size limits ───────────────────────────────────────────────────────────
    train_size: int | None = None
    val_size: int | None = None

    # ── Overfit mode ──────────────────────────────────────────────────────────
    overfit_text: str | None = None
    n_overfit_copies: int = 64

    # ── Tokenisation ──────────────────────────────────────────────────────────
    add_special_tokens: bool = False
    stride_words: int = 10

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
