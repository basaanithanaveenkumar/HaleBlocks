"""Hugging Face streaming for LLM datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from hale_core.data.llm.types import LLMDatasetSpec


def open_hf_stream(spec: LLMDatasetSpec, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "LLM datasets require the `datasets` package; install with: uv add 'hale-blocks[llm]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "path": spec.hf_repo,
        "split": spec.split,
        "streaming": spec.streaming,
        "cache_dir": str(cache_dir / "hf"),
    }
    if spec.hf_subset is not None:
        kwargs["name"] = spec.hf_subset

    logger.info(
        "opening LLM stream dataset={} stage={} repo={} subset={}",
        spec.name,
        spec.stage,
        spec.hf_repo,
        spec.hf_subset,
    )
    dataset = load_dataset(**kwargs)
    for row in dataset:
        if spec.source_filter is not None:
            source = row.get("source") or row.get("dataset") or row.get("subset")
            if source is not None and str(source) != spec.source_filter:
                continue
        yield row
