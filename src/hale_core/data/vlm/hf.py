"""Hugging Face streaming helpers for VLM datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from hale_core.data.vlm.types import VLMDatasetSpec


def open_hf_stream(spec: VLMDatasetSpec, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "VLM datasets require the `datasets` package; install with: uv add 'hale-blocks[vlm]'"
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
        "opening VLM stream dataset={} repo={} subset={} split={}",
        spec.name,
        spec.hf_repo,
        spec.hf_subset,
        spec.split,
    )
    dataset = load_dataset(**kwargs)
    yield from dataset
