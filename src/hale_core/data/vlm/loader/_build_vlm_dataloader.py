"""``build_vlm_dataloader``: build_vlm_dataloader (Sequential VLM dataloader with on-demand media caching)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hale_core.data.vlm.loader.sequential_vlm_data_loader import SequentialVLMDataLoader
from hale_core.data.vlm.types import DEFAULT_VLM_MIXTURE


def build_vlm_dataloader(
    dataset_names: Sequence[str] | None = None,
    *,
    cache_dir: str | Path = "~/.cache/hale-blocks/vlm",
    cache_max_bytes: int = 2 * 1024**3,
    samples_per_dataset: int | None = None,
    prefetch_workers: int = 4,
    prefetch_queue: int = 8,
) -> SequentialVLMDataLoader:
    """Build a :class:`SequentialVLMDataLoader` over named VLM datasets.

    Args:
        dataset_names: registry names to stream (default: the standard mixture).
        cache_dir: root of the media cache.
        cache_max_bytes: budget of the media cache (least recently used files are evicted).

    Returns:
        The loader (iterating yields :class:`VLMSample`).
    """
    names = tuple(dataset_names or DEFAULT_VLM_MIXTURE)
    return SequentialVLMDataLoader(
        dataset_names=names,
        cache_dir=Path(cache_dir),
        cache_max_bytes=cache_max_bytes,
        samples_per_dataset=samples_per_dataset,
        prefetch_workers=prefetch_workers,
        prefetch_queue=prefetch_queue,
    )
