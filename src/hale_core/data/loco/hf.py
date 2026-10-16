"""Hugging Face streaming for locomotion / whole-body-control datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from hale_core.exceptions import MissingDependencyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hale_core.data.loco.types import LocoDatasetSpec


def open_loco_hf_stream(spec: LocoDatasetSpec, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
    """Stream the rows of ``spec.hf_repo`` (``split``, optionally streaming) from the HF Hub.

    ``LocoDatasetSpec.open_stream`` has always delegated here; the module was missing, so
    any locomotion dataset with an ``hf_repo`` failed with ``ModuleNotFoundError``.

    Args:
        spec: the dataset spec; ``hf_repo`` must be set.
        cache_dir: root of the local cache (HF files go to ``cache_dir / "hf"``).

    Yields:
        One raw row dict per sample.

    Raises:
        MissingDependencyError: if the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise MissingDependencyError("datasets", "install with: uv add 'hale-blocks[vlm]'") from exc
    logger.info("opening locomotion stream dataset={} repo={} split={}", spec.name, spec.hf_repo,
                spec.split)
    yield from load_dataset(path=spec.hf_repo, split=spec.split, streaming=spec.streaming,
                            cache_dir=str(cache_dir / "hf"))
