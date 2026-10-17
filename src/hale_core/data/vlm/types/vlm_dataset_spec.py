"""``VLMDatasetSpec``: VLMDatasetSpec (VLM dataset sample skeleton)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hale_core.data.kinds import DatasetKind, VLMModality
from hale_core.data.vlm.types.vlm_sample import VLMSample


@dataclass(frozen=True)
class VLMDatasetSpec:
    """Where a VLM dataset lives on the Hub, its modality, and which fields hold media/text."""
    name: str
    hf_repo: str
    modality: VLMModality
    kind: DatasetKind = DatasetKind.VLM
    hf_subset: str | None = None
    split: str = "train"
    streaming: bool = True
    text_fields: tuple[str, ...] = ("text", "caption", "answer")
    image_fields: tuple[str, ...] = ("image", "images", "image_path")
    video_fields: tuple[str, ...] = ("video", "video_path", "video_file")
    conversation_fields: tuple[str, ...] = ("conversations", "messages", "dialogue")
    row_parser: Callable[[dict[str, Any], VLMDatasetSpec], VLMSample] | None = None
    description: str = ""

    def open_stream(self, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
        """Stream this dataset's raw rows from the Hub (see :func:`open_hf_stream`)."""
        from hale_core.data.vlm.hf import open_hf_stream

        return open_hf_stream(self, cache_dir=cache_dir)
