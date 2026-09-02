"""VLM dataset sample skeleton."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hale_core.data.kinds import DatasetKind, VLMModality


@dataclass
class VLMSample:
    dataset: str
    modality: VLMModality
    text: str | None = None
    images: list[Path] = field(default_factory=list)
    video: Path | None = None
    conversations: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_vision(self) -> bool:
        return bool(self.images or self.video)


@dataclass(frozen=True)
class VLMDatasetSpec:
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
        from hale_core.data.vlm.hf import open_hf_stream

        return open_hf_stream(self, cache_dir=cache_dir)


DEFAULT_VLM_MIXTURE: tuple[str, ...] = (
    "llava_onevision",
    "m4_instruct",
    "mammoth",
    "llava_video_178k",
    "finevideo",
    "videostar",
    "vript",
    "vista_400k",
    "moviechat",
    "sharegpt4video",
)
