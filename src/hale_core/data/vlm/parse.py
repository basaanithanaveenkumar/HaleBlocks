"""Row normalization into :class:`VLMSample`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hale_core.data.vlm.cache import MediaCache
from hale_core.data.vlm.types import VLMDatasetSpec, VLMModality, VLMSample


def _first_str(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for key in fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def _resolve_media(
    refs: list[str],
    *,
    cache: MediaCache,
    dataset: str,
    sample_idx: int,
    suffix: str,
) -> list[Path]:
    resolved: list[Path] = []
    for offset, ref in enumerate(refs):
        dest = cache.reserve(dataset, sample_idx * 100 + offset, suffix)
        if dest.exists():
            resolved.append(dest)
            continue
        local = cache.resolve_local(ref, dest=dest)
        if local is not None:
            resolved.append(local)
    return resolved


def default_row_parser(
    row: dict[str, Any],
    spec: VLMDatasetSpec,
    *,
    cache: MediaCache,
    sample_idx: int,
) -> VLMSample:
    if spec.row_parser is not None:
        return spec.row_parser(row, spec)

    text = _first_str(row, spec.text_fields)
    conversations = None
    for key in spec.conversation_fields:
        value = row.get(key)
        if isinstance(value, list):
            conversations = value
            break

    image_value = next((row.get(k) for k in spec.image_fields if row.get(k) is not None), None)
    images = _resolve_media(
        _as_path_list(image_value),
        cache=cache,
        dataset=spec.name,
        sample_idx=sample_idx,
        suffix=".img",
    )
    video_ref = _first_str(row, spec.video_fields)
    video = None
    if video_ref:
        dest = cache.reserve(spec.name, sample_idx, ".mp4")
        if dest.exists():
            video = dest
        else:
            video = cache.resolve_local(video_ref, dest=dest)

    modality = spec.modality
    if not video and not images:
        modality = VLMModality.TEXT
    elif modality == VLMModality.MIXED:
        if video is not None:
            modality = VLMModality.VIDEO
        elif len(images) > 1:
            modality = VLMModality.MULTI_IMAGE
        elif images:
            modality = VLMModality.IMAGE
        else:
            modality = VLMModality.TEXT

    return VLMSample(
        dataset=spec.name,
        modality=modality,
        text=text,
        images=images,
        video=video,
        conversations=conversations,
        metadata={"row_id": row.get("id"), "subset": spec.hf_subset},
    )
