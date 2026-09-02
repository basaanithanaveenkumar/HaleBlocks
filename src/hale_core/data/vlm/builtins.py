"""Built-in VLM dataset registrations (SmolVLM2 / LLaVA-OneVision mixture)."""

from __future__ import annotations

from hale_core.data.vlm.types import VLMDatasetSpec, VLMModality
from hale_core.registry import register_dataset


@register_dataset("llava_onevision")
def _llava_onevision() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="llava_onevision",
        hf_repo="lmms-lab/LLaVA-OneVision-Data",
        modality=VLMModality.IMAGE,
        split="train",
        text_fields=("text", "caption", "question", "answer"),
        image_fields=("image", "images", "image_path"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("m4_instruct")
def _m4_instruct() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="m4_instruct",
        hf_repo="lmms-lab/M4-Instruct-Data",
        modality=VLMModality.MULTI_IMAGE,
        split="train",
        text_fields=("text", "caption", "question", "answer"),
        image_fields=("images", "image", "image_paths"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("mammoth")
def _mammoth() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="mammoth",
        hf_repo="MAmmoTH-VL/MAmmoTH-VL-Instruct-12M",
        modality=VLMModality.MIXED,
        split="train",
        text_fields=("text", "caption", "question", "answer", "response"),
        image_fields=("image", "images", "image_path"),
        video_fields=("video", "video_path"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("llava_video_178k")
def _llava_video_178k() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="llava_video_178k",
        hf_repo="lmms-lab/LLaVA-Video-178K",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "caption", "question", "answer"),
        video_fields=("video", "video_path", "video_file"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("finevideo")
def _finevideo() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="finevideo",
        hf_repo="HuggingFaceFV/finevideo",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "caption", "description", "title"),
        video_fields=("video", "video_path", "mp4"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("videostar")
def _videostar() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="videostar",
        hf_repo="orrzohar/Video-STaR",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "question", "answer", "caption"),
        video_fields=("video", "video_path"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("vript")
def _vript() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="vript",
        hf_repo="Mutonix/Vript",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "caption", "description"),
        video_fields=("video", "video_path"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("vista_400k")
def _vista_400k() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="vista_400k",
        hf_repo="TIGER-Lab/VISTA-400K",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "question", "answer", "caption"),
        video_fields=("video", "video_path"),
        conversation_fields=("conversations", "messages"),
    )


@register_dataset("moviechat")
def _moviechat() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="moviechat",
        hf_repo="Enxin/MovieChat-1K_train",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "question", "answer", "caption"),
        video_fields=("video", "video_path"),
        conversation_fields=("conversations", "messages", "dialogue"),
    )


@register_dataset("sharegpt4video")
def _sharegpt4video() -> VLMDatasetSpec:
    return VLMDatasetSpec(
        name="sharegpt4video",
        hf_repo="ShareGPT4Video/ShareGPT4Video",
        modality=VLMModality.VIDEO,
        split="train",
        text_fields=("text", "caption", "title"),
        video_fields=("video", "video_path", "video_file"),
        conversation_fields=("conversations", "messages"),
    )
