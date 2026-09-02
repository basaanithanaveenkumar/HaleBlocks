"""Unit tests for VLM dataset registry and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from hale_core.data.kinds import VLMModality
from hale_core.data.vlm.cache import MediaCache
from hale_core.data.vlm.loader import SequentialVLMDataLoader, build_vlm_dataloader
from hale_core.data.vlm.parse import default_row_parser
from hale_core.data.vlm.types import DEFAULT_VLM_MIXTURE
from hale_core.registry import DATASETS, get_dataset, list_datasets


@pytest.mark.parametrize(
    "name",
    DEFAULT_VLM_MIXTURE,
)
def test_vlm_dataset_registered(name: str):
    assert name in DATASETS
    spec = get_dataset(name)
    assert spec.name == name
    assert spec.hf_repo
    assert spec.modality in VLMModality


def test_default_mixture_has_ten_vlm_datasets():
    vlm_names = list_datasets(kind="vlm")
    assert len(DEFAULT_VLM_MIXTURE) == 10
    assert set(DEFAULT_VLM_MIXTURE).issubset(set(vlm_names))


def test_llava_onevision_spec():
    spec = get_dataset("llava_onevision")
    assert spec.hf_repo == "lmms-lab/LLaVA-OneVision-Data"
    assert spec.modality == VLMModality.IMAGE


def test_m4_instruct_is_multi_image():
    spec = get_dataset("m4_instruct")
    assert spec.modality == VLMModality.MULTI_IMAGE


def test_video_datasets_use_video_modality():
    for name in (
        "llava_video_178k",
        "finevideo",
        "videostar",
        "vript",
        "vista_400k",
        "moviechat",
        "sharegpt4video",
    ):
        assert get_dataset(name).modality == VLMModality.VIDEO


def test_default_row_parser_text_only(tmp_path: Path):
    cache = MediaCache(tmp_path, max_bytes=1024**2)
    spec = get_dataset("llava_onevision")
    sample = default_row_parser(
        {"text": "hello", "conversations": [{"from": "human", "value": "hi"}]},
        spec,
        cache=cache,
        sample_idx=0,
    )
    assert sample.text == "hello"
    assert sample.modality == VLMModality.TEXT
    assert sample.conversations is not None


def test_media_cache_eviction(tmp_path: Path):
    cache = MediaCache(tmp_path, max_bytes=30)
    p1 = cache.reserve("ds", 0, ".bin")
    cache.store_bytes(p1, b"a" * 20)
    p2 = cache.reserve("ds", 1, ".bin")
    cache.store_bytes(p2, b"b" * 20)
    assert cache._current_bytes() <= 30


def test_sequential_loader_with_mock_stream(tmp_path: Path, monkeypatch):
    rows = [
        {"text": "one"},
        {"text": "two"},
    ]

    def _fake_open(self, *, cache_dir: Path):
        yield from rows

    monkeypatch.setattr(
        "hale_core.data.vlm.types.VLMDatasetSpec.open_stream",
        _fake_open,
    )

    loader = SequentialVLMDataLoader(
        dataset_names=["llava_onevision"],
        cache_dir=tmp_path,
        samples_per_dataset=2,
        prefetch_workers=1,
        prefetch_queue=1,
    )
    samples = list(loader)
    assert len(samples) == 2
    assert samples[0].text == "one"
    assert samples[1].dataset == "llava_onevision"


def test_build_vlm_dataloader_defaults():
    loader = build_vlm_dataloader(samples_per_dataset=1)
    assert len(loader.dataset_names) == 10
