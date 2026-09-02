"""Unit tests for SmolLM2 LLM dataset registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from hale_core.data.kinds import DatasetKind, LLMDomain, TrainingStage
from hale_core.data.llm.loader import (
    SequentialLLMDataLoader,
    build_llm_dataloader,
    build_smollm2_stage_loader,
)
from hale_core.data.llm.parse import default_row_parser
from hale_core.data.llm.types import (
    DEFAULT_SMOLLM2_RL_MIXTURE,
    DEFAULT_SMOLLM2_SFT_MIXTURE,
    SMOLLM2_PRETRAIN_STAGES,
)
from hale_core.registry import get_dataset, list_datasets

PRETRAIN_DATASETS = (
    "fineweb_edu",
    "dclm",
    "openwebmath",
    "infimm_webmath",
    "finemath4_plus",
    "finemath3_plus",
    "infi_webmath4_plus",
    "infi_webmath3_plus",
    "aug_gsm8k",
    "starcoderdata",
    "stack_edu",
    "starcoder2data",
    "starcoder2_jupyter",
    "cosmopedia_v2",
    "dolma_books",
)


@pytest.mark.parametrize("name", PRETRAIN_DATASETS)
def test_pretrain_dataset_registered(name: str):
    spec = get_dataset(name)
    assert spec.name == name
    assert spec.kind == DatasetKind.LLM
    assert spec.stage == TrainingStage.PRETRAIN
    assert spec.hf_repo


@pytest.mark.parametrize("name", DEFAULT_SMOLLM2_SFT_MIXTURE)
def test_sft_dataset_registered(name: str):
    spec = get_dataset(name)
    assert spec.stage == TrainingStage.FINETUNE
    assert spec.domain in (LLMDomain.INSTRUCTION, LLMDomain.MATH, LLMDomain.CODE)


@pytest.mark.parametrize("name", DEFAULT_SMOLLM2_RL_MIXTURE)
def test_rl_dataset_registered(name: str):
    spec = get_dataset(name)
    assert spec.stage == TrainingStage.RL
    assert spec.domain == LLMDomain.PREFERENCE


def test_smollm2_stage_mixtures():
    assert len(SMOLLM2_PRETRAIN_STAGES) == 4
    stage1 = SMOLLM2_PRETRAIN_STAGES[1]
    assert stage1.datasets[0][0] == "fineweb_edu"


def test_list_datasets_filters():
    llm_names = list_datasets(kind="llm")
    vlm_names = list_datasets(kind="vlm")
    assert "fineweb_edu" in llm_names
    assert "llava_onevision" in vlm_names
    assert "fineweb_edu" not in vlm_names
    assert len(list_datasets(stage="pretrain")) == len(PRETRAIN_DATASETS)


def test_fineweb_edu_spec():
    spec = get_dataset("fineweb_edu")
    assert spec.hf_repo == "HuggingFaceFW/fineweb-edu"
    assert spec.domain == LLMDomain.WEB


def test_smoltalk_spec():
    spec = get_dataset("smoltalk")
    assert spec.hf_repo == "HuggingFaceTB/smoltalk"


def test_ultrafeedback_rl_spec():
    spec = get_dataset("ultrafeedback")
    assert spec.stage == TrainingStage.RL
    assert "chosen" in spec.chosen_fields


def test_default_row_parser_pretrain():
    spec = get_dataset("fineweb_edu")
    sample = default_row_parser({"text": "hello world"}, spec, sample_idx=0)
    assert sample.text == "hello world"
    assert sample.stage == TrainingStage.PRETRAIN


def test_default_row_parser_sft():
    spec = get_dataset("metamathqa")
    sample = default_row_parser(
        {"query": "2+2?", "response": "4"},
        spec,
        sample_idx=0,
    )
    assert sample.prompt == "2+2?"
    assert sample.response == "4"


def test_default_row_parser_rl():
    spec = get_dataset("ultrafeedback")
    sample = default_row_parser(
        {"prompt": "hi", "chosen": "good", "rejected": "bad"},
        spec,
        sample_idx=0,
    )
    assert sample.chosen == "good"
    assert sample.rejected == "bad"


def test_sequential_loader_mock(tmp_path: Path, monkeypatch):
    rows = [{"text": "tok1"}, {"text": "tok2"}]

    def _fake_open(self, *, cache_dir: Path):
        yield from rows

    monkeypatch.setattr(
        "hale_core.data.llm.types.LLMDatasetSpec.open_stream",
        _fake_open,
    )
    loader = SequentialLLMDataLoader(
        dataset_names=["fineweb_edu"],
        cache_dir=tmp_path,
        samples_per_dataset=2,
        prefetch_workers=1,
        prefetch_queue=1,
    )
    samples = list(loader)
    assert len(samples) == 2
    assert samples[0].dataset == "fineweb_edu"


def test_build_smollm2_stage_loader():
    loader = build_smollm2_stage_loader(1, samples_per_dataset=1)
    assert "fineweb_edu" in loader.dataset_names


def test_build_llm_dataloader_sft_defaults():
    loader = build_llm_dataloader(stage="finetune", samples_per_dataset=1)
    assert len(loader.dataset_names) == len(DEFAULT_SMOLLM2_SFT_MIXTURE)
