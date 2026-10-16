"""Enums and paths in the codec, plain-iterable sources, and the mixture data modules' flag."""

from enum import Enum, IntEnum
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("webdataset")

from hale_core.config.sections import WebDatasetConfig  # noqa: E402
from hale_core.data.kinds import VLMModality  # noqa: E402
from hale_core.data.vlm import loader as vlm_loader  # noqa: E402
from hale_core.data.vlm.types import VLMSample  # noqa: E402
from hale_core.data.webdataset import SampleCodec, WebDatasetAdapter  # noqa: E402


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    LOW = 1
    HIGH = 2


def test_codec_round_trips_enums_and_paths() -> None:
    codec = SampleCodec()
    sample = {
        "modality": VLMModality.IMAGE,  # StrEnum: also a str
        "level": Level.HIGH,  # IntEnum: also an int
        "colors": [Color.RED, Color.BLUE],
        "path": Path("/data/a.png"),
        "posix": PurePosixPath("rel/b.mp4"),
        "plain": "image",
    }
    out = codec.decode(codec.encode(sample, key="000000001"))
    assert out == sample
    assert type(out["modality"]) is VLMModality and out["modality"].value == "image"
    assert type(out["level"]) is Level
    assert type(out["path"]) is type(Path("/data/a.png"))
    assert type(out["posix"]) is PurePosixPath
    assert type(out["plain"]) is str


def test_codec_round_trips_vlm_sample_dataclass() -> None:
    codec = SampleCodec()
    sample = VLMSample(
        dataset="toy", modality=VLMModality.MULTI_IMAGE, text="two cats",
        images=[Path("/cache/1.jpg"), Path("/cache/2.jpg")], video=None,
        conversations=[{"from": "human", "value": "hi"}], metadata={"row": 3},
    )
    assert codec.decode(codec.encode(sample, key="000000002")) == sample


def _samples(n: int) -> list[VLMSample]:
    return [
        VLMSample(dataset="toy", modality=VLMModality.IMAGE, text=f"caption {i}",
                  images=[Path(f"/cache/{i}.jpg")])
        for i in range(n)
    ]


def test_prepare_iterable_flag_off_returns_the_iterable() -> None:
    items = _samples(3)
    out = WebDatasetAdapter(WebDatasetConfig()).prepare_iterable(lambda: items, "x", fingerprint="f")
    assert out is items


def test_prepare_iterable_converts_once_then_streams(tmp_path: Path) -> None:
    cfg = WebDatasetConfig(enabled=True, root=str(tmp_path), samples_per_shard=4, shuffle=False)
    calls = []

    def factory():
        calls.append(1)
        return iter(_samples(10))  # a one-shot iterator: no len()

    adapter = WebDatasetAdapter(cfg)
    first = list(adapter.prepare_iterable(factory, "toy", fingerprint="v1", length=10))
    second = list(adapter.prepare_iterable(factory, "toy", fingerprint="v1", length=10))
    assert calls == [1]  # the source is only read when the shards are written
    assert sorted(s.text for s in first) == [f"caption {i}" for i in range(10)]
    assert first == second and all(type(s.modality) is VLMModality for s in first)


def test_prepare_iterable_unbounded_stream_needs_a_length(tmp_path: Path) -> None:
    cfg = WebDatasetConfig(enabled=True, root=str(tmp_path))
    with pytest.raises(ValueError, match="unbounded"):
        WebDatasetAdapter(cfg).prepare_iterable(lambda: iter(_samples(2)), "toy", fingerprint="f")


def test_mixed_vlm_module_streams_shards_with_flag(tmp_path: Path, monkeypatch) -> None:
    """MixedVLMDataModule: same batches with and without the flag; source untouched on reuse."""
    built = []

    def fake_build(names, **kwargs):
        built.append(names)
        return _samples(6)

    # MixedVLMDataModule looks the builder up in its own module (one class per file).
    from hale_core.data.vlm.loader import mixed_vlm_data_module

    monkeypatch.setattr(mixed_vlm_data_module, "build_vlm_dataloader", fake_build)
    plain = vlm_loader.MixedVLMDataModule(dataset_names=["toy"], samples_per_dataset=6, batch_size=3)
    cfg = WebDatasetConfig(enabled=True, root=str(tmp_path), samples_per_shard=2, shuffle=False)
    sharded = vlm_loader.MixedVLMDataModule(
        dataset_names=["toy"], samples_per_dataset=6, batch_size=3, webdataset=cfg
    )
    again = vlm_loader.MixedVLMDataModule(
        dataset_names=["toy"], samples_per_dataset=6, batch_size=3, webdataset=cfg
    )
    assert len(built) == 2  # plain module + first conversion; the third reuses the shards
    texts = lambda module: sorted(t for b in module.train_loader() for t in b["text"])  # noqa: E731
    assert texts(plain) == texts(sharded) == texts(again)
    assert {m for b in sharded.train_loader() for m in b["modality"]} == {"image"}
