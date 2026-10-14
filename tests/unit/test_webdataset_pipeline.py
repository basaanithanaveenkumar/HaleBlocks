"""Dataset -> shards -> stream: identical samples, caching, flags, ranks and DataLoader workers."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

pytest.importorskip("webdataset")

from hale_core.config.sections import DataLoaderConfig, WebDatasetConfig  # noqa: E402
from hale_core.data.loaders import _default_dataloader  # noqa: E402
from hale_core.data.webdataset import (  # noqa: E402
    ShardManifest,
    WebDatasetAdapter,
    WebDatasetConverter,
    maybe_webdataset,
)


class ToyScenes(Dataset):
    """Deterministic dict samples with mixed dtypes, like a real multi-camera dataset."""

    def __init__(self, n: int = 23) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(i)
        return {
            "index": i,
            "images": torch.rand(2, 3, 4, 4, generator=g),
            "labels": torch.randint(0, 3, (4, 4), generator=g),
            "caption": f"scene {i}",
        }


class ToyStream(IterableDataset):
    def __iter__(self):
        for i in range(7):
            yield {"index": i, "x": torch.full((2,), float(i))}


def _cfg(tmp_path: Path, **kw) -> WebDatasetConfig:
    base = dict(enabled=True, root=str(tmp_path / "shards"), samples_per_shard=5, shuffle=False)
    base.update(kw)
    return WebDatasetConfig(**base)


def _by_index(samples) -> dict[int, dict]:
    return {int(s["index"]): s for s in samples}


def test_disabled_flag_returns_the_original_dataset(tmp_path):
    ds = ToyScenes()
    assert maybe_webdataset(ds, WebDatasetConfig(enabled=False, root=str(tmp_path)), "train") is ds
    assert not (tmp_path / "train").exists()


def test_samples_round_trip_exactly(tmp_path):
    ds = ToyScenes()
    stream = WebDatasetAdapter(_cfg(tmp_path)).prepare(ds, "train")
    assert len(stream) == len(ds)
    got = _by_index(stream)
    assert sorted(got) == list(range(len(ds)))
    for i, sample in got.items():
        ref = ds[i]
        assert (
            torch.equal(sample["images"], ref["images"])
            and sample["images"].dtype == ref["images"].dtype
        )
        assert torch.equal(sample["labels"], ref["labels"])
        assert sample["caption"] == ref["caption"]
    manifest = ShardManifest.load(tmp_path / "shards" / "train")
    assert manifest.num_samples == 23 and len(manifest.shards) == 5  # ceil(23 / 5)


def test_shards_are_reused_and_invalidated(tmp_path):
    cfg = _cfg(tmp_path)
    conv = WebDatasetConverter(cfg)
    conv.convert(ToyScenes(), "train")
    shard = tmp_path / "shards" / "train" / "shard-000000.tar"
    stamp = shard.stat().st_mtime_ns
    conv.convert(ToyScenes(), "train")  # same source -> reused
    assert shard.stat().st_mtime_ns == stamp
    conv.convert(ToyScenes(n=11), "train")  # different length -> rewritten
    assert ShardManifest.load(tmp_path / "shards" / "train").num_samples == 11
    conv.convert(ToyScenes(n=11), "train", fingerprint="v2")  # explicit fingerprint -> rewritten
    assert ShardManifest.load(tmp_path / "shards" / "train").fingerprint == "v2"
    assert not list((tmp_path / "shards").glob(".train.tmp-*"))  # no temp dirs left behind


def test_parallel_conversion_and_max_samples(tmp_path):
    stream = WebDatasetAdapter(_cfg(tmp_path, convert_workers=2, max_samples=9)).prepare(
        ToyScenes(), "train"
    )
    assert sorted(_by_index(stream)) == list(range(9))


def test_iterable_dataset_is_converted(tmp_path):
    stream = WebDatasetAdapter(_cfg(tmp_path, convert_workers=2)).prepare(ToyStream(), "stream")
    assert sorted(_by_index(stream)) == list(range(7))


def test_dataloader_workers_see_every_sample_once(tmp_path):
    stream = WebDatasetAdapter(_cfg(tmp_path, shuffle=True)).prepare(
        ToyScenes(), "train", loader_workers=2
    )
    loader = _default_dataloader(
        stream, DataLoaderConfig(num_workers=2, shuffle=True), batch_size=4
    )
    indices = torch.cat([b["index"] for b in loader]).tolist()
    assert sorted(indices) == list(range(23))
    assert len(loader) == 6  # ceil(23 / 4), from the stream's __len__


def test_multi_rank_resampling_gives_equal_epochs(tmp_path, monkeypatch):
    WebDatasetConverter(_cfg(tmp_path)).convert(ToyScenes(), "train")
    monkeypatch.setenv("WORLD_SIZE", "2")
    for rank in ("0", "1"):
        monkeypatch.setenv("RANK", rank)
        stream = WebDatasetAdapter(_cfg(tmp_path, shuffle=True)).prepare(ToyScenes(), "train")
        assert len(stream) == 11  # 23 // 2 on every rank
        assert len(list(stream)) == 11  # resampled -> exactly one epoch
    eval_stream = WebDatasetAdapter(_cfg(tmp_path)).prepare(ToyScenes(), "train", training=False)
    assert len(eval_stream) == 11


def test_default_loader_splits_map_style_data_across_ranks(monkeypatch):
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a: 2)
    seen = []
    for rank in (0, 1):
        monkeypatch.setattr(dist, "get_rank", lambda *a, r=rank: r)
        loader = _default_dataloader(ToyScenes(n=10), DataLoaderConfig(shuffle=True), batch_size=5)
        assert isinstance(loader.sampler, DistributedSampler)
        seen.append(torch.cat([b["index"] for b in loader]).tolist())
    assert sorted(seen[0] + seen[1]) == list(range(10)) and not set(seen[0]) & set(seen[1])


def test_default_loader_accepts_streams_with_shuffle_flag():
    loader = _default_dataloader(ToyStream(), DataLoaderConfig(shuffle=True), batch_size=2)
    assert len(list(loader)) == 4


def test_dataset_factory_is_only_called_when_shards_must_be_written(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return ToyScenes()

    adapter = WebDatasetAdapter(_cfg(tmp_path))
    assert sorted(_by_index(adapter.prepare(build, "train", fingerprint="v1"))) == list(range(23))
    assert sorted(_by_index(adapter.prepare(build, "train", fingerprint="v1"))) == list(range(23))
    assert len(calls) == 1  # second run reused the shards
    adapter.prepare(build, "train", fingerprint="v2")  # stale -> rebuilt
    assert len(calls) == 2
    with pytest.raises(ValueError):
        adapter.prepare(build, "other")  # factory without fingerprint
    off = WebDatasetAdapter(WebDatasetConfig(enabled=False))
    assert isinstance(off.prepare(build, "x"), ToyScenes)  # flag off: plain dataset
