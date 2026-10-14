"""SampleCodec: every supported leaf and container type survives encode -> decode exactly."""

from dataclasses import dataclass

import pytest
import torch

np = pytest.importorskip("numpy")  # optional for HaleBlocks; the codec handles it when present
Image = pytest.importorskip("PIL.Image")

from hale_core.data.webdataset import SampleCodec  # noqa: E402


@dataclass
class Point:
    xyz: torch.Tensor
    label: str


def _assert_same(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, Point):
        _assert_same(a.xyz, b.xyz) and _assert_same(a.label, b.label)
    elif isinstance(a, Image.Image):
        assert a.size == b.size and a.tobytes() == b.tobytes()
    else:
        assert a == b


SAMPLES = [
    torch.arange(6, dtype=torch.int16).view(2, 3),
    {
        "images": torch.rand(2, 3, 4, 5),
        "mask": torch.tensor([True, False]),
        "depth": torch.rand(4).half(),
    },
    {
        "cams": {"K": torch.eye(3), "T": torch.eye(4, dtype=torch.float64)},
        "meta": {"id": 3, "tags": ["a", "b"]},
    },
    (torch.zeros(2), np.arange(5, dtype=np.float32), "caption", b"\x00\x01raw"),
    [1, 2.5, None, True],
    {"point": Point(torch.ones(3), "car"), "score": np.float32(0.25)},
    {"img": Image.new("RGB", (4, 3), (10, 20, 30))},
]


@pytest.mark.parametrize("sample", SAMPLES, ids=[f"sample{i}" for i in range(len(SAMPLES))])
def test_round_trip(sample):
    codec = SampleCodec()
    entries = codec.encode(sample, key="000000007")
    assert entries["__key__"] == "000000007"
    assert all(isinstance(v, bytes) for k, v in entries.items() if k != "__key__")
    decoded = codec.decode(entries)
    expected = (
        {"point": sample["point"], "score": 0.25}
        if isinstance(sample, dict) and "score" in sample
        else sample
    )
    _assert_same(expected, decoded)


def test_tensor_views_and_graphs_are_stored_as_plain_values():
    base = torch.arange(12.0, requires_grad=True)
    view = (base * 2).view(3, 4)[:, 1]  # non-contiguous view with an autograd graph
    out = SampleCodec().decode(SampleCodec().encode({"v": view}, key="k"))
    assert torch.equal(out["v"], view.detach()) and not out["v"].requires_grad


def test_rejects_unsupported_values_and_bad_names():
    codec = SampleCodec()
    with pytest.raises(TypeError):
        codec.encode({"x": object()}, key="k")
    with pytest.raises(ValueError):
        codec.encode({"a.b": torch.zeros(1)}, key="k")
