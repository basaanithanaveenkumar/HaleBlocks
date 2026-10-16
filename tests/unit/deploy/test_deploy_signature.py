"""DimSpec / TensorSpec / IOSignature: validation and every backend dialect."""

from __future__ import annotations

import pytest
import torch

from hale_core.deploy.core.dim_spec import DimSpec
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError

CFG = {
    "inputs": [
        {"name": "input_ids", "shape": ["batch", "seq"], "dtype": "int64", "value_range": [0, 100]},
        {"name": "t", "shape": ["batch"], "dtype": "float32"},
        {
            "name": "image",
            "shape": ["batch", 3, 16, 16],
            "dtype": "float16",
            "value_range": [-1, 1],
        },
    ],
    "outputs": ["logits", {"name": "aux"}],
    "dims": {"batch": [1, 1, 8], "seq": {"min": 4, "opt": 32, "max": 64}},
}


def test_dim_spec_forms_validation_and_trace_corner() -> None:
    assert DimSpec.from_config("b", 4) == DimSpec("b", 4, 4, 4)
    assert DimSpec.from_config("b", [1, 2, 3]).is_dynamic
    with pytest.raises(ConfigError):
        DimSpec("b", 4, 2, 8)
    with pytest.raises(ConfigError):
        DimSpec.from_config("b", "big")
    assert DimSpec("b", 1, 1, 8).trace == 2  # never trace a dynamic dim at size 1
    assert DimSpec("b", 1, 1, 1).trace == 1 and DimSpec("s", 4, 32, 64).trace == 32
    with pytest.raises(ValueError):
        DimSpec("b", 1, 1, 8).value("huge")


def test_shapes_samples_and_value_ranges() -> None:
    sig = IOSignature.from_config(CFG)
    assert sig.input_names == ["input_ids", "t", "image"] and sig.outputs == ("logits", "aux")
    assert sig.shape(sig.inputs[0], "max") == (8, 64)
    sample = sig.sample("trace", torch.Generator().manual_seed(0))
    assert sample["input_ids"].shape == (2, 32) and sample["input_ids"].dtype == torch.int64
    assert int(sample["input_ids"].max()) < 100
    assert sample["image"].dtype == torch.float16 and float(sample["image"].abs().max()) <= 1.0
    assert sig.as_args(sample)[1] is sample["t"]


def test_backend_dialects() -> None:
    sig = IOSignature.from_config(CFG)
    assert sig.onnx_dynamic_axes() == {
        "input_ids": {0: "batch", 1: "seq"},
        "t": {0: "batch"},
        "image": {0: "batch"},
    }
    profiles = sig.tensorrt_profiles()
    assert profiles["input_ids"] == ((1, 4), (1, 32), (8, 64)) and profiles["image"][2] == (
        8,
        3,
        16,
        16,
    )
    (per_input,) = sig.torch_dynamic_shapes()
    assert per_input[0][0] is per_input[1][0]  # one shared Dim per name
    static = IOSignature.from_config({"inputs": [{"name": "x", "shape": [2, 3]}], "outputs": ["y"]})
    assert (
        static.torch_dynamic_shapes() is None
        and not static.is_dynamic
        and static.onnx_dynamic_axes() == {}
    )


def test_invalid_signatures() -> None:
    with pytest.raises(ConfigError, match="undeclared dim"):
        IOSignature.from_config({"inputs": [{"name": "x", "shape": ["n"]}], "outputs": ["y"]})
    with pytest.raises(ConfigError, match="unique"):
        IOSignature.from_config({"inputs": [{"name": "x", "shape": [1]}], "outputs": ["x"]})
    with pytest.raises(ConfigError, match="dtype"):
        IOSignature.from_config(
            {"inputs": [{"name": "x", "shape": [1], "dtype": "complex"}], "outputs": ["y"]}
        )
    with pytest.raises(ConfigError, match="missing"):
        IOSignature.from_config({"inputs": [{"shape": [1]}], "outputs": ["y"]})
