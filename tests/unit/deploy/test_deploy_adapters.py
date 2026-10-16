"""Adapters, the export wrapper and the export context."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from hale_core.deploy.adapters.export_module import ExportModule
from hale_core.deploy.adapters.hale_sequence_backbone_adapter import HaleSequenceBackboneAdapter
from hale_core.deploy.adapters.torch_module_adapter import TorchModuleAdapter
from hale_core.deploy.exceptions import ConfigError, DeployError


@dataclass
class Out:
    a: torch.Tensor
    b: torch.Tensor


def test_export_module_maps_every_output_kind() -> None:
    x = torch.ones(2, 3)
    as_dict = ExportModule(
        torch.nn.Identity(), lambda m, i: {"b": i["x"] * 2, "a": i["x"]}, ["x"], ["a", "b"]
    )
    assert torch.equal(as_dict(x)[1], x * 2)
    as_dc = ExportModule(torch.nn.Identity(), lambda m, i: Out(i["x"], i["x"] + 1), ["x"], ["b"])
    assert torch.equal(as_dc(x)[0], x + 1)
    assert ExportModule(torch.nn.Identity(), lambda m, i: i["x"], ["x"], ["y"])(x)[0] is x
    with pytest.raises(DeployError, match="lack"):
        ExportModule(torch.nn.Identity(), lambda m, i: {"a": i["x"]}, ["x"], ["z"])(x)
    with pytest.raises(DeployError, match="expected 1 inputs"):
        as_dict(x, x)


def test_checkpoint_loading_variants(tmp_path, tiny_adapter) -> None:
    model = tiny_adapter.build_model()
    torch.save(
        {"model": {f"module.{k}": v for k, v in model.state_dict().items()}}, tmp_path / "ck.pt"
    )
    tiny_adapter.checkpoint = str(tmp_path / "ck.pt")
    loaded = tiny_adapter.load()
    assert not loaded.training and torch.equal(loaded.fc1.weight, model.fc1.weight)
    tiny_adapter.checkpoint = str(tmp_path / "missing.pt")
    with pytest.raises(DeployError, match="not found"):
        tiny_adapter.load()


def test_torch_module_adapter_factory_and_errors() -> None:
    adapter = TorchModuleAdapter(
        factory="torch.nn:Linear",
        kwargs={"in_features": 4, "out_features": 2},
        positional=True,
        signature={
            "inputs": [{"name": "x", "shape": ["b", 4]}],
            "outputs": ["y"],
            "dims": {"b": [1, 2, 4]},
        },
    )
    model = adapter.load()
    assert adapter.export_module(model)(torch.ones(2, 4))[0].shape == (2, 2)
    with pytest.raises(ConfigError, match="package.module:Name"):
        TorchModuleAdapter(factory="torch.nn.Linear")
    with pytest.raises(ConfigError, match="cannot import"):
        TorchModuleAdapter(factory="torch.nn:Nope").build_model()
    with pytest.raises(ConfigError, match="not an nn.Module"):
        TorchModuleAdapter(factory="builtins:dict", kwargs={"x": 4}).build_model()
    with pytest.raises(ConfigError, match="rejected kwargs"):
        TorchModuleAdapter(factory="torch.nn:Linear", kwargs={"bogus": 1}).build_model()


def test_sequence_backbone_adapter_signature_and_forward() -> None:
    adapter = HaleSequenceBackboneAdapter(
        vocab_size=50,
        max_batch=2,
        opt_seq=8,
        backbone={
            "arch": "transformer",
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 64,
            "max_length": 16,
            "use_time_cond": True,
        },
    )
    sig = adapter.signature()
    assert sig.input_names == ["input_ids", "t"] and sig.dims["seq"].max == 16
    model = adapter.load()
    out = adapter.export_module(model)(*sig.as_args(sig.sample("trace")))
    assert out[0].shape == (2, 8, 50) and sig.outputs == ("logits",)


def test_export_context_caches_reference_outputs(tiny_context) -> None:
    first = tiny_context.reference_outputs("opt")
    assert tiny_context.reference_outputs("opt") is first
    assert tiny_context.example_args()[0].shape == (2, 8)  # trace corner keeps batch symbolic
    assert tiny_context.stage_dir("s").is_dir()
