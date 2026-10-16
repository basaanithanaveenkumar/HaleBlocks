"""ATen → MLX bridge: op coverage, dynamic shapes, quantization, liveness (Apple Silicon)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

mx = pytest.importorskip("mlx.core")

from hale_core.deploy.adapters.export_module import ExportModule  # noqa: E402
from hale_core.deploy.exceptions import ConversionError  # noqa: E402
from hale_core.deploy.mlx_bridge import AtenToMLXConverter  # noqa: E402


class Block(nn.Module):
    """Covers embedding, layer/RMS norm, SDPA (causal + GQA), conv, pooling, softmax, slicing."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(40, 32)
        self.ln = nn.LayerNorm(32)
        self.rms = nn.RMSNorm(32)
        self.qkv = nn.Linear(32, 64)
        self.conv = nn.Conv1d(32, 32, 3, padding=1, groups=2)
        self.head = nn.Linear(32, 40)

    def forward(self, ids):
        h = self.ln(self.emb(ids))
        b, t, _ = h.shape
        q, kv = self.qkv(h).split([32, 32], -1)
        q = q.view(b, t, 4, 8).transpose(1, 2)
        k = kv.view(b, t, 4, 8)[:, :, :2].transpose(1, 2)  # 2 kv heads -> grouped-query attention
        att = nn.functional.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
        h = h + att.transpose(1, 2).reshape(b, t, 32)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.rms(nn.functional.silu(h))
        logits = self.head(h)
        return {
            "logits": logits,
            "probs": logits.softmax(-1)[:, -1],
            "pooled": h.mean(1).clamp(-1, 1),
        }


def convert(module, args, dynamic=None, **kw):
    wrapped = ExportModule(
        module.eval(), lambda m, i: m(**i), ["ids"], ["logits", "probs", "pooled"]
    )
    return wrapped, AtenToMLXConverter(**kw).convert(wrapped, args, dynamic)


def test_fp32_parity_across_dynamic_shapes() -> None:
    torch.manual_seed(0)
    batch, seq = torch.export.Dim("batch", min=1, max=4), torch.export.Dim("seq", min=2, max=32)
    wrapped, program = convert(Block(), (torch.randint(0, 40, (2, 9)),), (({0: batch, 1: seq},),))
    for shape in [(2, 9), (1, 3), (4, 32)]:
        ids = torch.randint(0, 40, shape)
        expected = [t.detach().numpy() for t in wrapped(ids)]
        got = [np.array(a) for a in program(mx.array(ids.numpy()))]
        for e, g in zip(expected, got, strict=True):
            np.testing.assert_allclose(g, e, atol=2e-5)
    assert program.op_histogram()["aten.scaled_dot_product_attention.default"] == 1


def test_weight_quantization_and_liveness() -> None:
    torch.manual_seed(0)
    ids = torch.randint(0, 40, (2, 8))
    wrapped, program = convert(Block(), (ids,), quantize={"bits": 8, "group_size": 32})
    assert program.op_histogram().get("hale.quantized_linear") == 2
    assert not any(k == "qkv.weight" for k in program.params)  # dense copy dropped
    got = np.array(program(mx.array(ids.numpy()))[0])
    ref = wrapped(ids)[0].detach().numpy()
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert cos > 0.999
    released = {name for node in program.nodes for name in node.release}
    assert released  # intermediates are freed after their last use


def test_unsupported_ops_are_listed_together() -> None:
    class Odd(nn.Module):
        def forward(self, ids):
            return {"logits": torch.fft.rfft(ids.float()).real, "probs": ids, "pooled": ids}

    with pytest.raises(ConversionError) as info:
        convert(Odd(), (torch.randint(0, 4, (2, 8)),))
    assert any("fft" in op for op in info.value.unsupported)


def test_mlx_convert_stage_writes_function_and_weights(tiny_context) -> None:
    from hale_core.deploy.runtimes.mlx_function_session import MLXFunctionSession
    from hale_core.deploy.stages.apple.mlx_convert_stage import MLXConvertStage

    fn, weights = MLXConvertStage(dtype="float32", export_corners=["min", "max"]).run(tiny_context)
    assert weights.path.exists() and (fn.path.parent / "mlx_manifest.json").exists()
    session = MLXFunctionSession(fn.path, ["x"])
    x = tiny_context.sample_inputs("max")["x"]
    (out,) = session.run({"x": x.numpy()})
    np.testing.assert_allclose(out, tiny_context.export_module(x)[0].detach().numpy(), atol=1e-5)
