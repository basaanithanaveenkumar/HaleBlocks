"""Looped variants: presets, sharing schedules, relaxed recursion, Mixture-of-Recursions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from hale_core.config.sections import LoopConfig
from hale_core.nn.backbones import build_backbone
from hale_core.nn.layers.loop import LoopLoRALinear, RecursionRouter
from hale_core.nn.stacks import LOOP_PRESETS, LoopSpec, loop_aux_loss, loop_stats
from hale_core.nn.stacks.looped import resolve_loop_preset

VOCAB = 40


def looped(n_layers: int = 2, attn_impl: str = "gqa", attn_type: str = "causal", **loop):
    torch.manual_seed(0)
    return build_backbone(
        VOCAB,
        arch="looped",
        d_model=32,
        n_heads=4,
        n_layers=n_layers,
        d_ff=64,
        max_length=16,
        attn_type=attn_type,
        attn_impl=attn_impl,
        loop=loop,
    )


def tokens(batch: int = 2, seq: int = 10) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))


# ---------------------------------------------------------------------------- presets
def test_every_preset_is_documented_and_valid() -> None:
    assert len(LOOP_PRESETS) >= 10
    for name, preset in LOOP_PRESETS.items():
        assert preset.name == name and preset.idea and "arXiv" in preset.reference
        spec = LoopSpec.preset_spec(name)
        assert spec.preset == name
        for key, value in preset.options.items():
            assert getattr(spec, key) == value, (name, key)


@pytest.mark.parametrize("name", sorted(LOOP_PRESETS))
def test_every_preset_trains(name: str) -> None:
    bb = looped(preset=name)
    padding = torch.ones(2, 10, dtype=torch.long)
    padding[1, -3:] = 0
    out = bb(tokens(), attention_mask=padding)
    aux = loop_aux_loss(bb)
    (out.square().mean() + (aux if aux is not None else 0)).backward()
    missing = [n for n, p in bb.named_parameters() if p.grad is None]
    assert not missing, (name, missing)


def test_preset_overrides_and_yaml_round_trip() -> None:
    spec = LoopSpec.coerce({"preset": "recurrent_depth", "n_loops": 16})
    assert spec.n_loops == 16 and spec.prelude_layers == 1 and spec.min_loops == 2
    cfg = LoopConfig(preset="mixture_of_recursions", n_loops=6)
    assert cfg.halting == "mor" and cfg.n_loops == 6 and cfg.prelude_layers == 1
    # A checkpoint stores the resolved dump; re-coercing it gives the same spec.
    assert LoopSpec.coerce(cfg.model_dump()) == LoopSpec.coerce(cfg)
    assert resolve_loop_preset({"n_loops": 3}) == {"n_loops": 3}
    with pytest.raises(ValueError, match="unknown loop preset"):
        LoopSpec.coerce({"preset": "huginn-xl"})
    with pytest.raises(ValidationError):
        LoopConfig(preset="nope")


# ---------------------------------------------------------------------------- sharing schedules
def test_sharing_schedules() -> None:
    def order(sharing: str, n_loops: int = 4) -> str:
        stack = looped(n_layers=2, n_loops=n_loops, sharing=sharing).stack
        names = {id(layer): "AB"[i] for i, layer in enumerate(stack.core)}
        return " ".join(
            "".join(names[id(m)] for m in stack.pass_layers(t, n_loops)) for t in range(n_loops)
        )

    assert order("cycle") == "AB AB AB AB"
    assert order("cycle_rev") == "AB BA AB BA"
    assert order("sequence") == "AA AA BB BB"
    with pytest.raises(ValueError, match="n_loops >= n_layers"):
        looped(n_layers=3, n_loops=2, sharing="sequence")


def test_sharing_changes_the_function_not_the_parameters() -> None:
    cycle, seq = looped(n_loops=4), looped(n_loops=4, sharing="sequence")
    seq.load_state_dict(cycle.state_dict())
    cycle.eval(), seq.eval()
    assert not torch.allclose(cycle(tokens()), seq(tokens()))


# ---------------------------------------------------------------------------- relaxed recursion
def test_relaxed_recursion_starts_exactly_tied_and_specialises_per_pass() -> None:
    tied = looped(n_loops=3).eval()
    relaxed = looped(n_loops=3, lora_rank=2).eval()
    wrapped = [m for m in relaxed.stack.core.modules() if isinstance(m, LoopLoRALinear)]
    assert wrapped and not any(
        isinstance(m, LoopLoRALinear) for m in relaxed.stack.ln_out.modules()
    )
    # Copy the shared weights across (the relaxed model stores them under ``.base``).
    state = {
        k.replace(".base.", "."): v
        for k, v in relaxed.state_dict().items()
        if not k.endswith((".down", ".up"))
    }
    tied.load_state_dict(state)
    torch.testing.assert_close(relaxed(tokens()), tied(tokens()))  # B_t = 0: identical
    with torch.no_grad():
        wrapped[0].up[1].normal_()
    assert not torch.allclose(relaxed(tokens()), tied(tokens()))
    assert torch.equal(wrapped[0].delta_weight(0), torch.zeros_like(wrapped[0].weight))
    extra = sum(p.numel() for n, p in relaxed.named_parameters() if n.endswith(("down", "up")))
    assert extra == sum(3 * 2 * (m.in_features + m.out_features) for m in wrapped)


def test_relaxed_recursion_skips_fused_pytorch_attention() -> None:
    gqa = looped(n_loops=2, lora_rank=2, attn_impl="gqa")
    mha = looped(n_loops=2, lora_rank=2, attn_impl="mha")
    n_gqa = sum(isinstance(m, LoopLoRALinear) for m in gqa.stack.core.modules())
    n_mha = sum(isinstance(m, LoopLoRALinear) for m in mha.stack.core.modules())
    assert n_gqa > n_mha > 0  # mha: only the FFN projections are adapted
    assert all(
        not isinstance(m, LoopLoRALinear)
        for a in mha.stack.core.modules()
        if isinstance(a, nn.MultiheadAttention)
        for m in a.modules()
    )


def test_test_time_loops_beyond_training_reuse_the_last_delta() -> None:
    lin = LoopLoRALinear(nn.Linear(4, 3), n_loops=2, rank=1)
    with torch.no_grad():
        lin.up.normal_()
    x = torch.randn(5, 4)
    lin.pass_index = 1
    last = lin(x)
    lin.pass_index = 7
    torch.testing.assert_close(lin(x), last)
    lin.pass_index = None
    torch.testing.assert_close(lin(x), lin.base(x))


# ---------------------------------------------------------------------------- Mixture-of-Recursions
def test_router_starts_at_full_depth_and_balances() -> None:
    router = RecursionRouter(16, n_loops=4)
    x = torch.randn(2, 6, 16)
    active = torch.ones(2, 6, dtype=torch.bool)
    active[1, -2:] = False
    decision = router(x, active, max_loops=4)
    assert decision.depth[active].eq(4).all() and decision.depth[~active].eq(0).all()
    assert ((decision.gate > 0) & (decision.gate < 1)).all()
    assert decision.balance_loss > 1.0  # everyone on one depth is maximally unbalanced
    assert router(x, active, max_loops=2).depth.max() == 2  # a sampled shorter loop caps the depth


def test_mor_tokens_stop_after_their_depth() -> None:
    bb = looped(n_loops=4, halting="mor").eval()
    router = bb.stack.router
    with torch.no_grad():  # route every token to exactly 2 passes
        router.proj.weight.zero_()
        router.proj.bias.copy_(torch.tensor([0.0, 5.0, 0.0, 0.0]))
    bb(tokens())
    stats = bb.stack.last_stats
    assert stats.loops == 2 and stats.token_updates.eq(2).all() and stats.mean_updates == 2.0
    assert float(bb.stack.aux_loss) == pytest.approx(0.01 * float(stats.router_balance))
    assert "loop/router_balance" in loop_stats(bb)


def test_mor_is_causal() -> None:
    bb = looped(n_loops=4, halting="mor", scratch_tokens=2).eval()
    with torch.no_grad():
        bb.stack.router.proj.weight.normal_(std=2.0)  # tokens get different depths
    a = tokens(1)
    b = a.clone()
    b[0, 7] = (a[0, 7] + 1) % VOCAB
    torch.testing.assert_close(bb(a)[:, :7], bb(b)[:, :7])


def test_mor_rejects_fixed_point_exit() -> None:
    with pytest.raises(ValueError, match="exit_tolerance"):
        LoopSpec(halting="mor", exit_tolerance=1e-3)


def test_record_passes_exposes_the_state_after_every_pass() -> None:
    bb = looped(n_loops=3).eval()
    with bb.stack.record_passes() as states:
        out = bb(tokens())
    assert len(states) == 3 and all(s.shape == (2, 10, 32) for s in states)
    torch.testing.assert_close(out, bb.lm_head(bb.stack.ln_out(states[-1])))
    bb(tokens())
    assert len(states) == 3  # recording stops when the block exits
