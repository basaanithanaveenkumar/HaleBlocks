"""Looped (weight-tied, recurrent-depth) transformer: LoopedStack / LoopedBackbone / LoopConfig."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from hale_core.config.sections import LoopConfig, ModelConfig
from hale_core.nn.backbones import LoopedBackbone, build_backbone
from hale_core.nn.layers import ACTHalting
from hale_core.nn.stacks import LoopSpec, loop_aux_loss, loop_budget, loop_stats, set_loop_budget

VOCAB = 40


def looped(
    attn_type: str = "causal", n_layers: int = 1, block_size: int | None = None, **loop
) -> LoopedBackbone:
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
        block_size=block_size,
        loop=loop,
    )


def tokens(batch: int = 2, seq: int = 10) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))


def test_parameters_are_those_of_one_block_whatever_the_loop_count() -> None:
    sizes = {n: sum(p.numel() for p in looped(n_loops=n).parameters()) for n in (1, 4, 12)}
    assert len(set(sizes.values())) == 1
    stack = looped(n_layers=2, n_loops=3, prelude_layers=1, coda_layers=1).stack
    assert len(stack.layers) == 4 and stack.effective_depth == 1 + 3 * 2 + 1
    assert [len(stack.prelude), len(stack.core), len(stack.coda)] == [1, 2, 1]


def test_one_loop_without_extras_is_a_gpt_block() -> None:
    """T=1, no depth embedding, no injection: exactly a 1-layer GPT backbone (same weights)."""
    loop = looped(n_loops=1, depth_embedding="none", input_injection="none")
    gpt = build_backbone(
        VOCAB, arch="transformer", d_model=32, n_heads=4, n_layers=1, d_ff=64, max_length=16
    )
    gpt.load_state_dict(loop.state_dict())
    loop.eval(), gpt.eval()
    torch.testing.assert_close(loop(tokens()), gpt(tokens()))


def test_the_loop_reuses_the_shared_block() -> None:
    """Unrolling the core block by hand reproduces the stack (h_t = core(h_{t-1} + e) + E^t)."""
    bb = looped(n_loops=3).eval()
    ids = tokens()
    stack = bb.stack
    e = bb.embed(ids)
    h = torch.zeros_like(e)
    mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), 1)
    for t in range(3):
        h = stack.core[0](h + e + stack.depth_embedding(t, e), attn_mask=mask)
    torch.testing.assert_close(bb(ids), bb.lm_head(stack.ln_out(h)))


@pytest.mark.parametrize("attn_type", ["causal", "bidirectional", "block_causal"])
@pytest.mark.parametrize(
    "loop",
    [
        {"n_loops": 3},
        {
            "n_loops": 4,
            "halting": "act",
            "prelude_layers": 1,
            "coda_layers": 1,
            "input_injection": "concat",
        },
        {
            "n_loops": 5,
            "min_loops": 2,
            "bptt_loops": 2,
            "depth_embedding": "learned",
            "sandwich_norm": True,
        },
        {"n_loops": 3, "scratch_tokens": 3, "halting": "act"},
    ],
    ids=["fixed", "act", "recurrent_depth", "scratchpad"],
)
def test_every_variant_trains_with_every_attention(attn_type: str, loop: dict) -> None:
    bb = looped(attn_type, block_size=5 if attn_type == "block_causal" else None, **loop)
    ids = tokens(seq=20 if attn_type == "block_causal" else 10)
    padding = torch.ones_like(ids)
    padding[1, -3:] = 0
    out = bb(ids, attention_mask=padding)
    assert out.shape == (*ids.shape, VOCAB) and torch.isfinite(out).all()
    aux = loop_aux_loss(bb)
    assert (aux is not None) == (loop.get("halting") == "act")
    (out.float().pow(2).mean() + (aux if aux is not None else 0)).backward()
    core = bb.stack.core[0]
    assert all(p.grad is not None for p in core.parameters())
    if aux is not None:
        assert bb.stack.halting.proj.bias.grad is not None


def test_causal_loops_do_not_leak_the_future_even_through_scratch_tokens() -> None:
    bb = looped(n_loops=3, scratch_tokens=4, halting="act", prelude_layers=1, coda_layers=1).eval()
    a = tokens(1)
    b = a.clone()
    b[0, 7] = (a[0, 7] + 1) % VOCAB
    torch.testing.assert_close(bb(a)[:, :7], bb(b)[:, :7])
    assert not torch.allclose(bb(a)[:, 7:], bb(b)[:, 7:])


def test_bidirectional_scratch_tokens_read_the_whole_sequence() -> None:
    bb = looped("bidirectional", n_loops=2, scratch_tokens=2).eval()
    grown, kw = bb.stack._prepend_scratch(
        torch.zeros(1, 5, 32),
        {"attn_mask": None, "key_padding_mask": None, "positions": torch.arange(5)},
    )
    assert grown.shape == (1, 7, 32) and kw["attn_mask"] is None
    assert kw["positions"].tolist() == [0, 0, 0, 1, 2, 3, 4]


def test_act_halting_weights_sum_to_one_and_ponder_counts_passes() -> None:
    halting = ACTHalting(4, threshold=0.99, init_bias=0.0)  # p = 0.5 everywhere
    hidden = torch.ones(1, 3, 4)
    active = torch.tensor([[True, True, False]])
    state = halting.init_state(hidden, active)
    for t in range(3):
        state = halting.step(state, hidden * (t + 1), last=t == 2)
    # p=.5 twice (sum 1.0 > .99 at the 2nd pass -> halts with remainder .5): weights .5 h1 + .5 h2
    torch.testing.assert_close(state.output[0, 0], torch.full((4,), 1.5))
    assert state.n_updates[0].tolist() == [2.0, 2.0, 0.0] and state.all_halted
    torch.testing.assert_close(halting.ponder_cost(state, active), torch.tensor(2.5))


def test_act_stops_early_and_reports_stats() -> None:
    bb = looped(n_loops=8, halting="act", act_init_bias=3.0).eval()  # p ~ .95: halt at pass 2
    bb(tokens())
    stats = loop_stats(bb)
    assert stats["loop/loops"] == 2.0 and stats["loop/mean_updates"] == 2.0
    assert float(bb.stack.aux_loss) == pytest.approx(0.01 * stats["loop/ponder_cost"])


def test_test_time_loop_budget_and_fixed_point_exit() -> None:
    bb = looped(n_loops=2).eval()
    with loop_budget(bb, 6):
        bb(tokens())
        assert loop_stats(bb)["loop/loops"] == 6.0
    bb(tokens())
    assert loop_stats(bb)["loop/loops"] == 2.0
    assert (
        set_loop_budget(
            build_backbone(VOCAB, arch="transformer", d_model=32, n_heads=4, n_layers=1), 3
        )
        == 0
    )

    converging = looped(
        n_loops=50, exit_tolerance=0.5, input_injection="none", depth_embedding="none"
    ).eval()
    converging(tokens())
    stats = loop_stats(converging)
    assert stats["loop/loops"] < 50 and stats["loop/residual"] < 0.5
    converging.train()
    converging(tokens())
    assert loop_stats(converging)["loop/loops"] == 50  # never exits early while training

    learned = looped(n_loops=2, depth_embedding="learned").eval()
    with loop_budget(learned, 3), pytest.raises(ValueError, match="sinusoidal"):
        learned(tokens())


def test_training_samples_the_loop_count_and_truncates_backprop() -> None:
    bb = looped(n_loops=6, min_loops=2).train()
    seen = set()
    for _ in range(30):
        bb(tokens())
        seen.add(bb.stack.last_stats.loops)
    assert seen <= set(range(2, 7)) and len(seen) > 2

    bb = looped(n_loops=4, bptt_loops=1, prelude_layers=1).train()
    calls = []
    hook = bb.stack.core[0].register_forward_hook(lambda m, i, o: calls.append(o.requires_grad))
    bb(tokens()).sum().backward()
    hook.remove()
    assert calls == [False, False, False, True]  # only the last pass is on the autograd tape
    assert (
        bb.stack.prelude[0].attn.attn.in_proj_weight.grad is not None
    )  # input injection reaches it


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"n_loops": 0}, "n_loops"),
        ({"min_loops": 9}, "min_loops"),
        ({"bptt_loops": 0}, "bptt_loops"),
        ({"exit_tolerance": -1.0}, "exit_tolerance"),
        ({"halting": "pondernet"}, "halting"),
    ],
)
def test_invalid_loop_specs_are_rejected(bad: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LoopSpec(**bad)
    with pytest.raises(ValidationError):
        LoopConfig(**bad)


def test_yaml_section_reaches_the_backbone() -> None:
    cfg = ModelConfig(
        arch="looped",
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        loop={"n_loops": 5, "halting": "act", "scratch_tokens": 2},
    )
    bb = LoopedBackbone.from_config(VOCAB, cfg)
    assert isinstance(bb, LoopedBackbone) and bb.loop == LoopSpec(
        n_loops=5, halting="act", scratch_tokens=2
    )
    assert set(LoopConfig.model_fields) == set(LoopSpec().to_dict())
    # Checkpoints store the model section as a plain dict (hale_core.nn.llm.hale_backbone).
    as_dict = SimpleNamespace(**cfg.model_dump())
    assert build_backbone(VOCAB, as_dict).loop == bb.loop
    with pytest.raises(ValueError, match="unknown loop options"):
        LoopSpec.coerce({"n_loop": 3})
