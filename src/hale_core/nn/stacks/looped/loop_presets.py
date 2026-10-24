"""Named looped-transformer variants: published recipes as ready-made :class:`LoopSpec` values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class LoopPreset:
    """A named looped-transformer recipe.

    Attributes:
        name: registry key (``loop: {preset: <name>}`` in YAML).
        idea: one sentence: what makes the variant different.
        reference: paper the recipe follows.
        options: :class:`LoopSpec` fields the recipe sets; anything not listed keeps the default,
            and explicit fields next to ``preset`` override these.
        core_layers: suggested ``n_layers`` of the shared core block.
    """

    name: str
    idea: str
    reference: str
    options: Mapping[str, Any] = field(default_factory=dict)
    core_layers: int = 2


_PRESETS: dict[str, LoopPreset] = {}


def register_loop_preset(preset: LoopPreset) -> LoopPreset:
    """Add ``preset`` to :data:`LOOP_PRESETS` (names are unique)."""
    if preset.name in _PRESETS:
        raise ValueError(f"loop preset {preset.name!r} is already registered")
    _PRESETS[preset.name] = preset
    return preset


#: Read-only view of every registered preset, by name.
LOOP_PRESETS: Mapping[str, LoopPreset] = MappingProxyType(_PRESETS)


def resolve_loop_preset(options: Mapping[str, Any]) -> dict[str, Any]:
    """Expand ``options["preset"]`` (if any) under the explicitly given options.

    ``{"preset": "recurrent_depth", "n_loops": 16}`` -> the recurrent-depth recipe with
    ``n_loops = 16``. The preset name is kept so a resolved spec remembers where it came from;
    resolving an already resolved dict is a no-op (its explicit values win).
    """
    out = dict(options)
    name = out.get("preset")
    if name is None:
        return out
    if name not in _PRESETS:
        raise ValueError(f"unknown loop preset {name!r}; available: {sorted(_PRESETS)}")
    return {**_PRESETS[name].options, **out}


for _preset in (
    LoopPreset(
        "universal_transformer",
        "One block applied T times; a depth embedding tells each pass which step it is; the state "
        "is fed back with no input re-injection.",
        "Dehghani et al., Universal Transformers, ICLR 2019, arXiv:1807.03819",
        {"n_loops": 6, "depth_embedding": "sinusoidal", "input_injection": "none"},
    ),
    LoopPreset(
        "universal_transformer_act",
        "Universal Transformer + Adaptive Computation Time: every token halts after its own number "
        "of passes and pays a ponder cost for each one.",
        "Dehghani et al. 2019; Graves, Adaptive Computation Time, arXiv:1603.08983",
        # tau = 0.001: on the toy char corpus tau = 0.01 makes nearly every token stop after 2
        # passes, 0.001 gives depths 2..8 that follow difficulty (measured, notebooks/looped/02).
        {"n_loops": 8, "input_injection": "none", "halting": "act", "act_init_bias": -1.0,
         "ponder_weight": 0.001},
    ),
    LoopPreset(
        "input_injection",
        "The embedded input is added back before every pass, so long loops never forget the "
        "question - the looped-transformer recipe for learning algorithms.",
        "Yang et al., Looped Transformers are Better at Learning Learning Algorithms, ICLR 2024, "
        "arXiv:2311.12424",
        {"n_loops": 6, "depth_embedding": "none", "input_injection": "add"},
    ),
    LoopPreset(
        "recurrent_depth",
        "Prelude -> shared core x T -> coda, trained with a random T and truncated backprop, so "
        "the loop count becomes a test-time compute knob.",
        "Geiping et al., Scaling up Test-Time Compute with Latent Reasoning, 2025, arXiv:2502.05171",
        # No depth embedding: a step coordinate the block never saw in training (t > 8) breaks
        # test-time scaling; without it the measured loss stays flat up to 24 passes.
        {"n_loops": 8, "prelude_layers": 1, "coda_layers": 1, "depth_embedding": "none",
         "input_injection": "concat", "sandwich_norm": True, "min_loops": 2, "bptt_loops": 4},
    ),
    LoopPreset(
        "deq_fixed_point",
        "Iterate the same map until it stops changing (h* = f(h*)); backprop only through the last "
        "pass (a one-step 'phantom' gradient) for O(1) memory. Post-norms keep the map contractive "
        "so the fixed point exists (without them the residual plateaus around 0.7 per pass).",
        "Bai, Kolter & Koltun, Deep Equilibrium Models, NeurIPS 2019, arXiv:1909.01377",
        {"n_loops": 16, "depth_embedding": "none", "input_injection": "concat", "bptt_loops": 1,
         "sandwich_norm": True, "exit_tolerance": 1e-2},
    ),
    LoopPreset(
        "scratchpad",
        "Learned memory slots ride along every pass as extra tokens - explicit working memory "
        "that is dropped from the output.",
        "Memory / register tokens: Burtsev et al., Memory Transformer, 2020, arXiv:2006.11527",
        {"n_loops": 6, "input_injection": "none", "scratch_tokens": 4},
    ),
    LoopPreset(
        "mixture_of_recursions",
        "A router gives every token its own depth: easy tokens leave the loop after one pass, hard "
        "ones keep recursing; first and last layers stay unshared.",
        "Bae et al., Mixture-of-Recursions, 2025, arXiv:2507.10524",
        {"n_loops": 4, "prelude_layers": 1, "coda_layers": 1, "input_injection": "none",
         "halting": "mor", "mor_balance_weight": 0.01},
    ),
    LoopPreset(
        "relaxed_recursive",
        "One shared block plus a small low-rank (LoRA) delta per pass: weight tying becomes a soft "
        "constraint, so passes can specialise for a few extra parameters.",
        "Bae et al., Relaxed Recursive Transformers, ICLR 2025, arXiv:2410.20672",
        {"n_loops": 4, "input_injection": "none", "lora_rank": 4},
    ),
    LoopPreset(
        "sequence_sharing",
        "With n core layers, run layer 1 for the first T/n passes, then layer 2, ... (AA BB) "
        "instead of cycling through them (AB AB).",
        "Takase & Kiyono, Lessons on Parameter Sharing across Layers in Transformers, 2021, "
        "arXiv:2104.06022",
        {"n_loops": 4, "input_injection": "none", "sharing": "sequence"},
    ),
    LoopPreset(
        "cycle_rev_sharing",
        "Cycle through the n core layers, reversing the order on every other pass (AB BA AB BA), "
        "so each layer sees the others' outputs in both orders.",
        "Takase & Kiyono 2021, arXiv:2104.06022 (CYCLE (REV))",
        {"n_loops": 4, "input_injection": "none", "sharing": "cycle_rev"},
    ),
):
    register_loop_preset(_preset)
