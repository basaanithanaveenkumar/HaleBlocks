"""``LoopSpec``: How a :class:`LoopedStack` iterates its shared block."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from hale_core.nn.layers.loop import (
    DepthEmbeddingKind,
    HaltingKind,
    InjectionKind,
    SharingKind,
)
from hale_core.nn.stacks.looped.loop_presets import resolve_loop_preset


@dataclass(frozen=True)
class LoopSpec:
    """How a :class:`LoopedStack` iterates its shared block.

    Attributes:
        n_loops: passes through the shared block (training maximum and inference default).
        prelude_layers: distinct layers before the loop (map tokens into the latent space).
        coda_layers: distinct layers after the loop (map the latent state back to tokens).
        depth_embedding: loop-step coordinate added to each pass's input
            (:class:`LoopDepthEmbedding`).
        input_injection: how the loop input re-enters every pass (:class:`LoopInputInjection`).
        sandwich_norm: add post-norms inside each core layer (keeps ``h_t`` bounded over long
            loops, i.e. keeps ``I + J_f`` well conditioned).
        halting: ``none`` = run ``T`` passes; ``act`` = Adaptive Computation Time per token;
            ``mor`` = Mixture-of-Recursions, a router picks each token's depth
            (:class:`~hale_core.nn.layers.loop.RecursionRouter`).
        act_threshold: ``1 - eps`` of ACT.
        act_init_bias: initial halting logit (positive halts early).
        ponder_weight: ``tau``; ``aux_loss = tau * mean(N + R)`` when ``halting == "act"``.
        min_loops: training only - sample ``T ~ U{min_loops .. n_loops}`` every forward so the
            model works for any loop count (``None`` = always ``n_loops``).
        bptt_loops: training only - backpropagate through the last ``k`` passes; earlier passes
            run without autograd (``None`` = full backprop through time). Memory becomes
            ``O(k)`` instead of ``O(T)``.
        exit_tolerance: inference only - stop once the largest per-token relative change
            ``|h_t - h_{t-1}| / |h_{t-1}|`` drops below this (``None`` = always run ``T``).
        scratch_tokens: learned memory tokens prepended to the sequence for the stack only.
            Every token reads them; under a causal / block mask they read only each other (so no
            future token leaks), with bidirectional attention they read the whole sequence.
        sharing: schedule of the ``n_layers`` core layers over the passes - ``cycle`` (ABAB),
            ``cycle_rev`` (AB BA) or ``sequence`` (AA BB); identical when ``n_layers == 1``.
        lora_rank: relaxed recursion - a rank-``r`` per-pass delta on every core linear layer
            (:class:`~hale_core.nn.layers.loop.LoopLoRALinear`); ``0`` = strict weight tying.
        mor_balance_weight: weight of the router load-balancing loss (``halting == "mor"``).
        preset: name of the :data:`LOOP_PRESETS` recipe the spec was built from (informational;
            :meth:`coerce` expands it under the explicit fields).
    """

    n_loops: int = 4
    prelude_layers: int = 0
    coda_layers: int = 0
    depth_embedding: DepthEmbeddingKind = "sinusoidal"
    input_injection: InjectionKind = "add"
    sandwich_norm: bool = False
    halting: HaltingKind = "none"
    act_threshold: float = 0.99
    act_init_bias: float = 1.0
    ponder_weight: float = 0.01
    min_loops: int | None = None
    bptt_loops: int | None = None
    exit_tolerance: float | None = None
    scratch_tokens: int = 0
    sharing: SharingKind = "cycle"
    lora_rank: int = 0
    mor_balance_weight: float = 0.01
    preset: str | None = None

    def __post_init__(self) -> None:
        if self.n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {self.n_loops}")
        if self.prelude_layers < 0 or self.coda_layers < 0 or self.scratch_tokens < 0:
            raise ValueError("prelude_layers, coda_layers and scratch_tokens must be >= 0")
        if self.min_loops is not None and not 1 <= self.min_loops <= self.n_loops:
            raise ValueError(
                f"min_loops must be in [1, n_loops={self.n_loops}], got {self.min_loops}"
            )
        if self.bptt_loops is not None and self.bptt_loops < 1:
            raise ValueError(f"bptt_loops must be >= 1, got {self.bptt_loops}")
        if self.halting not in ("none", "act", "mor"):
            raise ValueError(f"unknown halting {self.halting!r}")
        if self.sharing not in ("cycle", "cycle_rev", "sequence"):
            raise ValueError(f"unknown sharing {self.sharing!r}")
        if self.lora_rank < 0:
            raise ValueError(f"lora_rank must be >= 0, got {self.lora_rank}")
        if self.mor_balance_weight < 0:
            raise ValueError(f"mor_balance_weight must be >= 0, got {self.mor_balance_weight}")
        if self.halting == "mor" and self.exit_tolerance is not None:
            raise ValueError("halting='mor' fixes each token's depth; exit_tolerance does not apply")
        if self.exit_tolerance is not None and self.exit_tolerance <= 0:
            raise ValueError(f"exit_tolerance must be > 0, got {self.exit_tolerance}")

    @classmethod
    def preset_spec(cls, name: str, **overrides: Any) -> LoopSpec:
        """The :data:`LOOP_PRESETS` recipe ``name`` with ``overrides`` applied."""
        return cls(**resolve_loop_preset({"preset": name, **overrides}))

    @classmethod
    def coerce(cls, value: Any) -> LoopSpec:
        """A spec from ``None``, a mapping, a pydantic section or a spec (``preset`` expanded)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif not isinstance(value, dict):
            value = vars(value)
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown loop options {sorted(unknown)}; expected {sorted(known)}")
        return cls(**resolve_loop_preset(value))

    def to_dict(self) -> dict[str, Any]:
        """The spec as a plain dict (for logs and checkpoints)."""
        return asdict(self)
