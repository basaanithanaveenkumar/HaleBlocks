"""Looped (recurrent-depth) transformer stack: one shared block applied ``T`` times.

Layout (``prelude -> T x core -> coda``, Geiping et al., 2025, arXiv:2502.05171)::

    e   = prelude(x)                         distinct layers, run once (optional)
    h_0 = 0 (with input injection) or e
    h_t = core(inject(h_{t-1}, e) + E^t)     the same weights for t = 1 .. T
    out = coda(h_T  or  ACT mixture of h_t)  distinct layers, run once (optional)

``core`` is ``n_layers`` pre-norm transformer layers; the parameter count is that of
``prelude + core + coda`` while the compute depth is ``prelude + T * core + coda``. Supported
variants (all through :class:`LoopSpec`):

* fixed-``T`` looping with a depth embedding (Universal Transformer, arXiv:1807.03819);
* ACT-gated looping - tokens halt at different passes, ponder cost exposed as ``aux_loss``;
* randomised ``T`` during training and truncated backprop through the last ``k`` passes - the
  recurrent-depth recipe that makes the loop count a test-time-compute knob;
* fixed-point early exit at inference (``h_t ~= h_{t-1}``, the DEQ view, arXiv:1909.01377);
* scratchpad tokens - learned working-memory slots carried across passes and dropped from the
  output.
"""

from hale_core.nn.stacks.looped._variants import (
    loop_aux_loss,
    loop_budget,
    loop_stats,
    looped_stacks,
    set_loop_budget,
)
from hale_core.nn.stacks.looped.loop_spec import LoopSpec
from hale_core.nn.stacks.looped.loop_stats import LoopStats
from hale_core.nn.stacks.looped.looped_stack import LoopedStack

__all__ = [
    "LoopSpec",
    "LoopStats",
    "LoopedStack",
    "loop_aux_loss",
    "loop_budget",
    "loop_stats",
    "looped_stacks",
    "set_loop_budget",
]
