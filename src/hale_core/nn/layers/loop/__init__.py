"""Components of a looped (weight-tied, recurrent-depth) transformer.

A looped stack applies one shared block ``T`` times, ``h_t = f_W(h_{t-1})``. These modules are
what turns that raw weight tying into a trainable model:

* :class:`LoopDepthEmbedding` - the Universal Transformer's *depth* coordinate ``E^t``
  (Dehghani et al., 2019, arXiv:1807.03819), so the shared block can tell pass 1 from pass 5.
* :class:`LoopInputInjection` - re-feeds the embedded input at every pass, ``f_W(h_{t-1} + e)``
  (Yang et al., 2024, arXiv:2311.12424; Geiping et al., 2025, arXiv:2502.05171). Without it the
  input is only visible at ``t = 0`` and long loops forget it.
* :class:`RecursionRouter` - Mixture-of-Recursions: each token is routed to its own number of
  passes (Bae et al., 2025, arXiv:2507.10524).
* :class:`LoopLoRALinear` - relaxed recursion: one shared weight plus a low-rank delta per pass
  (Bae et al., 2024, arXiv:2410.20672).
* :class:`ACTHalting` - Adaptive Computation Time (Graves, 2016, arXiv:1603.08983): a learned
  per-token halting probability; each token's output is the halting-weighted mixture of its
  states and a ponder cost ``N(i) + R(i)`` penalises extra passes.
"""

from hale_core.nn.layers.loop._helpers import (
    DepthEmbeddingKind,
    HaltingKind,
    InjectionKind,
    SharingKind,
)
from hale_core.nn.layers.loop.act_halting import ACTHalting
from hale_core.nn.layers.loop.act_state import ACTState
from hale_core.nn.layers.loop.loop_depth_embedding import LoopDepthEmbedding
from hale_core.nn.layers.loop.loop_input_injection import LoopInputInjection
from hale_core.nn.layers.loop.loop_lora_linear import LoopLoRALinear
from hale_core.nn.layers.loop.recursion_router import RecursionRouter
from hale_core.nn.layers.loop.router_decision import RouterDecision

__all__ = [
    "ACTHalting",
    "ACTState",
    "DepthEmbeddingKind",
    "HaltingKind",
    "InjectionKind",
    "LoopDepthEmbedding",
    "LoopInputInjection",
    "LoopLoRALinear",
    "RecursionRouter",
    "RouterDecision",
    "SharingKind",
]
