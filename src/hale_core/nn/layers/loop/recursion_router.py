"""``RecursionRouter``: token-choice Mixture-of-Recursions router."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.layers.loop.router_decision import RouterDecision


class RecursionRouter(nn.Module):
    """Assigns every token its own number of passes through the shared block.

    Mixture-of-Recursions (Bae et al., 2025, arXiv:2507.10524), *token-choice* flavour: a linear
    router reads each token's loop input ``e_i`` once, scores the ``T`` possible depths and the
    token takes the best one. Because the decision only looks at token ``i``, it is causal -
    unlike expert-choice routing, which ranks tokens across the whole sequence.

    * **Gate.** The token's updates are scaled by ``g_i = sigmoid(z_i[d_i])`` (the logit of the
      chosen depth), so the router receives gradient from the task loss.
    * **Balance.** ``T * sum_d f_d * P_d`` (Switch Transformer) - ``f_d`` the share of tokens
      that chose depth ``d``, ``P_d`` the mean router probability of ``d`` - keeps the router
      from sending every token to one depth.
    * **Init.** Logit biases rise linearly with depth and the weights start near zero, so a new
      router sends every token through all ``T`` passes - a plain looped transformer - and
      learns to shorten from there.

    Args:
        d_model: hidden width.
        n_loops: number of depths ``T``.
    """

    def __init__(self, d_model: int, n_loops: int) -> None:
        super().__init__()
        self.n_loops = n_loops
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_loops)
        nn.init.normal_(self.proj.weight, std=0.01)
        with torch.no_grad():
            self.proj.bias.copy_(torch.linspace(0.0, 2.0, n_loops) + 1.0)

    def forward(self, inputs: torch.Tensor, active: torch.Tensor, max_loops: int) -> RouterDecision:
        """Route ``inputs`` ``(B, L, d)``; ``active`` ``(B, L)`` marks real tokens.

        ``max_loops`` caps the depth (e.g. a sampled training loop count below ``T``).
        """
        logits = self.proj(self.norm(inputs)).float()
        probs = logits.softmax(-1)
        choice = logits.argmax(-1)
        depth = (choice + 1).clamp(max=max_loops)
        gate = torch.sigmoid(logits.gather(-1, choice[..., None]).squeeze(-1))
        mask = active.float()
        count = mask.sum().clamp_min(1.0)
        share = (F.one_hot(choice, self.n_loops).float() * mask[..., None]).sum((0, 1)) / count
        mean_prob = (probs * mask[..., None]).sum((0, 1)) / count
        balance = self.n_loops * (share * mean_prob).sum()
        return RouterDecision(depth * active.long(), gate.to(inputs.dtype), balance, probs)
