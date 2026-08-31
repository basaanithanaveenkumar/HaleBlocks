"""Token-level losses as a library. Task losses (AR, diffusion) register separately."""

import hale_core.nn.losses.functions  # noqa: F401 — register ce, focal, …

from hale_core.nn.losses.registry import TOKEN_LOSSES, get_token_loss, register_token_loss
from hale_core.nn.losses.token import loss_settings, model_token_nll, token_nll

__all__ = [
    "TOKEN_LOSSES",
    "register_token_loss",
    "get_token_loss",
    "token_nll",
    "model_token_nll",
    "loss_settings",
]
