"""Optimizer plugins. Import this package so `@register_optimizer` runs."""

from hale_core.nn.optim.adam import AdamOptimizer
from hale_core.nn.optim.adam_w import AdamWOptimizer
from hale_core.nn.optim.base import BaseOptimizer
from hale_core.nn.optim.muon import MuonOptimizer
from hale_core.nn.optim.sgd import SGDOptimizer
from hale_core.registry import OPTIMIZERS, get_optimizer, register_optimizer

register_optimizer("adam")(AdamOptimizer)
register_optimizer("adamw")(AdamWOptimizer)
register_optimizer("sgd")(SGDOptimizer)
register_optimizer("muon")(MuonOptimizer)


def build_optimizer(name: str, params, config: dict):
    return get_optimizer(name)(params, config)


__all__ = [
    "BaseOptimizer",
    "AdamOptimizer",
    "AdamWOptimizer",
    "SGDOptimizer",
    "MuonOptimizer",
    "OPTIMIZERS",
    "register_optimizer",
    "get_optimizer",
    "build_optimizer",
]
