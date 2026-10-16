"""Register built-in plugins.

Order matters: registries are created in plugins.py (imported first), then each
module registers itself here as a side effect.

Categories registered here:
  config       — hale_core.config
  loggers      — hale_core.logging
  trainer      — hale_core.training.trainer
  optimizers   — hale_core.nn.optim
  token losses — hale_core.nn.losses.functions
  datasets     — hale_core.data.*.builtins
  norms        — hale_core.nn.layers.norm
  blocks       — hale_core.nn.layers.layer + hale_core.nn.layers.dit
  schedulers   — hale_core.training.schedulers
  collators    — hale_core.data.loaders
  dataloaders  — hale_core.data.loaders
"""

_BOOTSTRAPPED = False


def register_builtins() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    # Training-level
    import hale_core.config  # noqa: F401

    # Datasets
    import hale_core.data.llm.builtins  # noqa: F401
    import hale_core.data.loaders  # noqa: F401
    import hale_core.data.loco.builtins  # noqa: F401
    import hale_core.data.vlm.builtins  # noqa: F401
    import hale_core.logging  # noqa: F401
    import hale_core.nn.layers.dit  # noqa: F401   — registers "dit"
    import hale_core.nn.layers.layer  # noqa: F401  — registers "transformer"

    # NN components (norms + blocks trigger NORMS_REGISTRY / BLOCKS_REGISTRY)
    import hale_core.nn.layers.norm  # noqa: F401  — registers "layer", "rms"
    import hale_core.nn.losses.functions  # noqa: F401
    import hale_core.nn.optim  # noqa: F401

    # Vision backbones — lazy-loaded via registry._ensure_backbones_imported()
    # Do NOT import all backbone modules here at bootstrap time; they are heavy
    # and not needed for LLM/language-only runs.  The vision registry triggers
    # the imports lazily on first call to build_vision_backbone() / list_vision_backbones().
    # This line just ensures the vision package is on sys.path / importable.
    import hale_core.nn.vision  # noqa: F401

    # Schedulers + DataLoader factories
    import hale_core.training.schedulers  # noqa: F401
    import hale_core.training.trainer  # noqa: F401
