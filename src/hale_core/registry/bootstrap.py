"""Register built-in plugins (config, loggers, trainers, optimizers, token losses)."""

_BOOTSTRAPPED = False


def register_builtins() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    import hale_core.config  # noqa: F401
    import hale_core.data.llm.builtins  # noqa: F401
    import hale_core.data.vlm.builtins  # noqa: F401
    import hale_core.logging  # noqa: F401
    import hale_core.nn.losses.functions  # noqa: F401
    import hale_core.nn.optim  # noqa: F401
    import hale_core.training.trainer  # noqa: F401
