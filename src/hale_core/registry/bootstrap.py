"""Register built-in plugins (config, loggers, trainers, optimizers, token losses)."""


def register_builtins() -> None:
    import hale_core.config  # noqa: F401
    import hale_core.logging  # noqa: F401
    import hale_core.nn.losses.functions  # noqa: F401
    import hale_core.nn.optim  # noqa: F401
    import hale_core.training.trainer  # noqa: F401


register_builtins()
