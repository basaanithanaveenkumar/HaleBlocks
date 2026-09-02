"""Register all SmolLM2 LLM datasets step by step."""

from hale_core.data.llm.builtins import (
    pretrain_code,  # noqa: F401
    pretrain_math,  # noqa: F401
    pretrain_synthetic,  # noqa: F401
    pretrain_web,  # noqa: F401
    rl,  # noqa: F401
    sft,  # noqa: F401
)

__all__ = [
    "pretrain_web",
    "pretrain_math",
    "pretrain_code",
    "pretrain_synthetic",
    "sft",
    "rl",
]
