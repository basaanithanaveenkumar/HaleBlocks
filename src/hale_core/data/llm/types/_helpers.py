"""Shared helpers of :mod:`hale_core.data.llm.types` (LLM dataset sample skeleton (SmolLM2 paper)).
"""

from __future__ import annotations

DEFAULT_SMOLLM2_SFT_MIXTURE: tuple[str, ...] = (
    "smoltalk",
    "magpie_ultra",
    "smol_constraint",
    "smol_rewrite",
    "smol_summarization",
    "numinamath_cot",
    "metamathqa",
    "self_oss_starcoder2_instruct",
    "apigen_function_calling",
    "systemchats2",
    "longalign",
    "everyday_conversations",
    "explore_instruct",
    "openhermes25",
)


DEFAULT_SMOLLM2_RL_MIXTURE: tuple[str, ...] = (
    "ultrafeedback",
    "ultrainteract",
    "capybara",
    "orca_dpo",
)
