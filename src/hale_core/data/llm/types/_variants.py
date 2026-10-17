"""Variants and registrations of :mod:`hale_core.data.llm.types` (LLM dataset sample skeleton
(SmolLM2
paper)).
"""

from __future__ import annotations

from hale_core.data.llm.types.stage_mixture import StageMixture

SMOLLM2_PRETRAIN_STAGES: dict[int, StageMixture] = {
    1: StageMixture(
        stage=1,
        description="0-6T tokens: 90% English web (60/40 FineWeb-Edu/DCLM), 10% StarCoderData",
        datasets=(
            ("fineweb_edu", 0.54),
            ("dclm", 0.36),
            ("starcoderdata", 0.10),
        ),
    ),
    2: StageMixture(
        stage=2,
        description="6-8T tokens: 75% web, 20% code, 5% OpenWebMath",
        datasets=(
            ("fineweb_edu", 0.45),
            ("dclm", 0.30),
            ("starcoderdata", 0.20),
            ("openwebmath", 0.05),
        ),
    ),
    3: StageMixture(
        stage=3,
        description="8-10T tokens: 74% web (40/60 FW-Edu/DCLM), 16% Stack-Edu, 10% math",
        datasets=(
            ("fineweb_edu", 0.296),
            ("dclm", 0.444),
            ("stack_edu", 0.16),
            ("openwebmath", 0.05),
            ("infimm_webmath", 0.05),
        ),
    ),
    4: StageMixture(
        stage=4,
        description="10-11T decay: 58% web, 24% Stack-Edu, 14% math, 4% Cosmopedia",
        datasets=(
            ("fineweb_edu", 0.232),
            ("dclm", 0.348),
            ("stack_edu", 0.24),
            ("finemath4_plus", 0.10),
            ("infi_webmath3_plus", 0.03),
            ("openwebmath", 0.008),
            ("aug_gsm8k", 0.002),
            ("cosmopedia_v2", 0.04),
        ),
    ),
}
