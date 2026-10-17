"""Shared helpers of :mod:`hale_core.data.vlm.loader` (Sequential VLM dataloader with on-demand
media
caching).
"""

from __future__ import annotations

from hale_core.data.vlm.types import VLMSample


def _collate_vlm_batch(samples: list[VLMSample]) -> dict:
    return {
        "samples": samples,
        "text": [s.text for s in samples],
        "modality": [s.modality.value for s in samples],
        "dataset": [s.dataset for s in samples],
    }
