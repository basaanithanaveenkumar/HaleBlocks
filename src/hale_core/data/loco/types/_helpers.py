"""Shared helpers of :mod:`hale_core.data.loco.types` (Locomotion policy training dataset types)."""

from __future__ import annotations

DEFAULT_LOCO_MIXTURE: tuple[str, ...] = (
    "humanplus",
    "exbody2",
    "hover_reference_motions",
    "h2o",
    "amp_reference_clips",
    "phc_full_amass",
    "nymeria",
    "berkeley_humanoid",
    "pulse_motion_prior",
    "asap",
)
