"""``LocoRobotTarget``: Target embodiment for retargeting / evaluation."""

from __future__ import annotations

from enum import StrEnum


class LocoRobotTarget(StrEnum):
    """Target embodiment for retargeting / evaluation."""

    H1 = "h1"
    G1 = "g1"
    SMPL_X = "smpl_x"
    GENERIC_HUMANOID = "generic_humanoid"
