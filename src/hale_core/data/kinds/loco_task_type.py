"""``LocoTaskType``: Primary task category for the dataset."""

from __future__ import annotations

from enum import StrEnum


class LocoTaskType(StrEnum):
    """Primary task category for the dataset."""

    LOCOMOTION = "locomotion"
    WHOLE_BODY_CONTROL = "whole_body_control"
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
