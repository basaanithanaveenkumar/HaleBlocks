"""``LocoSample``: Normalized sample from a locomotion / WBC policy dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hale_core.data.kinds import (
    LocoTaskType,
)


@dataclass
class LocoSample:
    """Normalized sample from a locomotion / WBC policy dataset."""

    dataset: str
    task_type: LocoTaskType
    # Primary motion payload — at least one of these will be set
    joint_angles: Any | None = None  # np.ndarray (T, J) or similar
    root_trajectory: Any | None = None  # np.ndarray (T, 7) pos+quat
    action_sequence: Any | None = None  # np.ndarray (T, A) policy actions
    # Optional extras
    rgb_frames: list[Path] = field(default_factory=list)
    depth_frames: list[Path] = field(default_factory=list)
    annotation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
