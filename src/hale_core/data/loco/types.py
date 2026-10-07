"""Locomotion policy training dataset types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hale_core.data.kinds import (
    DatasetKind,
    LocoMotionFormat,
    LocoRobotTarget,
    LocoTaskType,
)


@dataclass
class LocoSample:
    """Normalized sample from a locomotion / WBC policy dataset."""

    dataset: str
    task_type: LocoTaskType
    # Primary motion payload — at least one of these will be set
    joint_angles: Any | None = None        # np.ndarray (T, J) or similar
    root_trajectory: Any | None = None     # np.ndarray (T, 7) pos+quat
    action_sequence: Any | None = None     # np.ndarray (T, A) policy actions
    # Optional extras
    rgb_frames: list[Path] = field(default_factory=list)
    depth_frames: list[Path] = field(default_factory=list)
    annotation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocoDatasetSpec:
    """Spec for a humanoid locomotion / whole-body-control policy dataset."""

    name: str
    task_type: LocoTaskType
    kind: DatasetKind = DatasetKind.LOCOMOTION
    # Source: HuggingFace repo, project URL, or local directory description
    hf_repo: str | None = None
    project_url: str | None = None
    motion_format: LocoMotionFormat = LocoMotionFormat.MIXED
    robot_target: LocoRobotTarget = LocoRobotTarget.GENERIC_HUMANOID
    # Venue / origin metadata (informational only)
    venue: str = ""
    description: str = ""
    # Field name hints for HF datasets
    joint_angle_fields: tuple[str, ...] = ("joint_angles", "qpos", "joints")
    root_traj_fields: tuple[str, ...] = ("root_pos", "root_quat", "global_root")
    action_fields: tuple[str, ...] = ("action", "actions", "control")
    image_fields: tuple[str, ...] = ("image", "rgb", "obs_image")
    split: str = "train"
    streaming: bool = True

    def open_stream(self, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
        """Open a streaming iterator over this dataset.

        Delegates to the HF loader when `hf_repo` is set; raises for
        project-page-only datasets that require manual download.
        """
        if self.hf_repo is None:
            raise NotImplementedError(
                f"Dataset '{self.name}' has no HuggingFace repo. "
                "Download manually from the project page and pass a local cache_dir."
            )
        from hale_core.data.loco.hf import open_loco_hf_stream

        return open_loco_hf_stream(self, cache_dir=cache_dir)


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
