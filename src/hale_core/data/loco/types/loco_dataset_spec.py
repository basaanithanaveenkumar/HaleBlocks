"""``LocoDatasetSpec``: Spec for a humanoid locomotion / whole-body-control policy dataset."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hale_core.data.kinds import (
    DatasetKind,
    LocoMotionFormat,
    LocoRobotTarget,
    LocoTaskType,
)


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
