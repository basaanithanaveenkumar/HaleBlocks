"""Built-in humanoid locomotion policy training dataset registrations.

Category 08 — Humanoid Locomotion Policy Training (10 datasets).
"""

from __future__ import annotations

from hale_core.data.kinds import LocoMotionFormat, LocoRobotTarget, LocoTaskType
from hale_core.data.loco.types import LocoDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("humanplus")
def _humanplus() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="humanplus",
        task_type=LocoTaskType.WHOLE_BODY_CONTROL,
        hf_repo="humanoid-manipulation/HumanPlus",
        project_url="https://humanoid-manipulation.github.io",
        motion_format=LocoMotionFormat.PKL,
        robot_target=LocoRobotTarget.H1,
        venue="CoRL 2024",
        description=(
            "40h of whole-body teleoperation demos on Unitree H1. "
            "SMPL-X shadow retargeting; zero-shot real-robot transfer."
        ),
    )


@register_dataset("exbody2")
def _exbody2() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="exbody2",
        task_type=LocoTaskType.WHOLE_BODY_CONTROL,
        hf_repo="LeCAR-Lab/ExBody2",
        project_url="https://exbody2.github.io",
        motion_format=LocoMotionFormat.NPY,
        robot_target=LocoRobotTarget.H1,
        venue="Dec 2024",
        description=(
            "Cascaded IK solver separating leg vs. upper-body retargeting. "
            "H1 and G1 Isaac Lab NPY motion files included."
        ),
        joint_angle_fields=("qpos", "joint_angles", "joints"),
    )


@register_dataset("hover_reference_motions")
def _hover_reference_motions() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="hover_reference_motions",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo="LeCAR-Lab/HOVER",
        project_url="https://hover-versatile-humanoid.github.io",
        motion_format=LocoMotionFormat.NPY,
        robot_target=LocoRobotTarget.GENERIC_HUMANOID,
        venue="NeurIPS 2024",
        description=(
            "Reference motions for 6 locomotion modes: walk, run, jump, dance, crawl. "
            "Mode-conditioned single policy trained on all clips."
        ),
    )


@register_dataset("h2o")
def _h2o() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="h2o",
        task_type=LocoTaskType.WHOLE_BODY_CONTROL,
        hf_repo="LeCAR-Lab/H2O",
        project_url="https://lecar-lab.github.io/h2o",
        motion_format=LocoMotionFormat.NPY,
        robot_target=LocoRobotTarget.H1,
        venue="IROS 2024",
        description=(
            "Real-time human-to-robot (H2R) retargeting at 30 Hz, <50 ms latency. "
            "Input: commodity RGB camera only — no depth sensor required."
        ),
        image_fields=("rgb", "image", "obs_image"),
    )


@register_dataset("amp_reference_clips")
def _amp_reference_clips() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="amp_reference_clips",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo=None,
        project_url="https://xbpeng.github.io/projects/AMP/index.html",
        motion_format=LocoMotionFormat.BVH,
        robot_target=LocoRobotTarget.SMPL_X,
        venue="SIGGRAPH 2021",
        description=(
            "~100 CMU MoCap clips used as AMP discriminator reference motions. "
            "Foundational baseline for all physics-based locomotion RL. "
            "Download from the CMU MoCap database (http://mocap.cs.cmu.edu)."
        ),
    )


@register_dataset("phc_full_amass")
def _phc_full_amass() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="phc_full_amass",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo="ZhengyiLuo/PHC",
        project_url="https://zhengyiluo.github.io/PHC",
        motion_format=LocoMotionFormat.PKL,
        robot_target=LocoRobotTarget.SMPL_X,
        venue="ICCV 2023",
        description=(
            "All 11K AMASS sequences retargeted to SMPL-X humanoid. "
            "Curriculum-ordered for rare motions (gymnastics, martial arts). "
            "Requires AMASS license — download from https://amass.is.tue.mpg.de."
        ),
    )


@register_dataset("nymeria")
def _nymeria() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="nymeria",
        task_type=LocoTaskType.NAVIGATION,
        hf_repo="facebook/nymeria",
        project_url="https://www.projectaria.com/datasets/nymeria",
        motion_format=LocoMotionFormat.HDF5,
        robot_target=LocoRobotTarget.SMPL_X,
        venue="ECCV 2024",
        description=(
            "300h of egocentric motion capture using Meta Aria glasses. "
            "6-DoF global root trajectory — unique resource for navigation-locomotion."
        ),
        root_traj_fields=("global_root_pos", "global_root_rot", "root_pos"),
        image_fields=("aria_rgb", "rgb", "image"),
    )


@register_dataset("berkeley_humanoid")
def _berkeley_humanoid() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="berkeley_humanoid",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo="HumanoidX/berkeley-humanoid-loco",
        project_url="https://berkeley-humanoid.com",
        motion_format=LocoMotionFormat.NPY,
        robot_target=LocoRobotTarget.GENERIC_HUMANOID,
        venue="2024",
        description=(
            "Real-robot rough-terrain locomotion logs at 50 Hz. "
            "Sim and real paired sequences; low-cost platform proof-of-concept."
        ),
        joint_angle_fields=("qpos", "joint_angles", "obs"),
        action_fields=("action", "actions", "cmd"),
    )


@register_dataset("pulse_motion_prior")
def _pulse_motion_prior() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="pulse_motion_prior",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo="ZhengyiLuo/PULSE",
        project_url="https://zhengyiluo.github.io/PULSE",
        motion_format=LocoMotionFormat.PKL,
        robot_target=LocoRobotTarget.SMPL_X,
        venue="SIGGRAPH 2023",
        description=(
            "AMASS-trained motion VAE; latent space used as the action space "
            "for downstream RL — significantly stabilizes policy learning on rare motions."
        ),
    )


@register_dataset("asap")
def _asap() -> LocoDatasetSpec:
    return LocoDatasetSpec(
        name="asap",
        task_type=LocoTaskType.LOCOMOTION,
        hf_repo="LeCAR-Lab/ASAP",
        project_url="https://agile.human2humanoid.com",
        motion_format=LocoMotionFormat.NPY,
        robot_target=LocoRobotTarget.H1,
        venue="2024",
        description=(
            "Paired sim+real delta trajectories for residual adaptation. "
            "Trains adaptation networks that correct sim-to-real gap without full retraining."
        ),
        joint_angle_fields=("qpos_sim", "qpos_real", "joint_angles"),
        action_fields=("action_delta", "residual_action", "action"),
    )
