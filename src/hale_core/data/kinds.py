"""Shared dataset kind and stage enums."""

from __future__ import annotations

from enum import StrEnum


class DatasetKind(StrEnum):
    LLM = "llm"
    VLM = "vlm"
    LOCOMOTION = "locomotion"


class TrainingStage(StrEnum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    RL = "rl"


class LLMDomain(StrEnum):
    WEB = "web"
    MATH = "math"
    CODE = "code"
    SYNTHETIC = "synthetic"
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class VLMModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    MULTI_IMAGE = "multi_image"
    VIDEO = "video"
    MIXED = "mixed"


class LocoMotionFormat(StrEnum):
    """On-disk format for motion / trajectory data."""
    NPY = "npy"
    BVH = "bvh"
    PKL = "pkl"
    HDF5 = "hdf5"
    MIXED = "mixed"


class LocoTaskType(StrEnum):
    """Primary task category for the dataset."""
    LOCOMOTION = "locomotion"
    WHOLE_BODY_CONTROL = "whole_body_control"
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"


class LocoRobotTarget(StrEnum):
    """Target embodiment for retargeting / evaluation."""
    H1 = "h1"
    G1 = "g1"
    SMPL_X = "smpl_x"
    GENERIC_HUMANOID = "generic_humanoid"
