"""``LocoMotionFormat``: On-disk format for motion / trajectory data."""

from __future__ import annotations

from enum import StrEnum


class LocoMotionFormat(StrEnum):
    """On-disk format for motion / trajectory data."""

    NPY = "npy"
    BVH = "bvh"
    PKL = "pkl"
    HDF5 = "hdf5"
    MIXED = "mixed"
