"""Schema of a deployment config (validated with pydantic after inheritance is resolved)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeployConfig(BaseModel):
    """Top-level keys of a deployment config.

    Component sections (``adapter``, ``pipeline`` items, ``calibration``) stay loose dicts
    because their keys depend on the registered ``type``; each component validates its own
    arguments in its constructor.

    Attributes:
        name: run name; artifacts go to ``work_dir/name``.
        work_dir: root output directory.
        custom_imports: modules imported before building (mmdet ``custom_imports``), e.g.
            ``["hale_llm.deploy"]`` so a repo's adapters register themselves.
        device: target device profile (vendor, chip, runtime hints) - informational for most
            stages, used by e.g. TensorRT DLA or QNN SoC selection.
        adapter: ``{type: ..., **args}`` of the model adapter.
        calibration: ``{type: ..., **args}`` of the calibration source (quantizing stages).
        pipeline: ordered stage configs ``[{type: ..., **args}, ...]``.
        on_error: ``raise`` aborts at the first failing stage, ``continue`` records the failure
            and runs the remaining stages whose inputs still exist.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "model"
    work_dir: str = "work_dirs/deploy"
    custom_imports: list[str] = Field(default_factory=list)
    device: dict[str, Any] = Field(default_factory=dict)
    adapter: dict[str, Any]
    calibration: dict[str, Any] = Field(default_factory=lambda: {"type": "RandomCalibrationSource"})
    pipeline: list[dict[str, Any]]
    on_error: Literal["raise", "continue"] = "raise"
    seed: int = 0

    @field_validator("adapter", "calibration")
    @classmethod
    def _has_type(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "type" not in value:
            raise ValueError("component config needs a 'type' key")
        return value

    @field_validator("pipeline")
    @classmethod
    def _stages_have_type(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("pipeline must contain at least one stage")
        missing = [i for i, stage in enumerate(value) if "type" not in stage]
        if missing:
            raise ValueError(f"pipeline stages {missing} have no 'type' key")
        return value
