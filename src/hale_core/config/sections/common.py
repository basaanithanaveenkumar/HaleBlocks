"""Shared pydantic base model for config sections (strict: unknown keys are errors)."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Pydantic model that rejects unknown YAML keys."""

    model_config = ConfigDict(extra="forbid")
