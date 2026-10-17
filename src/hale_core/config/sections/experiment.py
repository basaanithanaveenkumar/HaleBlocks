"""``experiment:`` section of a run config (run folders and provenance)."""

from hale_core.config.sections.common import StrictModel


class ExperimentConfig(StrictModel):
    """Where run folders are created and which YAML produced the run."""

    enabled: bool = True
    root: str = "data/experiments"
    name: str | None = None
    source_yaml: str | None = None
