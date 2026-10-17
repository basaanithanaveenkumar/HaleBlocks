"""Compose the full run config from section models. No YAML I/O here."""

from pydantic import Field

from hale_core.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)
from hale_core.config.sections.common import StrictModel


class RunConfig(StrictModel):
    """Validated configuration of one LLM training run (the ``"llm"`` config schema).

    Groups the ``model``, ``train``, ``data``, ``sample``, ``eval``, ``viz``, ``logging`` and
    ``experiment`` sections; ``variant`` selects the model/loss/sampler from the registries.
    """
    variant: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    device: str | None = None
