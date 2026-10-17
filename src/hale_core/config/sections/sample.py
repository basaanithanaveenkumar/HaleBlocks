"""``sample:`` section of a run config (generation settings)."""

from hale_core.config.sections.common import StrictModel


class SampleConfig(StrictModel):
    """Sampling settings used for evaluation and visualisation (length, temperature, steps)."""
    max_new_tokens: int = 32
    temperature: float = 1.0
    sampling_steps: int = 32
    eps: float = 1e-3
    prompt: str = "hello"
