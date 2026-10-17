"""VLM dataset sample skeleton."""

from hale_core.data.kinds import VLMModality  # noqa: F401 - re-exported for existing imports
from hale_core.data.vlm.types._helpers import DEFAULT_VLM_MIXTURE
from hale_core.data.vlm.types.vlm_dataset_spec import VLMDatasetSpec
from hale_core.data.vlm.types.vlm_sample import VLMSample

__all__ = ["DEFAULT_VLM_MIXTURE", "VLMDatasetSpec", "VLMSample"]
