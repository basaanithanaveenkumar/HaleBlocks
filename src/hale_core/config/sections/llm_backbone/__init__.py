"""Language-model backbone sub-config shared by the VLM and VLA repos."""

from hale_core.config.sections.llm_backbone.llm_backbone_config import LLMBackboneConfig
from hale_core.config.sections.llm_backbone.lo_ra_settings import LoRASettings

__all__ = ["LLMBackboneConfig", "LoRASettings"]
