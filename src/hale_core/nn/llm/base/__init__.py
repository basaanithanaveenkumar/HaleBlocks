"""The single interface the VLM and VLA use to talk to any language backbone."""

from hale_core.nn.llm.base.language_backbone import LanguageBackbone
from hale_core.nn.llm.base.language_backbone_output import LanguageBackboneOutput

__all__ = ["LanguageBackbone", "LanguageBackboneOutput"]
