"""Model adapters: how a repo's model is built and called for export."""

from hale_core.deploy.adapters.export_module import ExportModule
from hale_core.deploy.adapters.hale_sequence_backbone_adapter import HaleSequenceBackboneAdapter
from hale_core.deploy.adapters.hale_vision_backbone_adapter import HaleVisionBackboneAdapter
from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.adapters.torch_module_adapter import TorchModuleAdapter

__all__ = [
    "ExportModule",
    "HaleSequenceBackboneAdapter",
    "HaleVisionBackboneAdapter",
    "ModelAdapter",
    "TorchModuleAdapter",
]
