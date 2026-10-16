"""AMD stages: MIGraphX (Instinct / Radeon GPUs, ROCm) and Quark (Ryzen AI NPU)."""

from hale_core.deploy.stages.amd.mi_graph_x_compile_stage import MIGraphXCompileStage
from hale_core.deploy.stages.amd.quark_quantize_stage import QuarkQuantizeStage

__all__ = ["MIGraphXCompileStage", "QuarkQuantizeStage"]
