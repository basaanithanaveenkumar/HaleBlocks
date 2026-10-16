"""Qualcomm stages: AI Hub (cloud compile/profile on real Snapdragon devices) and ExecuTorch QNN."""

from hale_core.deploy.stages.qualcomm.execu_torch_qnn_stage import ExecuTorchQNNStage
from hale_core.deploy.stages.qualcomm.qai_hub_compile_stage import QAIHubCompileStage

__all__ = ["ExecuTorchQNNStage", "QAIHubCompileStage"]
