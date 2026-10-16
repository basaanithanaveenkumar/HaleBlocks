"""AMD Quark post-training quantization of ONNX for the Ryzen AI NPU (Vitis AI EP).

The Ryzen AI NPU runs INT8 / BF16 ONNX models through ONNX Runtime's
``VitisAIExecutionProvider``. Quark produces those models; AMD recommends ``A16W8`` (16-bit
activations, 8-bit weights) for transformers and ``A8W8`` / ``XINT8`` for CNNs.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from hale_core.deploy.calibration.onnx_calibration_reader import OnnxCalibrationReader
from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

QUARK = OptionalDependency("quark", "pip install amd-quark (see ryzenai.docs.amd.com)")
ORT = OptionalDependency("onnxruntime", "pip install onnxruntime")


@STAGES.register_module()
class QuarkQuantizeStage(Stage):
    """Quantize the latest ONNX with a Quark preset and the context's calibration batches.

    Supports both Quark ONNX APIs (feature-detected):

    * 0.10+: ``QConfig.get_default_config(preset)`` + ``ModelQuantizer(config)``
    * older: ``Config(global_quant_config=get_default_config(preset))``

    Args:
        preset: Quark preset name (``A16W8``, ``A8W8``, ``XINT8``, ``BF16``, ``U16S8_AAWS`` ...).
        overrides: attributes set on the global quant config after the preset is loaded.
    """

    requires = ("onnx",)
    provides = ("onnx",)
    dependencies = (QUARK, ORT)

    def __init__(
        self, preset: str = "A16W8", overrides: dict[str, Any] | None = None, **base
    ) -> None:
        super().__init__(**base)
        self.preset = preset
        self.overrides = dict(overrides or {})

    def run(self, ctx: ExportContext) -> list[Artifact]:
        from quark.onnx import ModelQuantizer

        source = ctx.store.require("onnx")
        out = ctx.stage_dir(self.name) / f"model.{self.preset.lower()}.onnx"
        config = self._config()
        OnnxCalibrationReader.register_with_onnxruntime()
        reader = OnnxCalibrationReader(ctx.calibration, ctx.signature)
        logger.info("Quark {} quantization of {}", self.preset, source.path.name)
        ModelQuantizer(config).quantize_model(str(source.path), str(out), reader)
        return [
            self.artifact(
                ctx,
                "onnx",
                out,
                quantization=self.preset,
                target="ryzen_ai_npu",
                **{k: v for k, v in source.metadata.items() if k in ("opset", "inputs", "outputs")},
            )
        ]

    def _config(self):
        try:
            from quark.onnx.quantization.config.config import QConfig

            config = QConfig.get_default_config(self.preset)
            target = getattr(config, "global_config", config)
        except ImportError:
            from quark.onnx.quantization.config import Config, get_default_config

            target = get_default_config(self.preset)
            config = Config(global_quant_config=target)
        for key, value in self.overrides.items():
            setattr(target, key, value)
        return config
