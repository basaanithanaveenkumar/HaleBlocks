"""Build a serialized TensorRT engine from ONNX (TensorRT 10 and 11, data-center, RTX, Jetson).

.. code-block:: text

    ONNX ─OnnxParser─▶ INetworkDefinition ─IBuilderConfig(profiles, precision, DLA)─▶ .engine

Version handling (feature-detected, no version pinning):

* **TensorRT 11** - strongly typed only; precision comes from the ONNX graph (run
  :class:`ModelOptAutoCastStage` / :class:`ModelOptQuantizeStage` first). Precision flags are
  ignored with a warning.
* **TensorRT 10** - ``strongly_typed: true`` uses the ``STRONGLY_TYPED`` network flag (same
  semantics as 11); otherwise the legacy ``FP16`` / ``BF16`` / ``INT8`` / ``FP8`` builder flags
  are applied ("weak typing").
* **TensorRT for RTX** (``runtime_module: tensorrt_rtx``) - same builder API, JIT-specialized
  on the end-user GPU.
* **Jetson** - ``dla_core`` targets the Deep Learning Accelerator with GPU fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import StageError
from hale_core.deploy.registry import STAGES

#: Legacy (weakly typed) builder flags per requested precision, TensorRT <= 10 only.
PRECISION_FLAGS = {
    "fp32": (),
    "fp16": ("FP16",),
    "bf16": ("BF16",),
    "int8": ("INT8", "FP16"),
    "fp8": ("FP8", "FP16"),
}


@STAGES.register_module()
class TensorRTEngineStage(Stage):
    """Compile the latest ONNX artifact into an engine for the build machine's GPU.

    Args:
        precision: requested precision for weakly typed TensorRT 10 builds.
        strongly_typed: ``None`` = auto (on for TensorRT >= 11 or when the ONNX is marked as
            AutoCast/ModelOpt output), ``True``/``False`` to force on TensorRT 10.
        workspace_gb: builder scratch memory limit.
        optimization_level: builder optimization level 0-5 (higher = slower build, faster engine).
        hardware_compatibility: ``none`` or ``ampere_plus`` (one engine for all Ampere+ GPUs).
        version_compatible: engine loadable by later TensorRT runtimes (lean runtime).
        dla_core: Jetson DLA core index (``None`` = GPU only); ``gpu_fallback`` lets
            unsupported layers run on the GPU.
        timing_cache: path of a persistent timing cache (speeds up rebuilds a lot).
        runtime_module: ``tensorrt`` or ``tensorrt_rtx``.
        filename: engine file name.
    """

    requires = ("onnx",)
    provides = ("tensorrt_engine",)

    def __init__(
        self,
        precision: Literal["fp32", "fp16", "bf16", "int8", "fp8"] = "fp16",
        strongly_typed: bool | None = None,
        workspace_gb: float = 4.0,
        optimization_level: int = 3,
        hardware_compatibility: Literal["none", "ampere_plus"] = "none",
        version_compatible: bool = False,
        dla_core: int | None = None,
        gpu_fallback: bool = True,
        timing_cache: str | None = None,
        runtime_module: Literal["tensorrt", "tensorrt_rtx"] = "tensorrt",
        filename: str = "model.engine",
        **base,
    ) -> None:
        super().__init__(**base)
        self.precision = precision
        self.strongly_typed = strongly_typed
        self.workspace_gb = workspace_gb
        self.optimization_level = optimization_level
        self.hardware_compatibility = hardware_compatibility
        self.version_compatible = version_compatible
        self.dla_core = dla_core
        self.gpu_fallback = gpu_fallback
        self.timing_cache = timing_cache
        self.runtime_module = runtime_module
        self.filename = filename
        self.dependency = OptionalDependency(
            runtime_module,
            "install TensorRT on a CUDA machine (pip install tensorrt / tensorrt-rtx)",
        )

    def missing_dependencies(self):
        return [] if self.dependency.available else [self.dependency]

    # ------------------------------------------------------------------ build
    def run(self, ctx: ExportContext) -> list[Artifact]:
        trt = self.dependency.load()
        source = ctx.store.require("onnx")
        major = int(trt.__version__.split(".")[0])
        strongly_typed = self._strongly_typed(major, source.metadata)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(self._network_flags(trt, strongly_typed))
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse_from_file(str(source.path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise StageError(self.name, "ONNX parse failed:\n" + "\n".join(errors))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(self.workspace_gb * (1 << 30))
        )
        config.builder_optimization_level = self.optimization_level
        if not strongly_typed:
            self._apply_precision(trt, config)
        self._apply_compatibility(trt, config)
        self._apply_dla(trt, config)
        self._add_profile(builder, config, network, ctx)
        cache = self._load_timing_cache(config)
        logger.info(
            "building TensorRT {} engine (strongly_typed={}, precision={}, dla={})",
            trt.__version__,
            strongly_typed,
            "graph" if strongly_typed else self.precision,
            self.dla_core,
        )
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise StageError(self.name, "TensorRT returned no engine (see the TensorRT log above)")
        if cache is not None:
            Path(self.timing_cache).write_bytes(memoryview(config.get_timing_cache().serialize()))
        out = ctx.stage_dir(self.name) / self.filename
        out.write_bytes(memoryview(serialized))
        meta = {
            "tensorrt": trt.__version__,
            "strongly_typed": strongly_typed,
            "precision": self.precision,
            "hardware_compatibility": self.hardware_compatibility,
            "dla_core": self.dla_core,
            "profiles": {
                k: [list(s) for s in v] for k, v in ctx.signature.tensorrt_profiles().items()
            },
        }
        (out.parent / "engine.json").write_text(json.dumps(meta, indent=2))
        return [self.artifact(ctx, "tensorrt_engine", out, **meta)]

    # ------------------------------------------------------------------ helpers
    def _strongly_typed(self, major: int, onnx_meta: dict) -> bool:
        if major >= 11:
            if self.strongly_typed is False:
                logger.warning(
                    "TensorRT {} is always strongly typed; ignoring strongly_typed=false", major
                )
            if self.precision not in ("fp32",) and not onnx_meta.get("strongly_typed_ready"):
                logger.warning(
                    "TensorRT 11 takes precision from the graph: add ModelOptAutoCastStage / "
                    "ModelOptQuantizeStage before this stage to get {}",
                    self.precision,
                )
            return True
        if self.strongly_typed is not None:
            return self.strongly_typed
        return bool(onnx_meta.get("strongly_typed_ready"))

    @staticmethod
    def _network_flags(trt, strongly_typed: bool) -> int:
        flags = 0
        flag_enum = trt.NetworkDefinitionCreationFlag
        if strongly_typed and hasattr(flag_enum, "STRONGLY_TYPED"):
            flags |= 1 << int(flag_enum.STRONGLY_TYPED)
        if hasattr(flag_enum, "EXPLICIT_BATCH"):  # deprecated no-op in 10, gone in 11
            flags |= 1 << int(flag_enum.EXPLICIT_BATCH)
        return flags

    def _apply_precision(self, trt, config) -> None:
        for flag in PRECISION_FLAGS[self.precision]:
            if hasattr(trt.BuilderFlag, flag):
                config.set_flag(getattr(trt.BuilderFlag, flag))
            else:
                logger.warning("BuilderFlag.{} not available in TensorRT {}", flag, trt.__version__)

    def _apply_compatibility(self, trt, config) -> None:
        if self.hardware_compatibility == "ampere_plus":
            config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
        if self.version_compatible:
            config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)

    def _apply_dla(self, trt, config) -> None:
        if self.dla_core is None:
            return
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = self.dla_core
        if self.gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    @staticmethod
    def _add_profile(builder, config, network, ctx: ExportContext) -> None:
        """One optimization profile spanning the signature's min / opt / max shapes."""
        profiles = ctx.signature.tensorrt_profiles()
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            if tensor.name not in profiles:
                raise StageError(
                    "TensorRTEngineStage",
                    f"network input {tensor.name!r} missing from the signature",
                )
            profile.set_shape(tensor.name, *profiles[tensor.name])
        config.add_optimization_profile(profile)

    def _load_timing_cache(self, config):
        if not self.timing_cache:
            return None
        path = Path(self.timing_cache)
        cache = config.create_timing_cache(path.read_bytes() if path.exists() else b"")
        config.set_timing_cache(cache, ignore_mismatch=False)
        return cache
