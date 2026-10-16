"""HF checkpoint → MLX with the upstream converters (``mlx_lm.convert`` / ``mlx_vlm.convert``).

For LLM / VLM text towers that ``mlx-lm`` / ``mlx-vlm`` already implement natively (Llama,
Qwen, Gemma, Phi, SmolLM, SigLIP-based VLMs ...) this gives the fastest Mac inference path:
fused attention, KV cache, speculative decoding and the ``mlx_lm.server`` OpenAI API.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Literal

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES


@STAGES.register_module()
class MLXHuggingFaceConvertStage(Stage):
    """Call ``<package>.convert(hf_path, mlx_path, quantize, q_bits, q_group_size, dtype)``.

    Keyword arguments are filtered against the installed converter's signature, so the stage
    keeps working across mlx-lm / mlx-vlm releases that add or rename options.

    Args:
        package: ``mlx_lm`` (text models) or ``mlx_vlm`` (vision-language models).
        q_bits / q_group_size: quantization (``q_bits: null`` keeps full precision).
        dtype: dtype of the non-quantized weights.
    """

    requires = ("hf_checkpoint",)
    provides = ("mlx_checkpoint",)

    def __init__(
        self,
        package: Literal["mlx_lm", "mlx_vlm"] = "mlx_lm",
        q_bits: int | None = 4,
        q_group_size: int = 64,
        dtype: str = "float16",
        **base,
    ) -> None:
        super().__init__(**base)
        self.package = package
        self.q_bits = q_bits
        self.q_group_size = q_group_size
        self.dtype = dtype
        self.dependency = OptionalDependency(
            package, f"pip install {package.replace('_', '-')} (Apple Silicon)"
        )

    def missing_dependencies(self):
        return [] if self.dependency.available else [self.dependency]

    def run(self, ctx: ExportContext) -> list[Artifact]:
        convert = importlib.import_module(self.package).convert
        source = ctx.store.require("hf_checkpoint")
        out = ctx.stage_dir(self.name) / "mlx"
        wanted = {
            "hf_path": str(source.path),
            "mlx_path": str(out),
            "quantize": self.q_bits is not None,
            "q_bits": self.q_bits or 4,
            "q_group_size": self.q_group_size,
            "dtype": self.dtype,
        }
        accepted = inspect.signature(convert).parameters
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        dropped = sorted(set(wanted) - set(kwargs))
        if dropped:
            logger.warning("{}.convert does not accept {}; skipped", self.package, dropped)
        convert(**kwargs)
        return [
            self.artifact(
                ctx,
                "mlx_checkpoint",
                out,
                package=self.package,
                q_bits=self.q_bits,
                q_group_size=self.q_group_size,
            )
        ]
