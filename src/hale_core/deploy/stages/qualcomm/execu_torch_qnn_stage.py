"""Lower to an ExecuTorch ``.pte`` delegated to the Qualcomm HTP (QNN backend) for Android.

Runs locally (no cloud) with the QAIRT/QNN SDK: ``torch.export`` → (optional PT2E
quantization with ``QnnQuantizer``) → ``to_edge_transform_and_lower_to_qnn`` → ``.pte``. The
app loads it with the ExecuTorch runtime built with the QNN backend.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import MissingDependencyError, StageError
from hale_core.deploy.registry import STAGES

EXECUTORCH_QNN = OptionalDependency(
    "executorch.backends.qualcomm",
    "build ExecuTorch with the QNN backend and set QNN_SDK_ROOT "
    "(docs.pytorch.org/executorch backends-qualcomm)",
)

#: Quantization scheme names → ``QuantDtype`` members (activation bits / weight bits).
QUANT_DTYPES = {
    "8a8w": "use_8a8w",
    "16a8w": "use_16a8w",
    "16a16w": "use_16a16w",
    "16a4w": "use_16a4w",
}


@STAGES.register_module()
class ExecuTorchQNNStage(Stage):
    """Args:
    soc: ``QcomChipset`` member, e.g. ``SM8650`` (8 Gen 3), ``SM8750`` (8 Elite), ``SM8550``.
    quantization: ``none`` (fp16 on HTP) or ``8a8w`` / ``16a8w`` / ``16a16w`` / ``16a4w``.
    filename: output ``.pte`` name.
    """

    provides = ("executorch_pte",)
    dependencies = (EXECUTORCH_QNN,)

    def __init__(
        self,
        soc: str = "SM8650",
        quantization: Literal["none", "8a8w", "16a8w", "16a16w", "16a4w"] = "none",
        filename: str = "model_qnn.pte",
        **base,
    ) -> None:
        super().__init__(**base)
        self.soc = soc
        self.quantization = quantization
        self.filename = filename

    def run(self, ctx: ExportContext) -> list[Artifact]:
        if not os.environ.get("QNN_SDK_ROOT"):
            raise MissingDependencyError(
                "QNN_SDK_ROOT", "export QNN_SDK_ROOT=/path/to/qairt/<version>"
            )
        from executorch.backends.qualcomm.utils.utils import (
            QcomChipset,
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            to_edge_transform_and_lower_to_qnn,
        )

        args = ctx.example_args(
            "opt"
        )  # the HTP compiles static shapes: trace at the deployment shape
        module = ctx.export_module
        if self.quantization != "none":
            module = self._quantize(ctx, module, args)
        try:
            chipset = getattr(QcomChipset, self.soc)
        except AttributeError as exc:
            raise StageError(
                self.name,
                f"unknown SoC {self.soc!r}; QcomChipset has "
                f"{[m for m in dir(QcomChipset) if m.startswith('SM')]}",
            ) from exc
        spec = generate_qnn_executorch_compiler_spec(
            soc_model=chipset,
            backend_options=generate_htp_compiler_spec(use_fp16=self.quantization == "none"),
        )
        program = to_edge_transform_and_lower_to_qnn(module, args, spec).to_executorch()
        out = ctx.stage_dir(self.name) / self.filename
        out.write_bytes(program.buffer)
        logger.info("ExecuTorch QNN program {} ({:.1f} MB)", out, out.stat().st_size / 1e6)
        return [
            self.artifact(ctx, "executorch_pte", out, soc=self.soc, quantization=self.quantization)
        ]

    def _quantize(self, ctx: ExportContext, module, args):
        """PT2E post-training quantization calibrated on the context's batches."""
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(getattr(QuantDtype, QUANT_DTYPES[self.quantization]))
        graph = torch.export.export(module, args, strict=False).module()
        prepared = prepare_pt2e(graph, quantizer)
        with torch.no_grad():
            for batch in ctx.calibration.batches(ctx.signature):
                prepared(*ctx.signature.as_args(batch))
        return convert_pt2e(prepared)
