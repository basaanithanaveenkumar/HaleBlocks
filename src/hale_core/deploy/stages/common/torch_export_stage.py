"""Serialize a ``torch.export`` program (``.pt2``).

The ``.pt2`` is the input of Torch-TensorRT, ExecuTorch and AOTInductor.
"""

from __future__ import annotations

import torch

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES


@STAGES.register_module()
class TorchExportStage(Stage):
    """``torch.export.export`` with the signature's dynamic shapes, then ``torch.export.save``.

    Args:
        strict: use the strict (TorchDynamo) tracer; non-strict handles more Python.
        filename: output file name.
    """

    provides = ("exported_program",)

    def __init__(self, strict: bool = False, filename: str = "model.pt2", **base) -> None:
        super().__init__(**base)
        self.strict = strict
        self.filename = filename

    def run(self, ctx: ExportContext) -> list[Artifact]:
        path = ctx.stage_dir(self.name) / self.filename
        with torch.no_grad():
            program = torch.export.export(
                ctx.export_module,
                ctx.example_args("trace"),
                dynamic_shapes=ctx.signature.torch_dynamic_shapes(),
                strict=self.strict,
            )
        torch.export.save(program, str(path))
        return [self.artifact(ctx, "exported_program", path, strict=self.strict)]
