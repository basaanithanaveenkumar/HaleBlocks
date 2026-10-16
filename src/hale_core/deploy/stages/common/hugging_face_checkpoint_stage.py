"""Save the adapter's Hugging Face form (model + tokenizer) for LLM serving stacks.

TensorRT-LLM (PyTorch backend), mlx-lm / mlx-vlm, vLLM-ROCm and Qualcomm Genie tooling all
start from an HF checkpoint directory rather than a traced graph.
"""

from __future__ import annotations

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import StageError
from hale_core.deploy.registry import STAGES

TRANSFORMERS = OptionalDependency("transformers", "pip install transformers")


@STAGES.register_module()
class HuggingFaceCheckpointStage(Stage):
    """``save_pretrained`` of ``adapter.hf_checkpoint(model)`` into ``<stage>/hf``.

    Args:
        safe_serialization: write ``.safetensors`` shards.
        max_shard_size: shard size passed to ``save_pretrained``.
        dtype: cast floating weights before saving (``bfloat16`` / ``float16`` / ``None``).
    """

    provides = ("hf_checkpoint",)
    dependencies = (TRANSFORMERS,)

    def __init__(
        self,
        safe_serialization: bool = True,
        max_shard_size: str = "5GB",
        dtype: str | None = "bfloat16",
        **base,
    ) -> None:
        super().__init__(**base)
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.dtype = dtype

    def run(self, ctx: ExportContext) -> list[Artifact]:
        import torch

        pair = ctx.adapter.hf_checkpoint(ctx.model)
        if pair is None:
            raise StageError(
                self.name,
                f"{type(ctx.adapter).__name__} does not provide a Hugging Face checkpoint",
            )
        model, tokenizer = pair
        if self.dtype:
            model = model.to(getattr(torch, self.dtype))
        out = ctx.stage_dir(self.name) / "hf"
        model.save_pretrained(
            out, safe_serialization=self.safe_serialization, max_shard_size=self.max_shard_size
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(out)
        logger.info("saved HF checkpoint {} ({})", out, type(model).__name__)
        return [
            self.artifact(
                ctx, "hf_checkpoint", out, architecture=type(model).__name__, dtype=self.dtype
            )
        ]
