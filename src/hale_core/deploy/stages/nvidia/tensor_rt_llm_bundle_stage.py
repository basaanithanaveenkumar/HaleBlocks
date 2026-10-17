"""Package an HF checkpoint for TensorRT-LLM (PyTorch backend), optionally ModelOpt-quantized.

Since TensorRT-LLM 1.2 the PyTorch backend is the only backend: there is no ``trtllm-build``
engine step any more. A deployable unit is an HF-format checkpoint (quantized checkpoints
come from ModelOpt's ``export_hf_checkpoint``) plus serving options for ``trtllm-serve`` /
``tensorrt_llm.LLM``. This stage produces exactly that bundle.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any, Literal

import yaml
from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

MODELOPT = OptionalDependency("modelopt", "pip install nvidia-modelopt[torch]")

#: ModelOpt quantization config names per scheme (``modelopt.torch.quantization``).
QUANT_CONFIGS = {
    "fp8": "FP8_DEFAULT_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "int8_sq": "INT8_SMOOTHQUANT_CFG",
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",
}


@STAGES.register_module()
class TensorRTLLMBundleStage(Stage):
    """HF checkpoint (+ ModelOpt PTQ) → a ``trtllm-serve`` bundle.

    Output: ``checkpoint/``, ``extra_llm_api_options.yaml`` and ``serve.sh``.

    Args:
        quantization: ``none`` or a key of :data:`QUANT_CONFIGS` (``fp8`` for Hopper/Ada,
            ``nvfp4`` for Blackwell, ``int4_awq`` for memory-bound Jetson / consumer GPUs).
        calib_batches: calibration forward passes for the PTQ (adapter / random batches).
        max_batch_size / max_num_tokens / max_seq_len: serving limits.
        kv_cache_free_gpu_memory_fraction: KV-cache share of free GPU memory.
        llm_api_options: extra keys merged into the YAML (``cuda_graph_config`` ...).
    """

    requires = ("hf_checkpoint",)
    provides = ("trtllm_bundle",)

    def __init__(
        self,
        quantization: Literal["none", "fp8", "nvfp4", "int4_awq", "int8_sq", "w4a8_awq"] = "none",
        calib_batches: int = 16,
        max_batch_size: int = 8,
        max_num_tokens: int = 8192,
        max_seq_len: int = 4096,
        kv_cache_free_gpu_memory_fraction: float = 0.85,
        llm_api_options: Mapping[str, Any] | None = None,
        **base,
    ) -> None:
        super().__init__(**base)
        self.quantization = quantization
        self.calib_batches = calib_batches
        self.limits = {
            "max_batch_size": max_batch_size,
            "max_num_tokens": max_num_tokens,
            "max_seq_len": max_seq_len,
        }
        self.kv_fraction = kv_cache_free_gpu_memory_fraction
        self.llm_api_options = dict(llm_api_options or {})

    def missing_dependencies(self):
        """Optional packages or executables this stage needs that are not installed."""
        return [MODELOPT] if self.quantization != "none" and not MODELOPT.available else []

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        source = ctx.store.require("hf_checkpoint")
        out = ctx.stage_dir(self.name)
        checkpoint = out / "checkpoint"
        if self.quantization == "none":
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            shutil.copytree(source.path, checkpoint)
        else:
            self._quantize(ctx, source.path, checkpoint)
        options = {
            **self.limits,
            "kv_cache_config": {"free_gpu_memory_fraction": self.kv_fraction},
            **self.llm_api_options,
        }
        (out / "extra_llm_api_options.yaml").write_text(yaml.safe_dump(options, sort_keys=False))
        serve = out / "serve.sh"
        serve.write_text(
            "#!/usr/bin/env bash\n"
            "# OpenAI-compatible server on the PyTorch backend of TensorRT-LLM\n"
            'cd "$(dirname "$0")"\n'
            "exec trtllm-serve ./checkpoint --backend pytorch "
            '--extra_llm_api_options extra_llm_api_options.yaml "$@"\n'
        )
        serve.chmod(0o755)
        return [
            self.artifact(ctx, "trtllm_bundle", out, quantization=self.quantization, **self.limits)
        ]

    def _quantize(self, ctx: ExportContext, source, checkpoint) -> None:
        """ModelOpt PTQ on the HF model with calibration forwards, exported as an HF checkpoint."""
        import modelopt.torch.quantization as mtq
        import torch
        from modelopt.torch.export import export_hf_checkpoint
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = (
            AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        vocab = model.config.vocab_size
        generator = torch.Generator().manual_seed(ctx.seed)

        def forward_loop(m) -> None:
            for batch in ctx.calibration.batches(ctx.signature):
                ids = batch.get("input_ids")
                if ids is None:
                    ids = torch.randint(0, vocab, (1, 256), generator=generator)
                m(ids.clamp(0, vocab - 1).to(device))

        logger.info("ModelOpt PTQ {} ({})", self.quantization, QUANT_CONFIGS[self.quantization])
        mtq.quantize(model, getattr(mtq, QUANT_CONFIGS[self.quantization]), forward_loop)
        export_hf_checkpoint(model, export_dir=str(checkpoint))
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ):
            if (source / name).exists():
                shutil.copy2(source / name, checkpoint / name)
