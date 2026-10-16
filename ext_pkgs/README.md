# ext_pkgs — vendor SDK sources (third party)

Shallow git submodules of the upstream projects the deployment pipelines target. They are
**reference sources** (samples, op-support tables, converter internals, release notes) kept
next to the code that drives them; nothing here is imported at runtime. Runtime dependencies
are installed with the pip extras (`hale-blocks[deploy-nvidia]`, `[deploy-apple]`,
`[deploy-amd]`, `[deploy-qualcomm]`).

```bash
git submodule update --init --depth 1 ext_pkgs/<name>   # one
git submodule update --init --depth 1                    # all (never --recursive)
```

| name | upstream | vendor | why it is here | used by |
|---|---|---|---|---|
| `TensorRT` | [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) | NVIDIA | ONNX parser, plugins, strong-typing samples, `trtexec` | `TensorRTEngineStage`, `TensorRTSession` |
| `TensorRT-LLM` | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA | PyTorch-backend model support, `trtllm-serve` options | `TensorRTLLMBundleStage` |
| `TensorRT-Edge-LLM` | [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | NVIDIA | on-device LLM/VLM runtime for Jetson Thor / DRIVE | Jetson LLM deployments (docs) |
| `Model-Optimizer` | [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVIDIA | AutoCast, ONNX PTQ (INT8/FP8/INT4), HF checkpoint export | `ModelOptAutoCastStage`, `ModelOptQuantizeStage`, `TensorRTLLMBundleStage` |
| `Torch-TensorRT` | [pytorch/TensorRT](https://github.com/pytorch/TensorRT) | NVIDIA / PyTorch | dynamo lowering, converter coverage | `TorchTensorRTStage` |
| `mlx` | [ml-explore/mlx](https://github.com/ml-explore/mlx) | Apple | ops, `export_function`, quantized matmul | `MLXConvertStage`, ATen→MLX bridge |
| `mlx-lm` | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple | HF → MLX LLM conversion and server | `MLXHuggingFaceConvertStage` |
| `AMDMIGraphX` | [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | AMD | ONNX op support, Python API, `migraphx-driver` | `MIGraphXCompileStage`, `MIGraphXSession` |
| `Quark` | [amd/Quark](https://github.com/amd/Quark) | AMD | ONNX quantization presets (A16W8, BF16, XINT8) | `QuarkQuantizeStage` |
| `RyzenAI-SW` | [amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW) | AMD | Vitis AI EP examples, NPU deployment tutorials | `ryzen_ai_npu_*` configs |
| `ai-hub-models` | [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) | Qualcomm | reference export/compile/profile scripts per model | `QAIHubCompileStage` |
| `executorch` | [pytorch/executorch](https://github.com/pytorch/executorch) | PyTorch / Qualcomm | QNN backend (`backends/qualcomm`), Android runtime | `ExecuTorchQNNStage` |

Submodules are pinned to the commit recorded in the superproject; update one with
`git -C ext_pkgs/<name> fetch --depth 1 origin <ref> && git -C ext_pkgs/<name> checkout FETCH_HEAD`
and commit the new pointer.
