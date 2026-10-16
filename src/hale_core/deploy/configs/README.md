# Packaged deployment configs

Inherit them from any repo with `_base_: pkg://hale_core.deploy/configs/<path>.yaml` and add an
`adapter` (and usually `custom_imports`). Layout:

| directory  | contents |
|------------|----------|
| `_base_/`  | defaults and the shared ONNX front half |
| `devices/` | device profiles (vendor, chips, runtime, precisions) |
| `nvidia/`  | TensorRT 10/11 (FP16, INT8, FP8), TensorRT-RTX, Jetson Orin DLA/INT8, Torch-TensorRT, TensorRT-LLM |
| `apple/`   | MLX bridge (fp32/fp16/int4), mlx-lm, mlx-vlm |
| `amd/`     | MIGraphX fp16/int8, ORT MIGraphX EP, Ryzen AI NPU (Quark A16W8 / BF16) |
| `qualcomm/`| AI Hub (QNN DLC, TFLite, context binary, ONNX for X Elite), ExecuTorch QNN |
