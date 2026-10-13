"""Model export pipeline: ONNX, TensorRT, MLX.

Usage::

    from hale_core.export import ExportManager

    exporter = ExportManager(cfg.train.export, model, tokenizer)
    results = exporter.run_all(sample_input)
    # results: {"onnx": "/path/model.onnx", "tensorrt": "/path/model.trt", ...}

All sub-modules use lazy imports for optional dependencies (tensorrt, onnx, mlx,
safetensors) so this package can be imported even when those packages are absent.
"""

# Lazy imports to avoid pulling in optional heavy deps at package import time.
# Import the sub-modules directly when you need them:
#   from hale_core.export.manager import ExportManager
#   from hale_core.export.onnx import export_onnx

__all__ = ["ExportManager", "export_onnx", "export_tensorrt", "export_mlx"]


def __getattr__(name: str):
    if name == "ExportManager":
        from hale_core.export.manager import ExportManager
        return ExportManager
    if name == "export_onnx":
        from hale_core.export.onnx import export_onnx
        return export_onnx
    if name == "export_tensorrt":
        from hale_core.export.tensorrt import export_tensorrt
        return export_tensorrt
    if name == "export_mlx":
        from hale_core.export.mlx import export_mlx
        return export_mlx
    raise AttributeError(f"module 'hale_core.export' has no attribute {name!r}")
