"""Production model profiling: latency, FLOPs, and matmul counts per layer.

Quick start::

    from hale_core.profiling import ModelProfiler

    profiler = ModelProfiler(model, use_torch_profiler=True)
    with profiler.profile():
        _ = model(dummy_input)

    report = profiler.report()
    print(report)                          # layer-by-layer table
    report.save_chrome_trace("trace.json") # open in chrome://tracing
"""

from hale_core.profiling._profiler import LayerProfile, ModelProfiler, ProfileReport

__all__ = [
    "ModelProfiler",
    "LayerProfile",
    "ProfileReport",
]
