"""Production model profiler: layer-by-layer latency, FLOPs, and matmul count.

    profiler = ModelProfiler(model)
    with profiler.profile():
        output = model(inputs)
    print(profiler.report())

Latency is measured with CUDA events on GPU (no mid-forward synchronisation)
and with ``time.perf_counter`` on CPU. FLOPs and matmul counts are computed
analytically via forward hooks (see _flops.py). An optional ``torch.profiler``
trace can be exported for Chrome / TensorBoard inspection.
"""

from hale_core.profiling._profiler.layer_profile import LayerProfile
from hale_core.profiling._profiler.model_profiler import ModelProfiler
from hale_core.profiling._profiler.profile_report import ProfileReport

__all__ = ["LayerProfile", "ModelProfiler", "ProfileReport"]
