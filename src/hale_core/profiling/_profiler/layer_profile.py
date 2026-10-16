"""``LayerProfile``: LayerProfile (Production model profiler: layer-by-layer latency, FLOPs, and matmul count)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerProfile:
    name: str
    module_type: str
    params: int = 0
    latency_ms: float = 0.0
    flops: int = 0
    matmul_count: int = 0
    input_shape: tuple = ()
    output_shape: tuple = ()
    # CUDA timing handles — cleared after finalize
    _start_event: Any = field(default=None, repr=False)
    _end_event: Any = field(default=None, repr=False)

    def finalize_cuda_timing(self) -> None:
        if self._start_event is not None and self._end_event is not None:
            self._end_event.synchronize()
            self.latency_ms = self._start_event.elapsed_time(self._end_event)
        self._start_event = None
        self._end_event = None
