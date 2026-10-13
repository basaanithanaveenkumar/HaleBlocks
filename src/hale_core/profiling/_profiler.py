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

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from hale_core.profiling._flops import get_flops_hook


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class ModelProfiler:
    """Layer-by-layer profiler for latency, FLOPs and matmul counts.

    Args:
        model: the ``nn.Module`` to profile.
        use_cuda_events: use CUDA events for GPU timing (recommended on GPU).
        use_torch_profiler: also run ``torch.profiler.profile`` for Chrome
            traces and per-operator wall time (adds overhead ~5%).
        torch_profiler_trace: if set, export a Chrome trace to this path
            after each ``profile()`` call.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        use_cuda_events: bool = True,
        use_torch_profiler: bool = False,
        torch_profiler_trace: str | None = None,
    ) -> None:
        self.model = model
        self._device = self._infer_device(model)
        self._use_cuda = use_cuda_events and self._device.type == "cuda"
        self._use_torch_profiler = use_torch_profiler
        self._trace_path = torch_profiler_trace
        self._profiles: dict[str, LayerProfile] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._torch_prof: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def profile(self) -> Generator[None, None, None]:
        """Context manager: run your forward pass inside to collect profiles.

        Example::

            with profiler.profile():
                out = model(x)
            print(profiler.report())
        """
        self._reset()
        self._register_hooks()
        if self._use_torch_profiler:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self._use_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch_prof = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
                with_modules=True,
                profile_memory=True,
            )
            torch_prof.__enter__()
            try:
                yield
            finally:
                torch_prof.__exit__(None, None, None)
                self._torch_prof = torch_prof
                self._finalize()
                self._remove_hooks()
                if self._trace_path:
                    self._torch_prof.export_chrome_trace(self._trace_path)
        else:
            try:
                yield
            finally:
                self._finalize()
                self._remove_hooks()

    def report(self) -> ProfileReport:
        """Return a :class:`ProfileReport` from the last ``profile()`` call."""
        return ProfileReport(
            profiles=dict(self._profiles),
            torch_prof=self._torch_prof,
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _tensor_shape(x: Any) -> tuple:
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            for item in x:
                if isinstance(item, torch.Tensor):
                    return tuple(item.shape)
        return ()

    def _reset(self) -> None:
        self._profiles.clear()
        self._torch_prof = None

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name == "":
                continue

            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            self._profiles[name] = LayerProfile(
                name=name,
                module_type=type(module).__name__,
                params=own_params,
            )

            if self._use_cuda:
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                self._hooks.append(
                    module.register_forward_pre_hook(self._make_cuda_pre(name, se))
                )
                self._hooks.append(
                    module.register_forward_hook(self._make_cuda_post(name, se, ee))
                )
            else:
                self._hooks.append(
                    module.register_forward_pre_hook(self._make_cpu_pre(name))
                )
                self._hooks.append(
                    module.register_forward_hook(self._make_cpu_post(name))
                )

            flops_hook = get_flops_hook(name, module, self._profiles)
            if flops_hook is not None:
                self._hooks.append(module.register_forward_hook(flops_hook))

    # -- CUDA timing hooks -----------------------------------------------

    def _make_cuda_pre(self, name: str, se: torch.cuda.Event) -> Any:
        prof = self._profiles

        def pre(mod, inp) -> None:
            se.record()
            x = inp[0] if isinstance(inp, tuple) and inp else inp
            prof[name].input_shape = self._tensor_shape(x)

        return pre

    def _make_cuda_post(
        self, name: str, se: torch.cuda.Event, ee: torch.cuda.Event
    ) -> Any:
        prof = self._profiles

        def post(mod, inp, out) -> None:
            ee.record()
            prof[name]._start_event = se
            prof[name]._end_event = ee
            prof[name].output_shape = self._tensor_shape(out)

        return post

    # -- CPU timing hooks ------------------------------------------------

    def _make_cpu_pre(self, name: str) -> Any:
        prof = self._profiles

        def pre(mod, inp) -> None:
            mod._prof_t0 = time.perf_counter()
            x = inp[0] if isinstance(inp, tuple) and inp else inp
            prof[name].input_shape = self._tensor_shape(x)

        return pre

    def _make_cpu_post(self, name: str) -> Any:
        prof = self._profiles

        def post(mod, inp, out) -> None:
            t0 = getattr(mod, "_prof_t0", None)
            if t0 is not None:
                prof[name].latency_ms = (time.perf_counter() - t0) * 1000.0
            prof[name].output_shape = self._tensor_shape(out)

        return post

    # -- Cleanup ---------------------------------------------------------

    def _finalize(self) -> None:
        if self._use_cuda:
            torch.cuda.synchronize()
            for p in self._profiles.values():
                p.finalize_cuda_timing()

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ProfileReport:
    """Immutable snapshot of a profiling run."""

    profiles: dict[str, LayerProfile]
    torch_prof: Any
    device: torch.device

    # ------------------------------------------------------------------

    @property
    def total_latency_ms(self) -> float:
        """Sum of all leaf-layer latencies (may double-count nested modules)."""
        return sum(p.latency_ms for p in self.profiles.values())

    @property
    def total_flops(self) -> int:
        return sum(p.flops for p in self.profiles.values())

    @property
    def total_matmuls(self) -> int:
        return sum(p.matmul_count for p in self.profiles.values())

    def top_by_latency(self, k: int = 20) -> list[LayerProfile]:
        return sorted(self.profiles.values(), key=lambda p: p.latency_ms, reverse=True)[:k]

    def top_by_flops(self, k: int = 20) -> list[LayerProfile]:
        return sorted(self.profiles.values(), key=lambda p: p.flops, reverse=True)[:k]

    # ------------------------------------------------------------------

    def summary_table(self, *, top_k: int = 40, sort_by: str = "latency") -> str:
        """Return a formatted text table.

        Args:
            top_k: maximum number of rows to display.
            sort_by: one of ``"latency"``, ``"flops"``, or ``"matmuls"``.
        """
        key_fn = {
            "latency": lambda p: p.latency_ms,
            "flops":   lambda p: p.flops,
            "matmuls": lambda p: p.matmul_count,
        }.get(sort_by, lambda p: p.latency_ms)

        rows = sorted(self.profiles.values(), key=key_fn, reverse=True)[:top_k]

        total_lat = max(sum(p.latency_ms for p in self.profiles.values()), 1e-9)
        total_flops = max(sum(p.flops for p in self.profiles.values()), 1)

        col_w = {
            "name":    44,
            "type":    22,
            "lat":      9,
            "pct":      7,
            "gflops":   9,
            "mm":       6,
            "params":   9,
            "out":     20,
        }
        bar = "-" * (sum(col_w.values()) + len(col_w) - 1)
        hdr = (
            f"{'Layer':<{col_w['name']}}"
            f"{'Type':<{col_w['type']}}"
            f"{'Lat(ms)':>{col_w['lat']}}"
            f"{'  %':>{col_w['pct']}}"
            f"{'GFLOPs':>{col_w['gflops']}}"
            f"{'#MM':>{col_w['mm']}}"
            f"{'Params':>{col_w['params']}}"
            f"  {'Output shape':<{col_w['out']}}"
        )

        lines = [
            f"  Device: {self.device}  |  "
            f"Total latency: {total_lat:.2f} ms  |  "
            f"Total GFLOPs: {total_flops / 1e9:.3f}  |  "
            f"Total matmuls: {self.total_matmuls:,}",
            bar,
            hdr,
            bar,
        ]

        for p in rows:
            name = p.name if len(p.name) <= col_w["name"] else "…" + p.name[-(col_w["name"] - 1):]
            typ  = p.module_type[:col_w["type"]]
            lat  = f"{p.latency_ms:.3f}"
            pct  = f"{100 * p.latency_ms / total_lat:.1f}%"
            gf   = f"{p.flops / 1e9:.4f}" if p.flops else "-"
            mm   = str(p.matmul_count) if p.matmul_count else "-"
            par  = f"{p.params / 1e6:.2f}M" if p.params >= 1_000 else str(p.params)
            out  = str(p.output_shape)[:col_w["out"]]
            lines.append(
                f"{name:<{col_w['name']}}"
                f"{typ:<{col_w['type']}}"
                f"{lat:>{col_w['lat']}}"
                f"{pct:>{col_w['pct']}}"
                f"{gf:>{col_w['gflops']}}"
                f"{mm:>{col_w['mm']}}"
                f"{par:>{col_w['params']}}"
                f"  {out:<{col_w['out']}}"
            )

        lines.append(bar)
        lines.append(
            f"{'TOTAL':<{col_w['name']}}"
            f"{'':>{col_w['type']}}"
            f"{total_lat:>{col_w['lat']}.3f}"
            f"{'100%':>{col_w['pct']}}"
            f"{total_flops / 1e9:>{col_w['gflops']}.3f}"
            f"{self.total_matmuls:>{col_w['mm']}}"
            f"{'':>{col_w['params']}}"
        )
        lines.append(bar)
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict]:
        return {
            name: {
                "module_type":   p.module_type,
                "params":        p.params,
                "latency_ms":    p.latency_ms,
                "flops":         p.flops,
                "matmul_count":  p.matmul_count,
                "input_shape":   p.input_shape,
                "output_shape":  p.output_shape,
            }
            for name, p in self.profiles.items()
        }

    def save_chrome_trace(self, path: str) -> None:
        """Export a Chrome JSON trace (requires ``use_torch_profiler=True``)."""
        if self.torch_prof is None:
            raise RuntimeError("torch_profiler was not enabled. Pass use_torch_profiler=True.")
        self.torch_prof.export_chrome_trace(path)

    def save_stacks(self, path: str, *, metric: str = "self_cuda_time_total") -> None:
        """Export flame-graph stacks (requires ``use_torch_profiler=True``)."""
        if self.torch_prof is None:
            raise RuntimeError("torch_profiler was not enabled. Pass use_torch_profiler=True.")
        self.torch_prof.export_stacks(path, metric=metric)

    def torch_key_averages(self, *, group_by_shape: bool = False) -> str:
        """Pretty-print the torch.profiler key averages table."""
        if self.torch_prof is None:
            return "(torch_profiler not enabled)"
        sort_key = (
            torch.profiler.ProfilerSortKey.CUDA_TIME_TOTAL
            if self.device.type == "cuda"
            else torch.profiler.ProfilerSortKey.CPU_TIME_TOTAL
        )
        return str(
            self.torch_prof.key_averages(group_by_input_shape=group_by_shape).table(
                sort_by=sort_key, row_limit=50
            )
        )

    def __str__(self) -> str:
        return self.summary_table()

    def __repr__(self) -> str:
        return (
            f"ProfileReport(layers={len(self.profiles)}, "
            f"total_latency={self.total_latency_ms:.2f} ms, "
            f"total_gflops={self.total_flops / 1e9:.3f})"
        )
