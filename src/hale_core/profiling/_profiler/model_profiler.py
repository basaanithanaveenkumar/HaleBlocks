"""``ModelProfiler``: Layer-by-layer profiler for latency, FLOPs and matmul counts."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from hale_core.profiling._flops import get_flops_hook
from hale_core.profiling._profiler.layer_profile import LayerProfile
from hale_core.profiling._profiler.profile_report import ProfileReport


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
                self._hooks.append(module.register_forward_pre_hook(self._make_cuda_pre(name, se)))
                self._hooks.append(module.register_forward_hook(self._make_cuda_post(name, se, ee)))
            else:
                self._hooks.append(module.register_forward_pre_hook(self._make_cpu_pre(name)))
                self._hooks.append(module.register_forward_hook(self._make_cpu_post(name)))

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

    def _make_cuda_post(self, name: str, se: torch.cuda.Event, ee: torch.cuda.Event) -> Any:
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
