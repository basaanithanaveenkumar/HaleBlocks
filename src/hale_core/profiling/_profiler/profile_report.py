"""``ProfileReport``: Immutable snapshot of a profiling run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hale_core.profiling._profiler.layer_profile import LayerProfile


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
            "flops": lambda p: p.flops,
            "matmuls": lambda p: p.matmul_count,
        }.get(sort_by, lambda p: p.latency_ms)

        rows = sorted(self.profiles.values(), key=key_fn, reverse=True)[:top_k]

        total_lat = max(sum(p.latency_ms for p in self.profiles.values()), 1e-9)
        total_flops = max(sum(p.flops for p in self.profiles.values()), 1)

        col_w = {
            "name": 44,
            "type": 22,
            "lat": 9,
            "pct": 7,
            "gflops": 9,
            "mm": 6,
            "params": 9,
            "out": 20,
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
            name = p.name if len(p.name) <= col_w["name"] else "…" + p.name[-(col_w["name"] - 1) :]
            typ = p.module_type[: col_w["type"]]
            lat = f"{p.latency_ms:.3f}"
            pct = f"{100 * p.latency_ms / total_lat:.1f}%"
            gf = f"{p.flops / 1e9:.4f}" if p.flops else "-"
            mm = str(p.matmul_count) if p.matmul_count else "-"
            par = f"{p.params / 1e6:.2f}M" if p.params >= 1_000 else str(p.params)
            out = str(p.output_shape)[: col_w["out"]]
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
                "module_type": p.module_type,
                "params": p.params,
                "latency_ms": p.latency_ms,
                "flops": p.flops,
                "matmul_count": p.matmul_count,
                "input_shape": p.input_shape,
                "output_shape": p.output_shape,
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
