#!/usr/bin/env python3
"""Profile HaleBlocks backbone models: latency, FLOPs, and matmul count per layer.

Usage examples:
    python scripts/profile_model.py                          # GPT-small, CPU
    python scripts/profile_model.py --variant gpt --device cuda
    python scripts/profile_model.py --variant dit --d_model 512 --n_layers 12
    python scripts/profile_model.py --torch-profiler --trace out/trace.json
    python scripts/profile_model.py --sort flops --top 30

Supported variants: gpt, lgt, dit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from loguru import logger

from hale_core.profiling import ModelProfiler
from hale_core.nn import GPTBackbone, LGTBackbone, DiTBackbone


# ---------------------------------------------------------------------------
# Model builders (dummy instances, no real weights needed)
# ---------------------------------------------------------------------------

def build_gpt(args) -> tuple[torch.nn.Module, torch.Tensor]:
    model = GPTBackbone(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.seq_len,
        dropout=0.0,
        attn_type="causal",
        attn_impl="mha",
        ffn_type="mlp",
    )
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


def build_lgt(args) -> tuple[torch.nn.Module, torch.Tensor]:
    model = LGTBackbone(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.seq_len,
        dropout=0.0,
    )
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


def build_dit(args) -> tuple[torch.nn.Module, torch.Tensor]:
    model = DiTBackbone(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.seq_len,
        dropout=0.0,
        use_time_cond=True,
    )
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


_BUILDERS = {"gpt": build_gpt, "lgt": build_lgt, "dit": build_dit}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HaleBlocks model profiler")
    p.add_argument("--variant",      default="gpt", choices=list(_BUILDERS))
    p.add_argument("--device",       default="cpu")
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--seq-len",      type=int, default=128)
    p.add_argument("--vocab-size",   type=int, default=8192)
    p.add_argument("--d-model",      type=int, default=256)
    p.add_argument("--n-heads",      type=int, default=4)
    p.add_argument("--n-layers",     type=int, default=6)
    p.add_argument("--d-ff",         type=int, default=1024)
    p.add_argument("--top",          type=int, default=40, help="rows in summary table")
    p.add_argument("--sort",         default="latency", choices=["latency", "flops", "matmuls"])
    p.add_argument("--warmup",       type=int, default=2, help="warmup forward passes")
    p.add_argument("--torch-profiler", action="store_true", help="enable torch.profiler trace")
    p.add_argument("--trace",        default=None, help="path to export chrome trace JSON")
    p.add_argument("--json",         default=None, help="path to export per-layer JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info("profiling variant={} device={} batch={} seq={} d_model={} n_layers={}",
                args.variant, device, args.batch_size, args.seq_len, args.d_model, args.n_layers)

    builder = _BUILDERS[args.variant]
    model, dummy = builder(args)
    model = model.to(device).eval()
    dummy = dummy.to(device)

    # Warmup passes — not profiled
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)

    profiler = ModelProfiler(
        model,
        use_cuda_events=(device.type == "cuda"),
        use_torch_profiler=args.torch_profiler,
        torch_profiler_trace=args.trace,
    )

    with torch.no_grad(), profiler.profile():
        _ = model(dummy)

    report = profiler.report()

    # The report itself is this script's output, so it goes to stdout; status lines use the logger.
    print("\n" + report.summary_table(top_k=args.top, sort_by=args.sort))

    if args.torch_profiler:
        print("\n── torch.profiler key averages ──")
        print(report.torch_key_averages())

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("per-layer JSON saved → {}", args.json)

    if args.trace:
        logger.info("Chrome trace saved → {}", args.trace)


if __name__ == "__main__":
    main()
