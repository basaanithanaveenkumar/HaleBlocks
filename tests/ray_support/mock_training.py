"""A tiny model / loss / data module / evaluator that Ray workers can import."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VARIANT = "mock_ray"


class MockTokenizer:
    """Just enough tokenizer for the trainer (it only needs ``len``)."""

    vocab_size = 64

    def __len__(self) -> int:
        return self.vocab_size


class MockModel(nn.Module):
    def __init__(self, vocab_size: int, cfg=None, **kwargs) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.head = nn.Linear(16, vocab_size)

    def forward(self, batch):
        return self.head(self.emb(batch["input_ids"]))


def mock_loss(model, batch):
    return model(batch).float().pow(2).mean()


class DictLoader:
    def __init__(self, inner: DataLoader) -> None:
        self.inner = inner
        self.sampler = inner.sampler

    def __iter__(self):
        for (ids,) in self.inner:
            yield {"input_ids": ids}

    def __len__(self) -> int:
        return len(self.inner)


class MockData:
    def __init__(self, batch_size: int = 2, seq_len: int = 16, n_batches: int = 4) -> None:
        self.tokenizer = MockTokenizer()
        ids = torch.randint(
            2, 32, (n_batches * batch_size, seq_len), generator=torch.Generator().manual_seed(0)
        )
        self.loader = DataLoader(TensorDataset(ids), batch_size=batch_size, drop_last=True)

    def train_loader(self):
        return DictLoader(self.loader)

    def val_loader(self, train_loader):
        return DictLoader(self.loader)

    def viz_prompt(self, step: int) -> str:
        return "hi"


class MockEvaluator:
    def run(self, model, loader) -> dict[str, float]:
        return {"loss": 1.0}


def register_mock_variant() -> None:
    """Register the model and loss (idempotent: runs in the driver and in every worker)."""
    from hale_core.registry import LOSSES, MODELS, register_loss, register_model

    if VARIANT not in MODELS:
        register_model(VARIANT)(MockModel)
    if VARIANT not in LOSSES:
        register_loss(VARIANT)(mock_loss)


def make_data_module() -> MockData:
    """``data_module_factory``: called inside each worker."""
    register_mock_variant()
    return MockData()


def report_rank(loop_config: dict) -> None:
    """Train function for the launcher test: all-reduce the ranks and report with a checkpoint."""
    import torch.distributed as dist

    from hale_core.distributed import RayReporter, WorkerContext

    ctx = WorkerContext.current()
    assert ctx.in_ray and dist.is_initialized() and ctx.world_size == loop_config["expect"]
    total = torch.tensor([float(ctx.rank)])
    dist.all_reduce(total)  # the process group really works
    with tempfile.TemporaryDirectory() as ckpt:  # Ray V2 keeps metrics with checkpoints
        Path(ckpt, "state.pt").write_bytes(b"x")
        RayReporter().report(
            {"rank_sum": total.item(), "world": ctx.world_size}, checkpoint_dir=ckpt
        )
