"""Integration tests with mock training plugins."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import hale_core.registry.bootstrap  # noqa: F401
from hale_core.config import load_config
from hale_core.registry import register_loss, register_model
from hale_core.training.trainer import Trainer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@register_model("mock_ar")
class _MockModel(nn.Module):
    def __init__(self, vocab_size: int, cfg=None, **kwargs) -> None:
        super().__init__()
        d = cfg.model.d_model if cfg is not None else 16
        self.emb = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, batch):
        x = batch["input_ids"]
        return self.head(self.emb(x))


@register_loss("mock_ar")
def _mock_loss(model, batch):
    logits = model(batch)
    return logits.float().pow(2).mean()


class _MockData:
    def __init__(self, tokenizer, batch_size: int, seq_len: int, n_batches: int):
        self.tokenizer = tokenizer
        self._loader = DataLoader(
            TensorDataset(torch.randint(2, 32, (n_batches * batch_size, seq_len))),
            batch_size=batch_size,
            drop_last=True,
        )

    def train_loader(self):
        loader = self._loader

        class _DictLoader:
            def __init__(self, inner):
                self._inner = inner

            def __iter__(self):
                for (ids,) in self._inner:
                    yield {"input_ids": ids}

            def __len__(self):
                return len(self._inner)

        return _DictLoader(loader)

    def val_loader(self, train_loader):
        return self._loader

    def viz_prompt(self, step: int) -> str:
        return "hi"


class _MockEval:
    def run(self, model, loader) -> dict[str, float]:
        return {"loss": 1.0}


@pytest.mark.integration
def test_trainer_runs_few_steps(tiny_tokenizer, tmp_path):
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg.train.steps = 6
    cfg.train.batch_size = 2
    cfg.train.checkpoint_path = str(tmp_path / "last.pt")
    cfg.train.checkpoint_every_epoch = False
    cfg.logging.backend = "noop"
    cfg.logging.log_file = None
    cfg.device = "cpu"

    trainer = Trainer(
        cfg,
        tiny_tokenizer,
        data_module=_MockData(tiny_tokenizer, batch_size=2, seq_len=16, n_batches=4),
        evaluator=_MockEval(),
    )
    model, tok, losses = trainer.fit()
    assert len(losses) >= 4
    assert losses[-1] < losses[0]
    assert Path(cfg.train.checkpoint_path).exists()


@pytest.mark.integration
def test_config_fixture_inheritance():
    cfg = load_config(FIXTURES / "child.yaml")
    assert cfg.model.d_model == 48
    assert cfg.train.steps == 8
