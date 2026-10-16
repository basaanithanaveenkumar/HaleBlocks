"""Parity metrics and the tolerance policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hale_core.deploy.exceptions import ParityError
from hale_core.deploy.validation import ParityChecker, ParityMetrics


def test_metrics_values() -> None:
    ref = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    m = ParityMetrics.compute(ref, ref + 0.5)
    assert m.max_abs == pytest.approx(0.5) and m.argmax_agreement == 1.0 and 0.9 < m.cosine < 1.0
    assert ParityMetrics.compute(np.zeros(3), np.zeros(3)).cosine == 1.0


def test_checker_accepts_by_atol_or_cosine_and_rejects_otherwise() -> None:
    ref = torch.randn(4, 10)
    ParityChecker(atol=1e-6).check(["y"], [ref], [ref.numpy()])
    scaled = (ref * 1.001).numpy()  # large abs error on big values, same direction
    ParityChecker(atol=1e-9, min_cosine=0.999).check(["y"], [ref], [scaled])
    with pytest.raises(ParityError) as info:
        ParityChecker(atol=1e-3, min_cosine=0.999).check(["y"], [ref], [(-ref).numpy()])
    assert info.value.metrics["y"]["ok"] is False
    with pytest.raises(ParityError, match="argmax"):
        ParityChecker(atol=10, min_argmax_agreement=1.0).check(
            ["y"], [ref], [np.flip(ref.numpy(), -1).copy()], "argmax"
        )
    with pytest.raises(ParityError, match="outputs"):
        ParityChecker().check(["y"], [ref], [])
