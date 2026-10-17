"""Compare backend outputs against the PyTorch reference with explicit tolerances."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from loguru import logger

from hale_core.deploy.exceptions import ParityError
from hale_core.deploy.validation.parity_metrics import ParityMetrics


class ParityChecker:
    """Pass if, for every output, ``max_abs <= atol`` **or** ``cosine >= min_cosine``.

    The OR is deliberate: fp32 exports are judged on absolute error, while fp16 / int8 / 4-bit
    artifacts legitimately exceed a tight ``atol`` yet must still point the same way (cosine)
    and pick the same tokens (``min_argmax_agreement``, when set).

    Args:
        atol: absolute tolerance.
        min_cosine: cosine-similarity floor.
        min_argmax_agreement: optional floor on last-axis argmax agreement (logits).
    """

    def __init__(
        self,
        atol: float = 1e-3,
        min_cosine: float = 0.999,
        min_argmax_agreement: float | None = None,
    ) -> None:
        self.atol = atol
        self.min_cosine = min_cosine
        self.min_argmax_agreement = min_argmax_agreement

    def check(
        self,
        names: Sequence[str],
        reference: Sequence[torch.Tensor],
        candidate: Sequence[np.ndarray],
        label: str = "",
    ) -> dict[str, dict]:
        """Compare candidate outputs with the PyTorch reference, output by output.

        An output passes if it is within ``atol`` or above the cosine-similarity floor (and, if
        configured, matches the argmax closely enough).

        Args:
            names: output names.
            reference: PyTorch outputs.
            candidate: runtime outputs, same order.
            label: tag for log lines (e.g. the shape corner).

        Returns:
            name -> metrics of every output.

        Raises:
            ParityError: if the output counts differ or any output fails.
        """
        if len(reference) != len(candidate):
            raise ParityError(
                f"{label}: expected {len(reference)} outputs, backend returned {len(candidate)}"
            )
        report: dict[str, dict] = {}
        failures = []
        for name, ref, cand in zip(names, reference, candidate, strict=True):
            ref_np = (
                ref.detach().float().cpu().numpy() if ref.is_floating_point() else ref.cpu().numpy()
            )
            metrics = ParityMetrics.compute(ref_np, np.asarray(cand))
            ok = metrics.max_abs <= self.atol or metrics.cosine >= self.min_cosine
            if self.min_argmax_agreement is not None and metrics.argmax_agreement is not None:
                ok = ok and metrics.argmax_agreement >= self.min_argmax_agreement
            report[name] = {**metrics.to_dict(), "ok": ok}
            logger.info(
                "parity {} {}: max_abs={:.3e} cosine={:.6f} argmax={} -> {}",
                label,
                name,
                metrics.max_abs,
                metrics.cosine,
                metrics.argmax_agreement,
                "ok" if ok else "FAIL",
            )
            if not ok:
                failures.append(name)
        if failures:
            raise ParityError(
                f"{label}: outputs {failures} exceed tolerance (atol={self.atol}, "
                f"min_cosine={self.min_cosine})",
                report,
            )
        return report
