"""LocoDatasetSpec.open_stream delegates to the HF streaming helper."""

from __future__ import annotations

import sys
import types

from hale_core.data.kinds import LocoTaskType
from hale_core.data.loco.types import LocoDatasetSpec


def test_open_stream_uses_hf_helper(monkeypatch, tmp_path):
    calls = {}

    def fake_load_dataset(**kwargs):
        calls.update(kwargs)
        return iter([{"qpos": [0.0]}])

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    task = next(iter(LocoTaskType))
    spec = LocoDatasetSpec(name="toy", task_type=task, hf_repo="org/toy")
    assert list(spec.open_stream(cache_dir=tmp_path)) == [{"qpos": [0.0]}]
    assert calls["path"] == "org/toy" and calls["cache_dir"].endswith("hf")
