import torch
import torch.nn as nn

from hale_core.runtime.checkpoint import CheckpointStore


def test_checkpoint_roundtrip(tmp_path):
    store = CheckpointStore()
    path = tmp_path / "ckpt.pt"
    model = nn.Linear(4, 2)
    state = {"model_state_dict": model.state_dict(), "step": 3}
    store.save(str(path), **state)
    assert store.exists(str(path))
    loaded = store.load(str(path))
    assert loaded["step"] == 3
    model2 = nn.Linear(4, 2)
    model2.load_state_dict(loaded["model_state_dict"])
    x = torch.randn(2, 4)
    assert torch.allclose(model(x), model2(x))
