import torch

import hale_core.optimizers  # noqa: F401
from hale_core.backbones import BACKBONES, TransformerBackbone, build_backbone
from hale_core.components import build_attn_mask
from hale_core.optimizers import build_optimizer
from hale_core.registry import OPTIMIZERS, get_optimizer
from hale_core.tensors import format_model_summary


def test_backbone_registry():
    assert set(BACKBONES) >= {"default", "transformer", "lgt", "dit"}


def test_causal_mask_upper_triangle():
    m = build_attn_mask("causal", 4, "cpu")
    assert m[0, 1]
    assert not m[1, 0]


def test_backbone_causal_forward():
    m = TransformerBackbone(vocab_size=32, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    x = torch.randint(0, 32, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 32)


def test_build_backbone_factory():
  model = build_backbone(32, arch="transformer", d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
  x = torch.randint(0, 32, (2, 8))
  assert model(x).shape == (2, 8, 32)


def test_format_model_summary_lists_modules():
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

    text = format_model_summary(Wrapper(), title="model summary")
    assert "TOTAL" in text
    assert "Trainable" in text
    assert "backbone" in text


def test_optimizer_registry_has_builtins():
    assert {"adam", "adamw", "sgd", "muon"} <= set(OPTIMIZERS)
    assert get_optimizer("adamw") is get_optimizer("AdamW")


def test_build_adamw_has_torch_api():
    import torch.nn as nn

    model = nn.Linear(4, 2)
    opt = build_optimizer("adamw", model.parameters(), {"lr": 1e-3, "weight_decay": 0.0})
    loss = model(torch.ones(3, 4)).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert "lr" in opt.param_groups[0]
    state = opt.state_dict()
    opt.load_state_dict(state)
