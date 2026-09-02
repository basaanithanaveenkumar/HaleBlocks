import pytest
import torch


@pytest.mark.smoke
def test_gpt_backbone_forward():
    from hale_core.nn.backbones import TransformerBackbone

    m = TransformerBackbone(vocab_size=32, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    x = torch.randint(0, 32, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 32)
    assert torch.isfinite(y).all()


@pytest.mark.smoke
def test_token_nll_scalar():
    from hale_core.nn.losses import token_nll

    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    loss = token_nll(logits, targets, loss_type="ce", reduction="mean")
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.smoke
def test_optimizer_step():
    import torch.nn as nn

    from hale_core.nn.optim import build_optimizer

    model = nn.Linear(4, 2)
    opt = build_optimizer("adamw", model.parameters(), {"lr": 1e-3, "weight_decay": 0.0})
    loss = model(torch.ones(2, 4)).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert "lr" in opt.param_groups[0]
