from hale_core.nn.losses import TOKEN_LOSSES, token_nll


def test_token_loss_registry_has_ce():
    assert "ce" in TOKEN_LOSSES


def test_token_nll_shape():
    import torch

    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    loss = token_nll(logits, targets, loss_type="ce", reduction="mean")
    assert loss.ndim == 0
