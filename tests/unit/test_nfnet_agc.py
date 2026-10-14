"""Adaptive Gradient Clipping (NFNet): runs on 4-D conv weights and clips unit-wise."""

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.nfnet import agc_clip


def test_agc_runs_on_conv_and_vector_parameters():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.LazyLinear(2))
    model(torch.randn(2, 3, 8, 8)).sum().backward()
    agc_clip(model, lambda_=0.01)  # used to raise: Tensor.norm(dim=[1, 2, 3]) on a 4-D weight
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_agc_clips_each_output_unit_to_lambda_times_weight_norm():
    conv = nn.Conv2d(2, 3, 3, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)  # every unit has ||w|| = sqrt(2 * 3 * 3) = sqrt(18)
    conv.weight.grad = torch.full_like(conv.weight, 10.0)
    conv.weight.grad[1] = 1e-5  # this unit is already below the threshold
    agc_clip(conv, lambda_=0.01)
    unit_norms = torch.linalg.vector_norm(conv.weight.grad, dim=(1, 2, 3))
    limit = 0.01 * 18**0.5
    assert torch.allclose(unit_norms[[0, 2]], torch.tensor([limit, limit]), rtol=1e-5)
    assert torch.allclose(conv.weight.grad[1], torch.full_like(conv.weight.grad[1], 1e-5))
