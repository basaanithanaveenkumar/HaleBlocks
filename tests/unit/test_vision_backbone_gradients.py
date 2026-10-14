"""Every vision backbone family: forward contract holds and *every* parameter is trained.

A parameter without a gradient is either dead weight (and breaks DistributedDataParallel) or a
module whose output is silently detached - both were real bugs (InternImage's unused stem,
SwinV2's cached position bias).
"""

import pytest
import torch

from hale_core.nn.vision import build_vision_backbone

# The smallest variant of every family.
SMALLEST = [
    "alexnet",
    "vgg11",
    "resnet18",
    "resnet50_dcn",
    "resnext50_32x4d",
    "se_resnet50",
    "resnest50",
    "densenet121",
    "inception_v1",
    "inception_v3",
    "mobilenet_v1_025",
    "mobile_one_s0",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "regnetx_002",
    "regnety_002",
    "nfnet_f0",
    "convnext_v1_tiny",
    "convnext_tiny",
    "internimage_tiny",
    "vit_tiny",
    "deit3_small",
    "dino_vit_small",
    "eva_02_small",
    "swinv2_tiny",
    "coatnet_0",
    "maxvit_tiny",
]


def _train_step(model: torch.nn.Module, size: int) -> None:
    model.train().zero_grad(set_to_none=True)
    out = model(torch.randn(2, 3, size, size))
    loss = out.pooled_output.square().mean() + sum(f.square().mean() for f in out.feature_maps)
    loss.backward()


@pytest.mark.parametrize("name", SMALLEST)
def test_every_parameter_receives_a_gradient(name: str) -> None:
    torch.manual_seed(0)
    model = build_vision_backbone(name, img_size=224)
    _train_step(model, 224)
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"{name}: parameters without gradient: {dead[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_swinv2_survives_eval_then_two_training_steps() -> None:
    """The bias used to be cached: eval first -> CPB never trained; train twice -> crash."""
    model = build_vision_backbone("swinv2_tiny", img_size=224)
    with torch.no_grad():
        model.eval()(torch.randn(1, 3, 224, 224))
    _train_step(model, 224)
    _train_step(model, 224)  # used to raise "Trying to backward through the graph a second time"
    cpb = [p for n, p in model.named_parameters() if "cpb_mlp" in n]
    assert cpb and all(p.grad is not None and p.grad.abs().sum() > 0 for p in cpb)
