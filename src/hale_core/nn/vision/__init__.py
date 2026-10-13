"""
hale_core.nn.vision — production vision backbone library
=========================================================

All backbones inherit BaseVisionBackbone and return BackboneOutput, which
carries pooled embeddings (for VLM/action decoders) AND multi-scale feature
maps (for FPN/BEV/depth heads).

Quick start::

    from hale_core.nn.vision import build_vision_backbone, list_vision_backbones

    # See all registered backbones
    print(list_vision_backbones())

    # Build a backbone
    backbone = build_vision_backbone("convnext_base", img_size=224)
    out = backbone(images)               # images: [B, 3, H, W]

    out.pooled_output                    # [B, 1024]  — global embedding
    out.feature_maps                     # list of multi-scale tensors for FPN
    out.patch_tokens                     # [B, N, D]  — ViT patch tokens (or None)

Families
--------
  CNN (pure convolution)
    resnet18/34/50/101/152/200
    resnet50_dcn/resnet101_dcn          — ResNet + Deformable Conv v2
    convnext_tiny/small/base/large/xlarge/huge (V2 with FCMAE init)
    mobile_one_s0/s1/s2/s3/s4          — re-parameterisable efficient backbone
    efficientnet_v2_s/m/l/xl
    internimage_tiny/small/base/large/huge  — DCNv3-based

  ViT (pure transformer)
    vit_tiny/small/base/large/huge      — standard ViT (Dosovitskiy 2021)
    vit_22b                             — ViT-22B recipe
    dino_vit_base/large/giant           — DINOv2 with register tokens
    eva_02_small/base/large             — EVA-02 with ROPE + SwiGLU
    deit3_small/base/large/huge         — DeiT III (LayerScale + strong aug recipe)

  Swin (shifted-window transformer)
    swinv2_tiny/small/base/large/giant  — SwinV2 with log-space RPB

  Hybrid (conv + attention)
    coatnet_0/1/2/3/7                   — CoAtNet (conv stem + rel-attn)
    maxvit_tiny/small/base/large/xlarge — MaxViT (block-grid attention)
"""

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import (
    build_vision_backbone,
    get_vision_backbone_cls,
    list_vision_backbones,
    list_vision_backbones_by_family,
    register_vision_backbone,
)

__all__ = [
    "BaseVisionBackbone",
    "BackboneOutput",
    "register_vision_backbone",
    "build_vision_backbone",
    "get_vision_backbone_cls",
    "list_vision_backbones",
    "list_vision_backbones_by_family",
]
