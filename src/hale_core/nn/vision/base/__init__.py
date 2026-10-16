"""
BaseVisionBackbone — abstract base class for all vision encoders.

Design contract
---------------
Every vision backbone in hale_core.nn.vision must:

  1. Inherit from BaseVisionBackbone.
  2. Set the class-level attributes: arch_name, embed_dim, feature_dims.
  3. Implement forward(x: Tensor) → BackboneOutput.
  4. Implement num_parameters() (provided by base via count_params helper).
  5. Register itself via @register_vision_backbone("name").

BackboneOutput is a dataclass that carries BOTH the final pooled embedding
(for classification) and multi-scale feature maps (for dense tasks like
segmentation, detection, BEV lifting).  Dense task heads (FPN, BEV encoder)
read feature_maps; global embedding heads (VLM projector, action encoder)
read pooled_output.

Feature map conventions
------------------------
feature_maps is a list ordered from FINEST (highest resolution, smallest
channel count) to COARSEST (lowest resolution, most channels).  This matches
FPN's expected input order:

    feature_maps[0]  →  stride-4  features  (finest)
    feature_maps[1]  →  stride-8  features
    feature_maps[2]  →  stride-16 features
    feature_maps[3]  →  stride-32 features  (coarsest)

Not all backbones produce all four scales — ViT-based models produce a single
scale from their patch grid (typically stride-16 or stride-14).

Usage
-----
    from hale_core.nn.vision import build_vision_backbone

    backbone = build_vision_backbone("convnext_base", img_size=224, pretrained=False)
    out: BackboneOutput = backbone(images)      # images: [B, 3, H, W]

    # Dense tasks (FPN, BEV encoder):
    fpn_features = out.feature_maps             # list of [B, C_i, H_i, W_i]

    # Global embedding tasks (VLM, action encoder):
    embedding = out.pooled_output               # [B, embed_dim]

    # Patch-level tokens (VLM cross-attention):
    tokens = out.patch_tokens                   # [B, N, embed_dim] or None
"""

from hale_core.nn.vision.base.backbone_output import BackboneOutput
from hale_core.nn.vision.base.base_vision_backbone import BaseVisionBackbone

__all__ = ["BackboneOutput", "BaseVisionBackbone"]
