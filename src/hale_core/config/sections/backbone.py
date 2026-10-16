"""
BackboneConfig — config section for vision backbone selection and settings.

Usage in YAML:

    backbone:
      name: convnext_base          # registered backbone name
      img_size: 224
      in_chans: 3
      drop_rate: 0.1
      out_indices: [0, 1, 2, 3]   # which feature scales to return (null = all)
      freeze_encoder: false        # freeze all backbone weights
      freeze_up_to_stage: -1       # freeze stages 0..N (-1 = don't freeze any)
      pretrained_path: null        # path to local .pt checkpoint for weight loading

The backbone is built at model construction time:

    from hale_core.registry import build_vision_backbone

    backbone = build_vision_backbone(
        cfg.backbone.name,
        img_size=cfg.backbone.img_size,
        in_chans=cfg.backbone.in_chans,
        drop_rate=cfg.backbone.drop_rate,
        out_indices=cfg.backbone.out_indices,
    )
    if cfg.backbone.freeze_encoder:
        backbone.freeze_all()
    elif cfg.backbone.freeze_up_to_stage >= 0:
        backbone.freeze_up_to_stage(cfg.backbone.freeze_up_to_stage)
"""

from __future__ import annotations

from hale_core.config.sections.common import StrictModel


class BackboneConfig(StrictModel):
    """Vision backbone selection and configuration.

    Attributes
    ----------
    name:
        Registry name of the backbone to use, e.g. "convnext_base",
        "vit_large", "swinv2_tiny".  Must be registered in
        hale_core.nn.vision.registry.

    img_size:
        Input image spatial resolution (height == width).  Standard choices:
        224 for ImageNet-pretrained, 336 for high-res VLM, 800/1024 for detection.

    in_chans:
        Number of input channels.  Default 3 for RGB.  Use 1 for grayscale
        or 4 for RGBD.

    drop_rate:
        Dropout probability on the final pooled embedding (before the head).

    out_indices:
        Which feature pyramid stages to include in BackboneOutput.feature_maps.
        [0, 1, 2, 3] = all four scales; [2, 3] = only the two coarsest scales.
        None = all available (default).

    freeze_encoder:
        If True, freeze ALL backbone parameters (requires_grad=False).
        Use for fixed feature extraction; the backbone runs in eval mode.

    freeze_up_to_stage:
        If >= 0, freeze stages 0..freeze_up_to_stage while keeping later
        stages trainable.  -1 disables stage-wise freezing.

    pretrained_path:
        Path to a local .pt or .safetensors file containing backbone weights.
        If provided, weights are loaded after construction (before freezing).
        Keys are matched by name; missing/extra keys are logged and ignored.
    """

    name:               str             = "convnext_base"
    img_size:           int             = 224
    in_chans:           int             = 3
    drop_rate:          float           = 0.0
    out_indices:        list[int] | None = None
    freeze_encoder:     bool            = False
    freeze_up_to_stage: int             = -1
    pretrained_path:    str | None   = None

    def build(self):
        """Convenience: build and return the configured backbone.

        Returns:
            BaseVisionBackbone instance.
        """
        from hale_core.registry import build_vision_backbone
        backbone = build_vision_backbone(
            self.name,
            img_size=self.img_size,
            in_chans=self.in_chans,
            drop_rate=self.drop_rate,
            out_indices=self.out_indices,
        )
        # Load pretrained weights if provided
        if self.pretrained_path:
            _load_pretrained(backbone, self.pretrained_path)
        # Apply freezing
        if self.freeze_encoder:
            backbone.freeze_all()
        elif self.freeze_up_to_stage >= 0:
            backbone.freeze_up_to_stage(self.freeze_up_to_stage)
        return backbone


def _load_pretrained(backbone, path: str) -> None:
    """Load weights from a local checkpoint, tolerating missing/extra keys."""
    import torch
    from loguru import logger

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path)
        except ImportError:
            logger.warning("safetensors not installed; falling back to torch.load")
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        # Unwrap common checkpoint keys
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    result = backbone.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning(
            "BackboneConfig: {} missing keys when loading from {}",
            len(result.missing_keys), path,
        )
    if result.unexpected_keys:
        logger.warning(
            "BackboneConfig: {} unexpected keys when loading from {}",
            len(result.unexpected_keys), path,
        )
    logger.info("BackboneConfig: loaded weights from {}", path)
