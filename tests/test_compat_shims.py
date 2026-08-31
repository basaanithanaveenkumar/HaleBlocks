def test_legacy_component_shim_exports_build_attn_mask():
    from hale_core.components import build_attn_mask
    from hale_core.nn.layers import build_attn_mask as new_build_attn_mask

    assert build_attn_mask is new_build_attn_mask


def test_legacy_backbone_shim_exports_build_backbone():
    from hale_core.backbones import build_backbone
    from hale_core.nn.backbones import build_backbone as new_build_backbone

    assert build_backbone is new_build_backbone
