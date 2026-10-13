"""
hale_core.nn.vision.cnn — CNN vision backbone implementations.

Importing this package triggers all @register_vision_backbone decorators,
making the following backbones available in the global vision registry:

  AlexNet (Krizhevsky et al. 2012)
    alexnet

  VGGNet (Simonyan & Zisserman 2014)
    vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

  GoogLeNet / Inception (Szegedy et al. 2014/2015)
    inception_v1, inception_v3

  ResNet (He et al. 2016)
    resnet18, resnet34, resnet50, resnet101, resnet152, resnet200

  ResNet + Deformable Convolution v2 (Zhu et al. 2019)
    resnet50_dcn, resnet101_dcn

  ResNeXt (Xie et al. 2016)
    resnext50_32x4d, resnext101_32x4d, resnext101_32x8d

  DenseNet (Huang et al. 2017)
    densenet121, densenet169, densenet201, densenet264

  MobileNetV1 (Howard et al. 2017)
    mobilenet_v1_025, mobilenet_v1_050, mobilenet_v1_075, mobilenet_v1_100

  SENet / SE-ResNet (Hu et al. 2018)
    se_resnet50, se_resnet101, se_resnet152, seresnext50_32x4d

  EfficientNet B0-B7 (Tan & Le 2019)
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

  EfficientNetV2 (Tan & Le 2021)
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, efficientnet_v2_xl

  ResNeSt (Zhang et al. 2020)
    resnest50, resnest101, resnest200, resnest269

  ConvNeXt V1 (Liu et al. 2022)
    convnext_v1_tiny, convnext_v1_small, convnext_v1_base, convnext_v1_large,
    convnext_v1_xlarge

  ConvNeXt V2 (Woo et al. 2023)
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    convnext_xlarge, convnext_huge

  MobileOne (Vasu et al. 2022)
    mobile_one_s0, mobile_one_s1, mobile_one_s2, mobile_one_s3, mobile_one_s4

  NFNet (Brock et al. 2021)
    nfnet_f0, nfnet_f1, nfnet_f2, nfnet_f3, nfnet_f4, nfnet_f5

  RegNet X/Y (Radosavovic et al. 2020)
    regnetx_002, regnetx_004, regnetx_008, regnetx_016, regnetx_032,
    regnetx_064, regnetx_080, regnetx_120, regnetx_160, regnetx_320,
    regnety_002, regnety_004, regnety_008, regnety_016, regnety_032,
    regnety_064, regnety_080, regnety_120, regnety_160, regnety_320

  InternImage (Wang et al. 2022)
    internimage_tiny, internimage_small, internimage_base,
    internimage_large, internimage_huge
"""

# Import all backbone modules so their @register_vision_backbone decorators run.
# Each import is intentional — do not remove.
from hale_core.nn.vision.cnn import alexnet        # noqa: F401
from hale_core.nn.vision.cnn import convnext       # noqa: F401
from hale_core.nn.vision.cnn import convnext_v1    # noqa: F401
from hale_core.nn.vision.cnn import densenet       # noqa: F401
from hale_core.nn.vision.cnn import efficientnet   # noqa: F401
from hale_core.nn.vision.cnn import efficientnet_b # noqa: F401
from hale_core.nn.vision.cnn import inception      # noqa: F401
from hale_core.nn.vision.cnn import internimage    # noqa: F401
from hale_core.nn.vision.cnn import mobile_one     # noqa: F401
from hale_core.nn.vision.cnn import mobilenet_v1   # noqa: F401
from hale_core.nn.vision.cnn import nfnet          # noqa: F401
from hale_core.nn.vision.cnn import regnet         # noqa: F401
from hale_core.nn.vision.cnn import resnest        # noqa: F401
from hale_core.nn.vision.cnn import resnet         # noqa: F401
from hale_core.nn.vision.cnn import resnet_dcn     # noqa: F401
from hale_core.nn.vision.cnn import resnext        # noqa: F401
from hale_core.nn.vision.cnn import senet          # noqa: F401
from hale_core.nn.vision.cnn import vgg            # noqa: F401

__all__ = [
    "alexnet",
    "convnext",
    "convnext_v1",
    "densenet",
    "efficientnet",
    "efficientnet_b",
    "inception",
    "internimage",
    "mobile_one",
    "mobilenet_v1",
    "nfnet",
    "regnet",
    "resnest",
    "resnet",
    "resnet_dcn",
    "resnext",
    "senet",
    "vgg",
]
