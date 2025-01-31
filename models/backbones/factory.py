from typing import TypeVar

import torchvision.models as models

from .abstract_backbone import AbstractBackbone, TorchvisionBackbone

BACKBONE_BUILDER_CONF = {
    "efficientnet": TorchvisionBackbone,
    "mobilenet_v2": TorchvisionBackbone,
}

BackboneType = TypeVar("BackboneType", bound=AbstractBackbone)


def create_backbone(
    backbone_name: str = "default", weights: str = None
) -> BackboneType:
    """Create backbone.
    Args:
        backbone_name (str): name of the backbone,
        weights (str): name of pretrained weights for torchvision pretrained models ('default' will work fine).
    Returns:
        BackboneType: backbone model which implements AbstractBackbone class
    """
    backbone_name_parsed = backbone_name.lower()
    backbone_class = None

    for name, conf_class in BACKBONE_BUILDER_CONF.items():
        if backbone_name_parsed.startswith(name):
            backbone_class = conf_class

    if backbone_class is None:
        raise ValueError(f"Backbone '{backbone_name}' is not supported yet")

    model = models.get_model(backbone_name_parsed, weights=weights)
    return backbone_class(model)
