from typing import List, Tuple

import torch
import torchvision
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.mobilenetv2 import InvertedResidual


def _analyze_layer(layer):
    """
    Analyzes a given layer to determine the number of input filters (channels)
    and the effective stride of the layer

    Args:
        layer: a model layer of type InvertedResidual, Conv2dNormActivation,
               Conv2d, or similar. It can also be a composite layer like
               nn.Sequential, MBConv, or FusedMBConv
    Returns:
        Tuple[int, int]: A tuple containing:
            - the number of input filters (channels);
            - the stride for the layer.
    """
    if isinstance(layer, torchvision.ops.Conv2dNormActivation):
        return layer[0].weight.shape[1], layer[0].stride[0]
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.shape[1], layer.stride[0]
    if isinstance(layer, InvertedResidual):
        input_filters = layer.conv[0][0].weight.shape[1]
        stride = 1
        for nestedlayer in layer.conv:
            _, nested_stride = _analyze_layer(nestedlayer)
            stride = max(stride, nested_stride)
        return input_filters, stride
    is_sequential = isinstance(layer, torch.nn.Sequential)
    if is_sequential or isinstance(layer, MBConv) or isinstance(layer, FusedMBConv):
        blocks = layer if is_sequential else layer.block
        input_filters, stride = _analyze_layer(blocks[0])
        for layer in blocks[1:]:
            _, layer_stride = _analyze_layer(layer)
            stride = max(stride, layer_stride)
        return input_filters, stride
    else:
        return 0, 0


def get_stride_features_and_filters(model) -> Tuple[List[int], List[int]]:
    """
    Identifies the indices of strided layers in the model and retrieves
    the number of input channels for those layers.

    The function processes the features of a model (assumed to be in a
    sequential structure) and detects layers with a stride of 2. Additionally,
    it includes the very last feature layer for context.

    Layers example:
    https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

    Args:
        model (nn.Module): torchvision model
    Returns:
        Tuple[List[int], List[int]]: A tuple containing:
            - a list of indices of strided layers;
            - a list of the corresponding input channels (filters) for
              each of these layers.
    """

    output_number = []
    nfilters = []
    for n, feature in enumerate(model.features):
        input_channels, stride = _analyze_layer(feature)
        if stride == 2:
            output_number.append(n)
            nfilters.append(input_channels)
    # assume that the very last feature layer (channel expansion to 1280)
    # actually creates features for classification and is redundant
    output_number.append(len(model.features) - 1)
    nfilters.append(_analyze_layer(model.features[-1])[0])
    return output_number[1:], nfilters[1:]  # ignore stride 1 data
