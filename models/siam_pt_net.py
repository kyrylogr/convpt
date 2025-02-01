import torch
import torch.nn as nn
from backbones import create_backbone
from blocks import CorrelationHead
from typing import List

# Single point tracker network
# problem statement:
#   Given a template and a search patches find template center in a search area.
# approach 'siamese tracker' with two heads for class position prediction and offsets regression.


class SiamPTLoss(nn.Module):
    """Loss function for single point tracker."""

    def __init__(self, lambda_cls=1.0, lambda_offset=1.0):
        self.lambda_cls = lambda_cls
        self.lambda_offset = lambda_offset
        self.delta = 1e-5

    def focal_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        :param y_true: encoded GT (with value==1. indicating true location)
        :param y_pred: predicted class (probability) value
        :return: class prediction error =
                  sum[ log(y_pred) * (1-y_pred)^2 ] if y_true == 1
                  sum[ log(1 - y_pred) * y_pred^2 * (1-y_true)^4 ] if y_true == 1
        """
        pos_inds = y_true.eq(1.0).float()
        pos_count = pos_inds.sum()
        neg_inds = 1.0 - pos_inds
        neg_weights = torch.pow(1.0 - y_true, 4)

        y_pred = torch.clamp(y_pred, self.delta, 1.0 - self.delta)
        pos_loss = torch.log(y_pred) * torch.pow(1.0 - y_pred, 2) * pos_inds
        neg_loss = torch.log(1 - y_pred) * torch.pow(y_pred, 2) * neg_weights * neg_inds
        return -(pos_loss.sum() + neg_loss.sum()) / torch.maximum(
            pos_count, torch.oneslike(pos_count)
        )

    def l1_loss(self, mask, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = (torch.abs(y_true - y_pred) * mask).sum()
        num_el = torch.maximum(mask.sum(), torch.ones_like(mask))
        return loss / num_el

    def forward(self, gt_cls_offset, pred_cls, pred_offset):
        gt_cls = gt_cls_offset[:, 0, :, :]
        gt_offset = gt_cls_offset[:, 1:, :, :]
        focal_l = self.focal_loss(gt_cls, pred_cls)
        offset_l = self.l1_loss(gt_cls.gt(0.0), gt_offset, pred_offset)
        return focal_l * self.lambda_cls + offset_l * self.lambda_offset


class SiamPTNet(nn.Module):
    def __init__(
        self,
        result_stride: int = 16,
        head_channels: int = 256,
        corr_channels: int = 64,
        tail_blocks: int = 3,
        backbone: str = "efficientnet_b0",
        backbone_weights: str = "DEFAULT",
        pre_encoder: int = True
    ):
        super().__init__()
        self.backbone = create_backbone(backbone, backbone_weights)
        backbone_output_block_no = [2, 4, 8, 16, 32].index(result_stride)
        assert backbone_output_block_no >= 0
        backbone_result_channels = self.backbone.filters_count[backbone_output_block_no]
        self.adjust_backbone_channels = nn.Sequential(
            [
                nn.Conv2d(backbone_result_channels, head_channels, stride=1),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(),
            ]
        )

        self.head_class = CorrelationHead(
            head_channels,
            pre_encoder,
            corr_channels,
            tail_blocks=tail_blocks,
            result_channels=1,
        )
        self.head_offset = CorrelationHead(
            head_channels,
            pre_encoder,
            corr_channels,
            tail_blocks=tail_blocks,
            result_channels=2,
        )
        self.loss = SiamPTLoss()

    def forward(self, z: torch.Tensor, x: torch.Tensor, gt: List[torch.Tensor] = None):
        z = self.backbone(z)
        x = self.backbone(x)
        z = self.adjust_backbone_channels(z)
        x = self.adjust_backbone_channels(x)
        cls = self.head_class(z, x)
        cls = torch.sigmoid(cls)
        offsets = self.head_offset(z, x)
        if gt is None:
            return cls, offsets
        else:
            return self.loss(gt, cls, offsets)
