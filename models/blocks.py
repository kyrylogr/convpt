import torch
import torch.nn as nn


class SepConv(nn.Module):
    """Separable convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    ):
        super().__init()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: torch.Tensor):
        x = self.depthwise(x)
        return self.pointwise(x)


def sepconv_bn_relu(in_channels: int, out_channels: int, kernel_size=3):
    return nn.Sequential(
        [
            SepConv(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
    )


def sepconv_bn_relu_times_n(in_channels: int, kernel_size: int, n: int):
    return nn.Sequential(
        [sepconv_bn_relu(in_channels, in_channels, kernel_size) for i in range(n)]
    )


class PointwiseCorrelation(nn.Module):
    """Pointwise correlation between two tensors.
    The result tensor has number of channels equal to spatial dimension of template(z).
    """

    def __init__(self):
        super().__init()

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        b, _, w, h = x.size()
        z = z.flatten(2).permute(0, 2, 1)
        result_channels = z.shape(1)
        x = x.flatten(2)
        corr = torch.matmul(z, x)
        assert corr.size(2) == result_channels
        return corr.reshape(b, result_channels, w, h)


class CorrelationHead(nn.Module):
    """Correlation head block.
    Does optional extra encode, pointwise correlation,
    The result tensor has number of channels equal to spatial dimension of template(z).
    """

    def __init__(
        self,
        in_channels: int,
        pre_encoder: bool = True,
        num_corr_channels: int = 64,
        post_corr_encoder: bool = True,
        tail_blocks: int = 3,
        result_channels: int = 1,
    ):
        super().__init()
        self.pre_encoder = (
            sepconv_bn_relu(in_channels, in_channels) if pre_encoder else None
        )
        self.correlation = PointwiseCorrelation()
        if post_corr_encoder:
            self.post_corr_encoder_corr = sepconv_bn_relu(
                num_corr_channels, num_corr_channels
            )
            self.post_corr_encoder_feature = sepconv_bn_relu(in_channels, in_channels)
        else:
            self.post_corr_encoder_feature, self.post_corr_encoder_corr = None, None
        self.corr_unite = sepconv_bn_relu(in_channels + num_corr_channels, in_channels)
        self.suffix = sepconv_bn_relu_times_n(in_channels, 3, tail_blocks)
        self.finish = SepConv(in_channels, result_channels, bias=True)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
        x_corr = self.correlation(z, x)
        if self.post_corr_encoder_corr is not None:
            x_corr = self.post_corr_encoder_corr(x_corr)
        if self.post_corr_encoder_feature is not None:
            x = self.post_corr_encoder_feature(x)
        x = torch.cat([x_corr, x])
        x = self.corr_unite(x)
        x = self.suffix(x)
        x = self.finish(x)
        return x
