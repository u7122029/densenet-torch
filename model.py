import torch
from torch import nn

class DenseConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 padding: str | int | tuple[int, int] = 0,
                 bias: bool = True,
                 stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               bias=bias,
                               padding=padding,
                               stride=stride)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = 4 * in_channels
        self.dc1 = DenseConv(in_channels, inter_channels, 1, bias=False)
        self.dc2 = DenseConv(inter_channels, out_channels, 3, 1, bias=False)

    def forward(self, x):
        out = self.dc1(x)
        out = self.dc2(out)
        return torch.cat([x, out], dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dc1 = DenseConv(in_channels, out_channels, 1, 1, bias=False)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.dc1(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, depth, growth_rate):
        super().__init__()
        self.growth_rate = growth_rate

        n_channels = 2 * growth_rate
        self.dc = DenseConv(3, n_channels, 3, padding=1)

        n_channels += growth_rate
        self.db1 = DenseBlock(n_channels, )
