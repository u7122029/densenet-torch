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
        x = self.conv1(x)
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
        self.dc1 = DenseConv(in_channels, out_channels, 1, bias=False)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.dc1(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, layers, growth_rate, out_classes, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate

        n_channels = 2 * growth_rate
        self.dc = DenseConv(3, n_channels, 3, padding=1)

        self.dense_layer1 = self.make_db_layers(layers[0], n_channels)
        n_channels += layers[0]*growth_rate
        out_channels = int(reduction * n_channels)
        self.trans1 = TransitionLayer(n_channels, out_channels)
        n_channels = out_channels

        self.dense_layer2 = self.make_db_layers(layers[1], n_channels)
        n_channels += layers[1] * growth_rate
        out_channels = int(reduction * n_channels)
        self.trans2 = TransitionLayer(n_channels, out_channels)
        n_channels = out_channels

        self.dense_layer3 = self.make_db_layers(layers[2], n_channels)
        n_channels += layers[2] * growth_rate
        out_channels = int(reduction * n_channels)
        self.trans3 = TransitionLayer(n_channels, out_channels)
        n_channels = out_channels

        self.dense_layer4 = self.make_db_layers(layers[3], n_channels)
        n_channels += layers[3] * growth_rate

        self.bn = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pooling = nn.AvgPool2d(4)
        self.linear = nn.Linear(n_channels, out_classes)

    def make_db_layers(self, n_layers, in_channels):
        layers = []
        for i in range(n_layers):
            layers.append(DenseBlock(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dc(x)
        out = self.trans1(self.dense_layer1(out))
        out = self.trans2(self.dense_layer2(out))
        out = self.trans3(self.dense_layer3(out))
        out = self.dense_layer4(out)
        out = self.bn(out)
        out = self.relu1(out)
        out = self.avg_pooling(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    model = DenseNet((6, 12, 24, 16), 32, 10)
    inp = torch.rand(16, 3, 32, 32)
    print(model(inp).shape)