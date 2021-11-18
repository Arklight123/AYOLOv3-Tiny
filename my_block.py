from utils.utils import *
from utils.deform_conv_v2 import DeformConv2d
import torch.nn.functional as F


class MaxPoolingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels
        mid_channels = in_channels // 2
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return out


def split(x, groups):
    out = x.chunk(groups, dim=1)

    return out


def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out


class ShuffleAddChannelUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shuffle(out, 2)
        return out


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        mid_channels = out_channels // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = split(x, 2)
            out = torch.cat((self.branch1(x1), self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, layers_num):
        super().__init__()
        self.stage = self.make_layers(in_channels, out_channels, layers_num, 2)

    def make_layers(self, in_channels, out_channels, layers_num, stride):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, stride))
        in_channels = out_channels

        for i in range(layers_num - 1):
            ShuffleUnit(in_channels, out_channels, 1)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stage(x)
        return out


class Separable(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.f1(x)
        out = self.f2(out)
        return out


class DeformUnit(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()

        # self.relu = nn.ReLU(inplace=True)

        features = []
        features.append(DeformConv2d(inplanes, outplanes, 3, padding=1, bias=False, modulation=True))
        features.append(nn.BatchNorm2d(outplanes))
        features.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SpatialAttentionMaxpool(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttentionMaxpool, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.sa = SpatialAttention(kernel_size)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.pool2 = nn.MaxPool2d(2, 1)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.sa(x1)
        out = x1 * x2.expand_as(x1)
        out = self.pool2(self.pad(out))
        return out


class SpatialAttentionMaxpoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialAttentionMaxpoolConv, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.sa = SpatialAttention(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.leaky(self.bn(self.conv(x1)))
        sa_out = self.sa(x1)
        out = x2 * sa_out.expand_as(x2)
        return out


class SpatialAttention2MaxpoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialAttention2MaxpoolConv, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.pool2 = nn.MaxPool2d(2, 1)
        self.sa = SpatialAttention(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.pool1(x)
        x1 = self.pool2(self.pad(x1))
        x2 = self.leaky(self.bn(self.conv(x1)))
        sa_out = self.sa(x1)
        out = x2 * sa_out.expand_as(x2)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 512, 26, 26)
    # net = MaxPoolingBlock(in_channels=64)
    # y = net(input)
    # print(y.size())
    #
    # net = ShuffleAddChannelUnit(in_channels=64)
    # y = net(input)
    # print(y.size())
    # net = ShuffleNetV2(64, 128, 5)
    # y = net(input)
    # print(y.size())
    net = DeformUnit(512, 1024)
    y = net(input)
    print(y.size())
