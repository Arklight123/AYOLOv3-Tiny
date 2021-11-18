from utils.utils import *


class Conv3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3x3Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.leaky(self.bn(self.conv(x)))
        return out


class MaxpoolConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(MaxpoolConvBlock, self).__init__()
