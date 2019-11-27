import torch.nn as nn
from normalization import _ConditionalInstanceNorm
class _ConvInstanceReLu(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride=2, labels=1, activation='relu'):
        super(_ConvInstanceReLu, self).__init__()
        padding = kernel_size // 2
        self.reflection = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(inch, outch, kernel_size, stride, padding=0)
        self.norm = _ConditionalInstanceNorm(outch, labels)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.02)

    def forward(self, x, label):
        x = self.reflection(x)
        x = self.conv(x)
        x = self.norm(x, label)
        x = self.act(x)
        return x


class _ResidualBlock(nn.Module):
    def __init__(self, inch, outch, kernel_size, labels, activation='relu'):
        super(_ResidualBlock, self).__init__()
        self.conv1 = _ConvInstanceReLu(inch, outch, kernel_size, stride=1, labels=labels)
        self.conv2 = _ConvInstanceReLu(outch, inch, kernel_size, stride=1, labels=labels)

    def forward(self, x, label):
        h1 = self.conv1(x, label)
        h2 = self.conv2(h1, label)
        return x + h2
        
class _UpsamplingConv(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, labels, activation='relu'):
        super(_UpsamplingConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        self.conv = _ConvInstanceReLu(inch, outch, kernel_size, stride=1, labels=labels)

    def forward(self, x, label):
        h1 = self.upsample(x)
        h2 = self.conv(h1, label)
        return h2

if __name__ == "__main__":
    conv = _ConvInstanceReLu(3,4,3)
    print(conv)
    res = _ResidualBlock(3,4,3,5)
    print(res)
    upsample = _UpsamplingConv(3,4,3,2,5)
    print(upsample)