import os
import torch
import torch.nn as nn
from layers import _ConvInstanceReLu, _ResidualBlock, _UpsamplingConv

class TransformNetwork(nn.Module):
    def __init__(self, inch, labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TransformNetwork, self).__init__()
        ngf = 32
        self.conv1 = _ConvInstanceReLu(inch, ngf, 9, 1, labels)
        self.conv2 = _ConvInstanceReLu(ngf, ngf*2, 3, 2, labels)
        self.conv3 = _ConvInstanceReLu(ngf*2, ngf*4, 3, 2, labels)
        self.residual1 = _ResidualBlock(ngf*4, ngf*8, 3, labels)
        self.residual2 = _ResidualBlock(ngf*4, ngf*8, 3, labels)
        self.residual3 = _ResidualBlock(ngf*4, ngf*8, 3, labels)
        self.residual4 = _ResidualBlock(ngf*4, ngf*8, 3, labels)
        self.residual5 = _ResidualBlock(ngf*4, ngf*8, 3, labels)
        self.upsampling1 = _UpsamplingConv(ngf*4, ngf*2, 3, 2, labels)
        self.upsampling2 = _UpsamplingConv(ngf*2, ngf, 3, 2, labels)
        self.upsampling3 = _UpsamplingConv(ngf, inch, 9, 1, labels)

    def forward(self, x, label):
        x = self.conv1(x, label)
        x = self.conv2(x, label)
        x = self.conv3(x, label)
        x = self.residual1(x, label)
        x = self.residual2(x, label)
        x = self.residual3(x, label)
        x = self.residual4(x, label)
        x = self.residual5(x, label)
        x = self.upsampling1(x, label)
        x = self.upsampling2(x, label)
        x = self.upsampling3(x, label)
        return torch.tanh(x)

    def load_model(self, path, attrib='model_state_dict'):
        if not os.path.exists(path):
            raise Exception('The path does not exists')

        cp = torch.load(path, map_location=self.device)
        self.load_state_dict(cp[attrib])

if __name__ == "__main__":
    t = TransformNetwork(3, 6)
    t.load_model('trained_model/transformer-v1_2.pt')
    print("Model Loaded")