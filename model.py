from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
                                   stride=(1, 1), padding=(0, 0 if kernel_size[0]>kernel_size[1] else max(kernel_size)//2), groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier, out_channels, kernel_size=(1, 1),
                                   stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64,
                 F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False),
            nn.BatchNorm2d(F1),
            DepthwiseSeparableConv2d(F1, F1, kernel_size=(Chans, 1), depth_multiplier=D),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropoutType(dropoutRate)
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv2d(F1, F2, kernel_size=(1, 16), depth_multiplier=1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            self.dropoutType(dropoutRate)
        )

        self.block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2*int(np.floor((np.floor((Samples+1)/4)+1)/8)), nb_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        return x



