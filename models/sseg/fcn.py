from models.block.conv import conv3x3
from models.sseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class FCN(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(FCN, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1]

        self.head = FCNHead(in_channels, nclass, lightweight)
        self.head_bin = FCNHead(in_channels, 1, lightweight)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1, bias=True))

    def forward(self, x):
        return self.head(x)
