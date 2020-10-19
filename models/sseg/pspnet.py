from models.block.conv import conv3x3
from models.sseg.base import BaseNet
from models.sseg.fcn import FCNHead

import torch
from torch import nn
import torch.nn.functional as F


class PSPNet(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(PSPNet, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1]

        self.head = FCNHead(in_channels, nclass, lightweight)
        self.head_bin = PSPHead(in_channels, 1, lightweight)


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(PyramidPooling(in_channels),
                                   conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)
