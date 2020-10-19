from models.block.conv import conv3x3
from models.sseg.base import BaseNet
from models.sseg.fcn import FCNHead

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(DeepLabV3Plus, self).__init__(backbone, pretrained)

        low_level_channels = self.backbone.channels[1]
        high_level_channels = self.backbone.channels[-1]

        self.head = FCNHead(high_level_channels, nclass, lightweight)

        self.reduce_bin = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU(True))

        self.fuse_bin = nn.Sequential(conv3x3(high_level_channels // 8 + 48, 256, lightweight),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(True),

                                      conv3x3(256, 256, lightweight),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(True),
                                      nn.Dropout(0.1, False))

        self.head_bin = ASPPModule(high_level_channels, [12, 24, 36], lightweight)

        self.classifier_bin = nn.Conv2d(256, 1, 1, bias=True)

    def base_forward(self, x1, x2):
        h, w = x1.shape[-2:]

        _, c1_1, _, _, c1_4 = self.backbone.base_forward(x1)
        _, c2_1, _, _, c2_4 = self.backbone.base_forward(x2)

        diff_4 = torch.abs(c1_4 - c2_4)
        diff_4 = self.head_bin(diff_4)
        diff_4 = F.interpolate(diff_4, size=c1_1.shape[-2:], mode="bilinear", align_corners=False)
        diff_1 = torch.abs(c1_1 - c2_1)
        diff_1 = self.reduce_bin(diff_1)
        out_bin = torch.cat([diff_4, diff_1], dim=1)
        out_bin = self.fuse_bin(out_bin)
        out_bin = self.classifier_bin(out_bin)
        out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
        out_bin = torch.sigmoid(out_bin)

        out1 = self.head(c1_4)
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=False)

        out2 = self.head(c2_4)
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear", align_corners=False)

        return out1, out2, out_bin.squeeze(1)


def ASPPConv(in_channels, out_channels, atrous_rate, lightweight):
    block = nn.Sequential(conv3x3(in_channels, out_channels, lightweight, atrous_rate),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=False)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, lightweight):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, lightweight)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, lightweight)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, lightweight)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
