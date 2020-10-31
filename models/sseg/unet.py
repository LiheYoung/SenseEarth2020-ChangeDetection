from models.block.conv import conv3x3
from models.block.attention import SCSEModule
from models.sseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class UNet(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(UNet, self).__init__(backbone, pretrained)

        n_channels = self.backbone.channels

        self.decoder1 = DecoderBlock(n_channels[-1] + n_channels[-2], n_channels[-2], lightweight)
        self.decoder2 = DecoderBlock(n_channels[-2] + n_channels[-3], n_channels[-3], lightweight)
        self.decoder3 = DecoderBlock(n_channels[-3] + n_channels[-4], n_channels[-4], lightweight)
        self.decoder4 = DecoderBlock(n_channels[-4] + n_channels[-5], n_channels[-4] // 2, lightweight)
        self.decoder5 = DecoderBlock(n_channels[-4] // 2, n_channels[-4] // 4, lightweight)

        self.classify = nn.Conv2d(n_channels[-4] // 4, nclass, 3, padding=1, bias=True)

        self.decoder1_bin = DecoderBlock(n_channels[-1] + n_channels[-2], n_channels[-2], lightweight)
        self.decoder2_bin = DecoderBlock(n_channels[-2] + n_channels[-3], n_channels[-3], lightweight)
        self.decoder3_bin = DecoderBlock(n_channels[-3] + n_channels[-4], n_channels[-4], lightweight)
        self.decoder4_bin = DecoderBlock(n_channels[-4] + n_channels[-5], n_channels[-4] // 2, lightweight)
        self.decoder5_bin = DecoderBlock(n_channels[-4] // 2, n_channels[-4] // 4, lightweight)

        self.head_bin = nn.Conv2d(n_channels[-4] // 4, 1, 3, padding=1, bias=True)

    def base_forward(self, x1, x2):
        features1 = self.backbone.base_forward(x1)
        features2 = self.backbone.base_forward(x2)

        out1 = self.decoder1(features1[-1], features1[-2])
        out1 = self.decoder2(out1, features1[-3])
        out1 = self.decoder3(out1, features1[-4])
        out1 = self.decoder4(out1, features1[-5])
        out1 = self.decoder5(out1)

        out1 = self.classify(out1)

        out2 = self.decoder1(features2[-1], features2[-2])
        out2 = self.decoder2(out2, features2[-3])
        out2 = self.decoder3(out2, features2[-4])
        out2 = self.decoder4(out2, features2[-5])
        out2 = self.decoder5(out2)

        out2 = self.classify(out2)

        out_bin = self.decoder1_bin(torch.abs(features1[-1] - features2[-1]), torch.abs(features1[-2] - features2[-2]))
        out_bin = self.decoder2_bin(out_bin, torch.abs(features1[-3] - features2[-3]))
        out_bin = self.decoder3_bin(out_bin, torch.abs(features1[-4] - features2[-4]))
        out_bin = self.decoder4_bin(out_bin, torch.abs(features1[-5] - features2[-5]))
        out_bin = self.decoder5_bin(out_bin)

        out_bin = self.head_bin(out_bin)
        out_bin = torch.sigmoid(out_bin)

        return out1, out2, out_bin.squeeze(1)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Sequential(conv3x3(in_channels, out_channels, lightweight),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(conv3x3(out_channels, out_channels, lightweight),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.attention1 = SCSEModule(in_channels)
        self.attention2 = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
