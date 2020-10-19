from torch import nn


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=atrous_rate, groups=in_channels,
                      dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


def conv3x3(in_channels, out_channels, lightweight, atrous_rate=1):
    if lightweight:
        return DSConv(in_channels, out_channels, atrous_rate)
    else:
        return nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False)
