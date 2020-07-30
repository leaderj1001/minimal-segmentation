import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet101


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.aspp = ['conv', 'conv', 'conv', 'conv', 'gap']
        self.kernel_size = [1, 3, 3, 3, 'gap']
        self.dilation = [1, 6, 12, 18, 'gap']
        self.n_module = 5

        self.layers = nn.ModuleList()
        for mode, k, d in zip(self.aspp, self.kernel_size, self.dilation):
            if mode == 'conv':
                layer = ConvBlock(in_channels, out_channels, kernel_size=k, dilation=d)
            else:
                layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    ConvBlock(in_channels, out_channels, kernel_size=1),
                )
            self.layers.append(layer)

        self.conv = ConvBlock(out_channels * self.n_module, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        outs = []
        size = x.size()[-2:]
        for idx, mode in enumerate(self.aspp):
            if mode == 'conv':
                outs.append(self.layers[idx](x))
            else:
                out = self.layers[idx](x)
                out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
                outs.append(out)
        out = torch.cat(outs, dim=1)

        out = self.dropout(self.conv(out))
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=19):
        super(DeepLabV3Plus, self).__init__()
        self.out_channels = 256

        self.backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, False, False])
        self.stem = nn.Sequential(
            *list(self.backbone.children())[:4]
        )
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4

        self.low_level_features_in_channels = 256
        self.low_level_features_out_channels = 48
        self.low_level_features_conv = ConvBlock(self.low_level_features_in_channels, self.low_level_features_out_channels, kernel_size=1)

        self.aspp = AtrousSpatialPyramidPooling(2048, self.out_channels)

        self.decoder = nn.Sequential(
            ConvBlock(self.low_level_features_out_channels + self.out_channels, self.out_channels, kernel_size=3),
            nn.Dropout(0.5),
            ConvBlock(self.out_channels, self.out_channels, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.out_channels, n_classes, kernel_size=1),
        )

    def forward(self, images):
        outs = []
        for key in images.keys():
            x = images[key]
            out = self.stem(x)
            backbone_out1 = self.block1(out)
            backbone_out2 = self.block2(backbone_out1)
            backbone_out3 = self.block3(backbone_out2)
            backbone_out4 = self.block4(backbone_out3)

            low_level_features = self.low_level_features_conv(backbone_out1)

            out = self.aspp(backbone_out4)
            out = F.interpolate(out, size=low_level_features.size()[-2:], mode='bilinear', align_corners=False)

            out = torch.cat([out, low_level_features], dim=1)
            out = self.decoder(out)
            out = F.interpolate(out, size=images['original_scale'].size()[-2:], mode='bilinear', align_corners=True)

            if 'flip' in key:
                out = torch.flip(out, dims=[-1])
            outs.append(out)
        out = torch.stack(outs, dim=-1).mean(dim=-1)

        return out
