from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import ResnetEncoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.conv(x))


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, use_refl: bool = True) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


def upsample(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc: np.ndarray,
        *,
        scales=range(4),
        num_output_channels: int = 1,
        use_skips: bool = True,
    ) -> None:
        super().__init__()

        self.alpha = 10.0
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = list(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs: OrderedDict = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = int(self.num_ch_enc[-1]) if i == 4 else int(self.num_ch_dec[i + 1])
            num_ch_out = int(self.num_ch_dec[i])
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = int(self.num_ch_dec[i])
            if self.use_skips and i > 0:
                num_ch_in += int(self.num_ch_enc[i - 1])
            num_ch_out = int(self.num_ch_dec[i])
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(int(self.num_ch_dec[s]), self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                disp = self.convs[("dispconv", i)](x)
                disp = self.alpha * self.sigmoid(disp) + self.beta
                outputs.append(disp)

        return outputs[::-1]


class DispResNet(nn.Module):
    def __init__(self, num_layers: int = 18, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ResnetEncoder(num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        outputs = self.decoder(features)
        if self.training:
            return outputs
        return outputs[0]

