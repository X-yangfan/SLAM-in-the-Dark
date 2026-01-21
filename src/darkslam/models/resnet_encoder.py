from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


def _build_resnet(num_layers: int, *, pretrained: bool) -> nn.Module:
    if hasattr(tv_models, f"ResNet{num_layers}_Weights"):
        weights_enum = getattr(tv_models, f"ResNet{num_layers}_Weights")
        weights = weights_enum.DEFAULT if pretrained else None
        return getattr(tv_models, f"resnet{num_layers}")(weights=weights)
    return getattr(tv_models, f"resnet{num_layers}")(pretrained=pretrained)


class ResNetMultiImageInput(tv_models.ResNet):
    def __init__(self, block, layers, *, num_input_images: int = 1) -> None:
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def resnet_multiimage_input(num_layers: int, *, pretrained: bool = False, num_input_images: int = 1) -> nn.Module:
    if num_layers not in [18, 50]:
        raise ValueError("Only resnet18/resnet50 are supported for multi-image input.")

    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: tv_models.resnet.BasicBlock, 50: tv_models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        base = _build_resnet(num_layers, pretrained=True)
        state_dict = base.state_dict()
        conv1 = state_dict["conv1.weight"]
        state_dict["conv1.weight"] = torch.cat([conv1] * num_input_images, dim=1) / float(num_input_images)
        model.load_state_dict(state_dict, strict=False)

    return model


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers: int, *, pretrained: bool, num_input_images: int = 1) -> None:
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers not in [18, 34, 50, 101, 152]:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained=pretrained, num_input_images=num_input_images)
        else:
            self.encoder = _build_resnet(num_layers, pretrained=pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []

        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)

        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features.append(x)

        x = self.encoder.layer2(x)
        features.append(x)

        x = self.encoder.layer3(x)
        features.append(x)

        x = self.encoder.layer4(x)
        features.append(x)

        return features

