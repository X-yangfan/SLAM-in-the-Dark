from __future__ import annotations

import torch
from torch import Tensor


class FeatureEncoder:
    """A lightweight image encoder for loop-closure retrieval.

    Notes:
    - Uses torchvision backbones when available.
    - Exposes a single global feature vector per image (N x D).
    """

    def __init__(self, device: torch.device, *, backbone: str = "mobilenet_v3_small") -> None:
        self.device = device
        self.backbone = backbone

        try:
            from torchvision import models, transforms
            from torchvision.models.feature_extraction import create_feature_extractor
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "FeatureEncoder requires torchvision. Install it via `pip install torchvision`."
            ) from e

        if backbone == "mobilenet_v3_small":
            if hasattr(models, "MobileNet_V3_Small_Weights"):
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                model = models.mobilenet_v3_small(weights=weights)
            else:
                model = models.mobilenet_v3_small(pretrained=True)
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            model = create_feature_extractor(model, return_nodes=["flatten"])
            self.num_features = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.model = model.to(self.device).eval()

    def __call__(self, image: Tensor) -> Tensor:
        image = self.normalize(image)
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)["flatten"]
        return features

