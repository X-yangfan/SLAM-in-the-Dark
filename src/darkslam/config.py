from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    depth_root: Optional[str] = None
    image_size: Tuple[int, int] = (256, 256)  # (H, W)

    resnet_layers: int = 18
    with_pretrain: bool = True

    epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-4
    num_workers: int = 2

    save_dir: str = "./outputs"
    resume: Optional[str] = None
    seed: int = 0
    device: str = default_device()

    min_depth: float = 0.1
    max_depth: float = 100.0
    smooth_weight: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferConfig:
    image: str
    checkpoint: str
    out: str
    resnet_layers: int = 18
    image_size: Optional[Tuple[int, int]] = None  # (H, W); None -> keep original
    out_png: Optional[str] = None
    device: str = default_device()
