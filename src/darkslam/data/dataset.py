from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_DEFAULT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def _iter_images(root: Path, exts: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for ext in exts:
        paths.extend(root.glob(f"*{ext}"))
        paths.extend(root.glob(f"*{ext.upper()}"))
    # stable + unique
    return sorted({p.resolve() for p in paths})


def _load_image(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_depth(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    depth = np.load(path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D array, got shape={depth.shape} from {path}")
    tensor = torch.from_numpy(depth)[None, None, ...]  # 1x1xHxW
    tensor = torch.nn.functional.interpolate(tensor, size=image_size, mode="nearest")
    return tensor[0]  # 1xHxW


class ImageFolderDataset(Dataset[dict]):
    def __init__(
        self,
        data_root: str | Path,
        *,
        depth_root: str | Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        image_exts: Sequence[str] = _DEFAULT_IMAGE_EXTS,
        depth_ext: str = ".npy",
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"data_root must be a directory: {self.data_root}")

        self.depth_root = None if depth_root is None else Path(depth_root).expanduser().resolve()
        if self.depth_root is not None and not self.depth_root.is_dir():
            raise NotADirectoryError(f"depth_root must be a directory: {self.depth_root}")

        self.image_size = image_size
        self.image_paths = _iter_images(self.data_root, image_exts)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_root} (exts={tuple(image_exts)})")

        self.depth_ext = depth_ext
        if self.depth_root is not None:
            missing: list[Path] = []
            for img in self.image_paths[: min(10, len(self.image_paths))]:
                expected = (self.depth_root / img.stem).with_suffix(self.depth_ext)
                if not expected.exists():
                    missing.append(expected)
            if missing:
                hint = "\n".join(str(p) for p in missing[:5])
                raise FileNotFoundError(
                    "depth_root is set but some depth files are missing.\n"
                    f"Expected files like: {self.depth_root}/<image_stem>{self.depth_ext}\n"
                    f"Examples:\n{hint}"
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = _load_image(img_path, self.image_size)

        if self.depth_root is None:
            depth = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        else:
            depth_path = (self.depth_root / img_path.stem).with_suffix(self.depth_ext)
            depth = _load_depth(depth_path, self.image_size)

        return {"image": image, "depth": depth, "path": str(img_path)}
