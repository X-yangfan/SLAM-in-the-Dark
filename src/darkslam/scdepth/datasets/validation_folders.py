from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def _crawl_folders(folders_list: list[Path], *, dataset: str = "nyu"):
    imgs: list[Path] = []
    depths: list[Path] = []
    for folder in folders_list:
        current_imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
        if dataset == "nyu":
            current_depth = sorted((folder / "depth").glob("*.png"))
        elif dataset == "kitti":
            current_depth = sorted(folder.glob("*.npy"))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        imgs.extend(current_imgs)
        depths.extend(current_depth)
    return imgs, depths


class ValidationSet(data.Dataset):
    def __init__(self, root, transform=None, dataset="nyu"):
        self.root = Path(root)
        scene_list_path = self.root / "val.txt"
        self.scenes = [self.root / line.strip() for line in scene_list_path.read_text().splitlines() if line.strip()]
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depth = _crawl_folders(self.scenes, dataset=self.dataset)

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.imgs[index]).convert("RGB"), dtype=np.float32)

        if self.dataset == "nyu":
            depth = torch.from_numpy(np.asarray(Image.open(self.depth[index]), dtype=np.float32)).float() / 5000.0
        elif self.dataset == "kitti":
            depth = torch.from_numpy(np.load(self.depth[index]).astype(np.float32))
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)

