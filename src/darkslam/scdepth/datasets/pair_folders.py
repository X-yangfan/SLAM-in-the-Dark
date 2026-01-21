import random
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


def load_as_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _glob_images(scene: Path) -> list[Path]:
    imgs = list(scene.glob("*.jpg")) + list(scene.glob("*.png")) + list(scene.glob("*.jpeg"))
    return sorted(imgs)


class PairFolder(data.Dataset):
    """Pair loader (tgt, ref) where intrinsics are per-pair."""

    def __init__(self, root, seed=None, train=True, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / ("train.txt" if train else "val.txt")
        self.scenes = [self.root / line.strip() for line in scene_list_path.read_text().splitlines() if line.strip()]
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self):
        pair_set = []
        for scene in self.scenes:
            imgs = _glob_images(scene)
            intrinsics_files = sorted(scene.glob("*.txt"))
            for i in range(0, len(imgs) - 1, 2):
                intrinsic = np.genfromtxt(intrinsics_files[int(i / 2)]).astype(np.float32).reshape((3, 3))
                sample = {"intrinsics": intrinsic, "tgt": imgs[i], "ref_imgs": [imgs[i + 1]]}
                pair_set.append(sample)
        random.shuffle(pair_set)
        self.samples = pair_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample["tgt"])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample["ref_imgs"]]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample["intrinsics"]))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample["intrinsics"])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)

