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


class SequenceFolder(data.Dataset):
    """Sequence loader expected by SC-Depth style training.

    Layout:
      root/<scene>/0000000.jpg
      root/<scene>/0000001.jpg
      ...
      root/<scene>/cam.txt
      root/train.txt, root/val.txt
    """

    def __init__(
        self,
        root,
        seed=None,
        train=True,
        sequence_length=3,
        transform=None,
        skip_frames=1,
        dataset="kitti",
    ):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / ("train.txt" if train else "val.txt")
        self.scenes = [self.root / line.strip() for line in scene_list_path.read_text().splitlines() if line.strip()]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape((3, 3))
            imgs = _glob_images(scene)
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs) - demi_length * self.k):
                sample = {"intrinsics": intrinsics, "tgt": imgs[i], "ref_imgs": []}
                for j in shifts:
                    sample["ref_imgs"].append(imgs[i + j])
                sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

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

