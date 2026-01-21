from __future__ import division

import shutil
from pathlib import Path

import torch


def save_checkpoint(save_path: Path, dispnet_state: dict, exp_pose_state: dict, is_best: bool, filename: str = "checkpoint.pth.tar"):
    save_path.mkdir(parents=True, exist_ok=True)

    file_prefixes = ["dispnet", "exp_pose"]
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / f"{prefix}_{filename}")

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path / f"{prefix}_{filename}", save_path / f"{prefix}_model_best.pth.tar")

