from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import InferConfig
from ..models.disp_resnet import DispResNet

def run_infer(cfg: InferConfig) -> None:
    device = torch.device(cfg.device)

    model = DispResNet(num_layers=cfg.resnet_layers, pretrained=False)
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    if state_dict is None:
        raise ValueError("Unsupported checkpoint format: expected dict with key `state_dict` or `model`.")
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    image = Image.open(cfg.image).convert("RGB")
    if cfg.image_size is not None:
        image = image.resize((cfg.image_size[1], cfg.image_size[0]), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.45) / 0.225
    tensor = tensor.to(device)

    with torch.no_grad():
        disp = model(tensor)  # Bx1xHxW
        if isinstance(disp, (list, tuple)):
            disp = disp[0]
        depth = (1.0 / disp).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth)

    if cfg.out_png is not None:
        png_path = Path(cfg.out_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        vis = depth.copy()
        vis = vis - vis.min()
        if vis.max() > 1e-12:
            vis = vis / vis.max()
        vis = (vis * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(png_path)
