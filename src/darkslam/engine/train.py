from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig
from ..data.dataset import ImageFolderDataset
from ..models.disp_resnet import DispResNet
from ..utils.logging import AverageMeter, set_seed


def _smoothness_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    # depth: Bx1xHxW, image: Bx3xHxW
    dx_depth = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy_depth = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    dx_img = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
    dy_img = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-dx_img)
    weight_y = torch.exp(-dy_img)
    return (dx_depth * weight_x).mean() + (dy_depth * weight_y).mean()


def _supervised_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    valid = gt_depth > 0
    if valid.sum() == 0:
        return torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype)
    return torch.abs(pred_depth[valid] - gt_depth[valid]).mean()


def _save_checkpoint(save_dir: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: TrainConfig) :  # type: ignore[override]
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "latest.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
        },
        path,
    )
    return path


def _append_metrics(save_dir: Path, metrics: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def run_train(cfg: TrainConfig) -> None:
    """A lightweight supervised training loop (optional).

    This is a public baseline and does NOT include the full self-supervised training strategy.
    For SC-Depth style self-supervised training, use `python -m darkslam.scdepth.train`.
    """

    if cfg.depth_root is None:
        raise ValueError("`depth_root` is required for this supervised training baseline.")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(cfg.data_root, depth_root=cfg.depth_root, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = DispResNet(num_layers=cfg.resnet_layers, pretrained=cfg.with_pretrain).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        sup_meter = AverageMeter()
        smooth_meter = AverageMeter()

        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}", ncols=100)
        for batch in pbar:
            image = batch["image"].to(device)
            gt_depth = batch["depth"].to(device)

            image = (image - 0.45) / 0.225
            disp = model(image)
            if isinstance(disp, (list, tuple)):
                disp = disp[0]
            pred_depth = 1.0 / disp

            smooth = _smoothness_loss(pred_depth, image)
            supervised = _supervised_depth_loss(pred_depth, gt_depth)
            loss = supervised + cfg.smooth_weight * smooth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), image.size(0))
            sup_meter.update(supervised.item(), image.size(0))
            smooth_meter.update(smooth.item(), image.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", smooth=f"{smooth_meter.avg:.4f}")

        ckpt_path = _save_checkpoint(save_dir, model=model, optimizer=optimizer, epoch=epoch, cfg=cfg)
        _append_metrics(
            save_dir,
            {
                "epoch": epoch,
                "loss": loss_meter.avg,
                "supervised": sup_meter.avg,
                "smooth": smooth_meter.avg,
                "checkpoint": str(ckpt_path),
            },
        )
