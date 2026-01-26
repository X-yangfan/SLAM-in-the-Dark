from __future__ import annotations

"""
============================================================================
DarkSLAM SC-Depth Style Self-Supervised Training

This module contains the self-supervised depth training pipeline.

** CODE NOT YET RELEASED **

The implementation will be released upon paper acceptance.

============================================================================
"""

# TODO: Code will be released soon


def main(argv=None):
    """Training entry - Code not yet released"""
    raise NotImplementedError(
        "SC-Depth style training code is not yet released. "
        "Please wait for future updates."
    )


if __name__ == "__main__":
    raise SystemExit(main())


def _try_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir)
    except Exception:
        return None


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]
    ref_depths = []
    for ref_img in ref_imgs:
        ref_depths.append([1 / disp for disp in disp_net(ref_img)])
    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))
    return poses, poses_inv


def train_one_epoch(args, train_loader, disp_net, pose_net, optimizer, device, writer=None):
    disp_net.train()
    pose_net.train()

    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    running = []

    pbar = tqdm(enumerate(train_loader), total=min(len(train_loader), args.epoch_size), ncols=100, desc="train")
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in pbar:
        if i >= args.epoch_size:
            break

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img,
            ref_imgs,
            intrinsics,
            tgt_depth,
            ref_depths,
            poses,
            poses_inv,
            args.num_scales,
            bool(args.with_ssim),
            bool(args.with_mask),
            bool(args.with_auto_mask),
            args.padding_mode,
        )
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(running[-50:]):.4f}")

        if writer is not None and (i % args.print_freq == 0):
            step = args._global_step
            writer.add_scalar("train/total_loss", loss.item(), step)
            writer.add_scalar("train/photo_loss", loss_1.item(), step)
            writer.add_scalar("train/smooth_loss", loss_2.item(), step)
            writer.add_scalar("train/geometry_loss", loss_3.item(), step)
        args._global_step += 1

    return float(np.mean(running)) if running else 0.0


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, device):
    disp_net.eval()
    pose_net.eval()

    losses = []
    pbar = tqdm(val_loader, total=len(val_loader), ncols=100, desc="valid(no-gt)")
    for tgt_img, ref_imgs, intrinsics, intrinsics_inv in pbar:
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = [[1 / disp_net(ref_img)] for ref_img in ref_imgs]
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img,
            ref_imgs,
            intrinsics,
            tgt_depth,
            ref_depths,
            poses,
            poses_inv,
            args.num_scales,
            bool(args.with_ssim),
            bool(args.with_mask),
            False,
            args.padding_mode,
        )
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        loss = loss_1.item()
        losses.append(loss)
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, device):
    disp_net.eval()

    error_names = ["abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
    errors_sum = np.zeros(len(error_names), dtype=np.float64)
    n = 0
    pbar = tqdm(val_loader, total=len(val_loader), ncols=100, desc="valid(gt)")
    for tgt_img, depth in pbar:
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)
        if depth.nelement() == 0:
            continue

        output_disp = disp_net(tgt_img)
        output_depth = 1 / output_disp[:, 0]
        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        metrics = compute_errors(depth, output_depth, args.dataset)
        errors_sum += np.array(metrics, dtype=np.float64)
        n += 1
        pbar.set_postfix(abs_rel=f"{(errors_sum[1]/max(1,n)):.4f}")

    if n == 0:
        return [0.0] * len(error_names), error_names
    return (errors_sum / n).tolist(), error_names


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SC-Depth style training (public baseline).")
    p.add_argument("data", help="Path to dataset root (expects train.txt/val.txt).")
    p.add_argument("--save-dir", default="checkpoints", help="Checkpoint root.")
    p.add_argument("--name", required=True, help="Experiment name.")

    p.add_argument("--folder-type", choices=["sequence", "pair"], default="sequence")
    p.add_argument("--sequence-length", type=int, default=3)
    p.add_argument("--dataset", choices=["kitti", "nyu"], default="kitti")
    p.add_argument("--with-gt", action="store_true", help="Use GT depth validation set if available.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--epoch-size", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.999)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--resnet-layers", type=int, default=18, choices=[18, 50])
    p.add_argument("--with-pretrain", type=int, default=1)
    p.add_argument("--pretrained-disp", default=None)
    p.add_argument("--pretrained-pose", default=None)

    p.add_argument("--num-scales", type=int, default=1)
    p.add_argument("--photo-loss-weight", type=float, default=1.0)
    p.add_argument("--smooth-loss-weight", type=float, default=0.1)
    p.add_argument("--geometry-consistency-weight", type=float, default=0.5)
    p.add_argument("--with-ssim", type=int, default=1)
    p.add_argument("--with-mask", type=int, default=1)
    p.add_argument("--with-auto-mask", type=int, default=0)
    p.add_argument("--padding-mode", choices=["zeros", "border"], default="zeros")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--log-tensorboard", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args._global_step = 0

    timestamp = _dt.datetime.now().strftime("%m-%d-%H%M%S")
    save_path = Path(args.save_dir) / args.name / timestamp
    save_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    writer = _try_summary_writer(save_path) if args.log_tensorboard else None

    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    train_transform = custom_transforms.Compose(
        [
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize,
        ]
    )
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    if args.folder_type == "sequence":
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset,
        )
    else:
        train_set = PairFolder(args.data, seed=args.seed, train=True, transform=train_transform)

    if args.with_gt:
        val_set = ValidationSet(args.data, transform=valid_transform, dataset=args.dataset)
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            dataset=args.dataset,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    disp_net = DispResNet(args.resnet_layers, bool(args.with_pretrain)).to(device)
    pose_net = PoseResNet(18, bool(args.with_pretrain)).to(device)

    if args.pretrained_disp:
        weights = torch.load(args.pretrained_disp, map_location="cpu")
        disp_net.load_state_dict(weights["state_dict"], strict=False)
    if args.pretrained_pose:
        weights = torch.load(args.pretrained_pose, map_location="cpu")
        pose_net.load_state_dict(weights["state_dict"], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    optimizer = torch.optim.Adam(
        [
            {"params": disp_net.parameters(), "lr": args.lr},
            {"params": pose_net.parameters(), "lr": args.lr},
        ],
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay,
    )

    summary_csv = save_path / "progress_log_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerow(["train_loss", "validation_metric"])

    best_metric = float("inf")
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(args, train_loader, disp_net, pose_net, optimizer, device, writer=writer)

        if args.with_gt:
            errors, names = validate_with_gt(args, val_loader, disp_net, device)
            decisive = float(errors[1])  # abs_rel
            print("  val:", ", ".join(f"{n}={v:.4f}" for n, v in zip(names, errors)))
        else:
            decisive = validate_without_gt(args, val_loader, disp_net, pose_net, device)
            print(f"  val photo loss={decisive:.4f}")

        is_best = decisive < best_metric
        best_metric = min(best_metric, decisive)

        save_checkpoint(
            save_path,
            {"epoch": epoch + 1, "state_dict": disp_net.module.state_dict()},
            {"epoch": epoch + 1, "state_dict": pose_net.module.state_dict()},
            is_best,
        )

        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow([train_loss, decisive])

    if writer is not None:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

