from __future__ import annotations

import argparse

from .config import InferConfig, TrainConfig, default_device
from .engine.infer import run_infer
from .engine.train import run_train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="darkslam", description="DarkSLAM (selective open-source release)")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Supervised baseline training (requires depth_root).")
    train.add_argument("--data-root", required=True, help="Folder containing an image sequence.")
    train.add_argument("--depth-root", default=None, help="Optional folder containing *.npy depth maps (supervised).")
    train.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=(256, 256))
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--save-dir", default="./outputs")
    train.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default=None, help="e.g. cpu / cuda / cuda:0 (default: auto)")

    infer = sub.add_parser("infer", help="Depth inference from a checkpoint.")
    infer.add_argument("--image", required=True, help="Input image path.")
    infer.add_argument("--checkpoint", required=True, help="Checkpoint path (*.pt).")
    infer.add_argument("--out", required=True, help="Output depth path (*.npy).")
    infer.add_argument("--resnet-layers", type=int, default=18, choices=[18, 50], help="Backbone depth encoder.")
    infer.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    infer.add_argument("--device", default=None, help="e.g. cpu / cuda / cuda:0 (default: auto)")
    infer.add_argument("--out-png", default=None, help="Optional visualization output (*.png).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    device = args.device or default_device()
    try:
        if args.command == "train":
            cfg = TrainConfig(
                data_root=args.data_root,
                depth_root=args.depth_root,
                image_size=tuple(args.image_size),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                save_dir=args.save_dir,
                resume=args.resume,
                seed=args.seed,
                device=device,
            )
            run_train(cfg)
            return 0

        if args.command == "infer":
            cfg = InferConfig(
                image=args.image,
                checkpoint=args.checkpoint,
                out=args.out,
                resnet_layers=args.resnet_layers,
                image_size=tuple(args.image_size) if args.image_size is not None else None,
                out_png=args.out_png,
                device=device,
            )
            run_infer(cfg)
            return 0
    except NotImplementedError as e:
        parser.exit(status=2, message=f"{e}\n")

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
