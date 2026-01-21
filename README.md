# DarkSLAM


## Dataset
- **Public dataset placeholder**: `http://xxx`  
  - Replace this URL with your own dataset hosting link when it is ready (e.g., internal server, HuggingFace, or another download page).

## Core Directory Structure
- `src/darkslam/models/`: Depth / pose networks (ResNet encoder + decoder).
- `src/darkslam/scdepth/`: SC-Depth-style public baseline training code (self-supervised: photo / smooth / geometry losses, etc.).
- `src/darkslam/slam/`:
  - `loop_closure_detection.py`: Loop closure retrieval (optional FAISS; falls back to brute-force search if FAISS is not available).
  - `pose_graph_optimization.py`: Pose graph optimization (requires Python bindings for `g2o`; will warn if not installed).
- `src/darkslam/engine/`: Lightweight CLI training / inference entry points (supervised baseline training; self-supervised training is under `scdepth/`).

## Installation
```bash
cd darkslam
pip install -e .
```

## Usage (Brief)
- **Self-supervised training (public baseline)**  
  - `python -m darkslam.scdepth.train <DATA_ROOT> --name <EXP_NAME>`
- **Supervised training (optional baseline, requires `*.npy` depth labels via `--depth-root`)**  
  - `darkslam train --data-root <IMG_DIR> --depth-root <DEPTH_DIR>`
- **Inference** (load from `dispnet_*.pth.tar` or `latest.pt` saved by this package)  
  - `darkslam infer --image <IMG> --checkpoint <CKPT> --out <OUT.npy>`

## Notes
- This repo is intended for academic exchange and as an engineering reference; performance and full reproduction depend on your own extensions.  
- If you need further guidance on which specific files/functions should remain closed (e.g., certain losses, loop edge selection strategies), you can selectively remove or obfuscate them on top of this base.
