"""Visualization utilities for the motion model.

Generates:
  1. Loss curves from train.log / per-epoch prints
  2. Motion parameter time-series GIFs that compare ground truth vs
     model predictions for a held-out validation clip
  3. Optional bar charts of parameter distributions at a single time step

The visualizer is lightweight (matplotlib + Pillow only) and runs on CPU
so it does not compete with training for GPU memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LIVE2D_PARAM_NAMES = [
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLX", "ParamBrowLY", "ParamBrowRX", "ParamBrowRY",
    "ParamMouthOpenY", "ParamMouthForm", "ParamCheek", "ParamBreath",
    "ParamArmLX", "ParamArmLY", "ParamArmRX", "ParamArmRY",
    "ParamHairFront", "ParamHairBack", "ParamHairSideL", "ParamHairSideR",
    "ParamTear", "ParamBlush", "ParamNose",
    "ParamLipUpper", "ParamLipLower", "ParamTongue",
    "ParamEarL", "ParamEarR", "ParamTail", "ParamWingL", "ParamWingR",
    "ParamItem1", "ParamItem2", "ParamItem3",
    "ParamExtra1", "ParamExtra2", "ParamExtra3", "ParamExtra4", "ParamExtra5",
]
assert len(LIVE2D_PARAM_NAMES) == 45


def parse_train_log(log_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    """Parse the standard loguru line: 'Epoch N/M | Loss: X | LR: Y | Val: Z'.

    Returns (epochs, train_losses, val_losses, lrs).
    """
    epochs, train_losses, val_losses, lrs = [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.eE+-]+)(?:\s*\|\s*Val:\s*([\d.]+))?"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        total = int(m.group(2))
        tl = float(m.group(3))
        lr = float(m.group(4))
        vl = float(m.group(5)) if m.group(5) else float("nan")
        epochs.append(ep)
        train_losses.append(tl)
        val_losses.append(vl)
        lrs.append(lr)
    return epochs, train_losses, val_losses, lrs


def plot_loss_curve(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    out_path: Path,
    title: str = "FullDuplexDiT training loss",
) -> None:
    """Render a 1- or 2-panel loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_losses, label="train", linewidth=1.5, color="#4a90d9")
    valid_val = [v for v in val_losses if not (v != v)]  # filter NaN
    if valid_val:
        val_epochs = [e for e, v in zip(epochs, val_losses) if not (v != v)]
        ax.plot(val_epochs, valid_val, label="val", linewidth=1.5, color="#d94a90")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (x-prediction)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_motion_comparison(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    out_path: Path,
    title: str = "Motion parameters: ground truth vs predicted",
    fps: int = 50,
    max_params_per_panel: int = 6,
) -> None:
    """Plot a side-by-side comparison of ground truth vs predicted motion.

    ground_truth, predicted: (T, 45) arrays in [0, 1].
    """
    T, P = ground_truth.shape
    assert predicted.shape == (T, P), f"shape mismatch: gt {ground_truth.shape}, pred {predicted.shape}"
    assert P == 45

    n_panels = (P + max_params_per_panel - 1) // max_params_per_panel
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    t_axis = np.arange(T) / fps
    for i in range(n_panels):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        idxs = list(range(i * max_params_per_panel, min((i + 1) * max_params_per_panel, P)))
        for idx in idxs:
            ax.plot(t_axis, ground_truth[:, idx], color="#4a90d9", alpha=0.7, linewidth=1)
            ax.plot(t_axis, predicted[:, idx], "--", color="#d94a90", alpha=0.7, linewidth=1)
        ax.set_title(f"params [{idxs[0]}-{idxs[-1]}]", fontsize=9)
        ax.set_ylim(0, 1)
        if r == n_rows - 1:
            ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for j in range(n_panels, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)

    # Legend once
    fig.legend(["ground truth", "predicted"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def motion_to_gif(
    params: np.ndarray,
    out_path: Path,
    fps: int = 50,
    title: str = "Live2D motion",
    param_subset: list[int] | None = None,
) -> None:
    """Render a 1-D Live2D motion sequence as a small animated GIF.

    For visualization only — we plot the parameters as a coloured bar
    chart and animate it frame by frame.
    """
    T, P = params.shape
    if param_subset is None:
        param_subset = list(range(P))

    # Render each frame as a Pillow image
    W, H = 480, 320
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, H), (245, 245, 245))
        # Draw a simple bar chart for the selected parameters
        n = len(param_subset)
        bar_w = W / (n + 1)
        for i, idx in enumerate(param_subset):
            v = float(np.clip(params[t, idx], 0.0, 1.0))
            x0 = int((i + 0.5) * bar_w)
            x1 = int((i + 1) * bar_w) - 2
            y1 = int(H * (1 - v))
            # Simple bar via Pillow
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            d.rectangle([x0, y1, x1, H - 30], fill=(74, 144, 217))
        frames.append(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, 1000 // fps)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], loop=0, duration=duration_ms
    )
    print(f"  wrote {out_path} ({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Amadeus training visualizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_loss = sub.add_parser("loss", help="Plot loss curve from train log")
    p_loss.add_argument("--log", type=Path, required=True)
    p_loss.add_argument("--out", type=Path, required=True)

    p_motion = sub.add_parser(
        "motion", help="Compare ground truth vs predicted motion"
    )
    p_motion.add_argument("--gt_npz", type=Path, required=True,
                          help="Path to .npz with 'live2d_params' key")
    p_motion.add_argument("--pred_npy", type=Path, required=True,
                          help="Path to .npy (T, 45) with predicted params")
    p_motion.add_argument("--out", type=Path, required=True)
    p_motion.add_argument("--fps", type=int, default=50)

    p_gif = sub.add_parser("gif", help="Animate one motion sequence as a GIF")
    p_gif.add_argument("--input", type=Path, required=True,
                       help="Path to .npz or .npy motion sequence")
    p_gif.add_argument("--out", type=Path, required=True)
    p_gif.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()
    if args.cmd == "loss":
        eps, tl, vl, lr = parse_train_log(args.log)
        if not eps:
            print(f"No epoch lines found in {args.log}")
            return
        print(f"Parsed {len(eps)} epochs from {args.log}")
        plot_loss_curve(eps, tl, vl, args.out)
    elif args.cmd == "motion":
        gt = np.load(args.gt_npz)["live2d_params"]
        pred = np.load(args.pred_npy)
        T = min(len(gt), len(pred))
        gt = gt[:T]
        pred = pred[:T]
        plot_motion_comparison(gt, pred, args.out, fps=args.fps)
    elif args.cmd == "gif":
        path = args.input
        if path.suffix == ".npz":
            arr = np.load(path)["live2d_params"]
        else:
            arr = np.load(path)
        motion_to_gif(arr, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
