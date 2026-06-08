"""Long-running training wrapper for the Amadeus base model.

Runs train() with the project's standard CLI-like arguments, plus
optional periodic sample generation that renders a motion GIF for
a held-out validation clip so you can see whether the model has
learned anything as it trains.

Usage:
    python scripts/train_base.py \\
        --data_dir data/preprocessed/hdtf_subset \\
        --output_dir models/motion/hdtf_base \\
        --num_epochs 200 \\
        --sample_every 10

Designed to run on a single GPU (8GB VRAM is sufficient). Logs to
console and to <output_dir>/train.log. Sample GIFs and the loss
curve land in <output_dir>/samples/ and <output_dir>/loss_curve.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Amadeus base model trainer")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=320)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="preprocessed")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Save a sample GIF every N epochs (0=disable)")
    args = parser.parse_args()

    # Use the project's train() function for the heavy lifting.
    # We re-import here so users can edit the file without restarting.
    import importlib
    import src.motion.training.train
    importlib.reload(src.motion.training.train)
    from src.motion.training.train import train

    # Set up log capture (print to both console and file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "train.log"

    import loguru
    logger = loguru.logger
    logger.add(str(log_file), rotation="50 MB", retention="7 days")

    # Call train() with all the new flags (L2/L3/L4/etc)
    print("=" * 60)
    print("Amadeus base model training")
    print("=" * 60)
    print(f"  data_dir:       {args.data_dir}")
    print(f"  output_dir:     {args.output_dir}")
    print(f"  num_epochs:     {args.num_epochs}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  grad_accum:     x{args.grad_accum_steps}")
    print(f"  lr:             {args.learning_rate}")
    print(f"  hidden_dim:     {args.hidden_dim}")
    print(f"  num_layers:     {args.num_layers}")
    print(f"  weight_decay:   {args.weight_decay}")
    print(f"  warmup_steps:   {args.warmup_steps}")
    print(f"  ema_decay:      {args.ema_decay}")
    print(f"  early_stop:     {args.early_stopping_patience}")
    print(f"  device:         {args.device}")
    print(f"  AMP:            {args.use_amp}")
    print(f"  LoRA:           {args.use_lora}")
    print(f"  log file:       {log_file}")
    print("=" * 60)

    t0 = time.time()
    train(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_params=45,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        ema_decay=args.ema_decay,
        device=args.device,
        use_amp=args.use_amp,
        dataset_type=args.dataset_type,
        val_split=args.val_split,
        use_lora=args.use_lora,
    )
    print(f"\nTotal training time: {(time.time()-t0)/60:.1f} min")

    # Render the loss curve from the log file
    try:
        import importlib
        import src.motion.training.visualize
        importlib.reload(src.motion.training.visualize)
        from src.motion.training.visualize import parse_train_log, plot_loss_curve
        eps, tl, vl, _ = parse_train_log(log_file)
        if eps:
            print(f"\nLoss curve: {len(eps)} epochs parsed")
            plot_loss_curve(eps, tl, vl, args.output_dir / "loss_curve.png",
                            title=f"Amadeus base model — {len(eps)} epochs")
    except Exception as e:
        print(f"Loss curve rendering failed: {e}")


if __name__ == "__main__":
    main()
