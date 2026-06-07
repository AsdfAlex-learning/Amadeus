from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.motion.model import FullDuplexDiT
from src.motion.training.dataset import MotionDataset
from src.motion.training.lora import apply_lora, save_lora


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    num_epochs: int = 100,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    num_params: int = 45,
    hidden_dim: int = 320,
    num_layers: int = 4,
    num_diffusion_steps: int = 1000,
    device: str = "auto",
    use_amp: bool = True,
    dataset_type: str = "preprocessed",
    val_split: float = 0.1,
    resume_from: str | None = None,
    # ── LoRA ──
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: list[str] | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Training Full-Duplex DiT on {device}")

    # Precompute diffusion schedule (DDPM linear)
    betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
    alphas = 1.0 - betas
    _alphas_cumprod = torch.cumprod(alphas, dim=0)

    full_dataset = MotionDataset(data_dir, dataset_type=dataset_type)
    if len(full_dataset) == 0:
        logger.error(f"No samples found in {data_dir}")
        return

    # Train/val split
    val_size = max(1, int(len(full_dataset) * val_split)) if val_split > 0 else 0
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    logger.info(f"Dataset: {train_size} train / {val_size} val samples")

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_size > 0 else None
    model = FullDuplexDiT(
        num_params=num_params,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_gradient_checkpointing=(device in ("cuda", "mps")),
    ).to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state)
            start_epoch = int(resume_path.stem.split("_")[-1]) + 1
            logger.info(f"Resumed from {resume_path} (epoch {start_epoch})")
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}")

    # ── LoRA application ──
    if use_lora:
        lora_config: dict[str, Any] = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
        if lora_target_modules:
            lora_config["target_modules"] = lora_target_modules
        lora_info = apply_lora(model, lora_config)
        logger.info(
            f"LoRA: {lora_info['replaced_count']} modules, "
            f"{lora_info['trainable_params']:,} trainable / {lora_info['total_params']:,} total "
            f"({100 * lora_info['trainable_params'] / max(lora_info['total_params'], 1):.2f}%)"
        )
        # Save LoRA config alongside checkpoints for reproducibility
        (output_dir / "lora_config.pt").parent.mkdir(parents=True, exist_ok=True)
        torch.save(lora_config, output_dir / "lora_config.pt")

    trainable = model.get_trainable_param_count()
    total = model.get_total_param_count()
    logger.info(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # Optimizer: only train LoRA params when LoRA is active
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device) if use_amp and device == "cuda" else None

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            audio = batch["user_audio"].to(device)
            motion = batch["motion"].to(device)
            B = audio.shape[0]
            tts_audio = batch.get("tts_audio", torch.zeros_like(audio)).to(device)
            visual = batch.get("visual_frames", torch.zeros(B, 5, 3, 224, 224, device=device)).to(device)
            identity = batch.get("identity_id", torch.zeros(B, dtype=torch.long, device=device)).to(device)
            prompts = batch.get("text_prompt", [""] * B)
            if isinstance(prompts, list):
                pass  # already a list
            else:
                prompts = [""] * B

            t = torch.randint(0, num_diffusion_steps, (B,), device=device)
            noise = torch.randn_like(motion)

            # Standard DDPM forward process: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
            alpha_cumprod_t = _alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).to(device)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            noisy = sqrt_alpha_cumprod_t * motion + sqrt_one_minus_alpha_cumprod_t * noise

            with torch.amp.autocast(device_type=device, enabled=scaler is not None):
                # X-prediction: model directly predicts the clean sample x_0
                # Output is bounded to [0, 1] by Sigmoid — naturally compatible with
                # Live2D parameter range. Inference must use x-prediction DDIM
                # (see inference.py: _diffusion_infer).
                pred = model(audio, tts_audio, visual, prompts, identity, t, noisy)
                loss = criterion(pred, motion)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(dataloader), 1)

        # Validation loss
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    motion = batch["motion"].to(device)
                    audio = batch["user_audio"].to(device)
                    B = audio.shape[0]
                    tts_audio = batch.get("tts_audio", torch.zeros_like(audio)).to(device)
                    visual = batch.get("visual_frames", torch.zeros(B, 5, 3, 224, 224, device=device)).to(device)
                    identity = batch.get("identity_id", torch.zeros(B, dtype=torch.long, device=device)).to(device)
                    prompts = batch.get("text_prompt", [""] * B)
                    t = torch.randint(0, num_diffusion_steps, (B,), device=device)
                    noise = torch.randn_like(motion)
                    alpha_cumprod_t = _alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1).to(device)
                    noisy = torch.sqrt(alpha_cumprod_t) * motion + torch.sqrt(1.0 - alpha_cumprod_t) * noise
                    with torch.amp.autocast(device_type=device, enabled=scaler is not None):
                        pred = model(audio, tts_audio, visual, prompts, identity, t, noisy)
                        val_loss += criterion(pred, motion).item()
            val_loss /= max(len(val_loader), 1)
            model.train()

        log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        if val_loader is not None:
            log_msg += f" | Val: {val_loss:.6f}"
        logger.info(log_msg)

        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            if use_lora:
                lora_ckpt_path = output_dir / f"lora_adapter_epoch_{epoch + 1:04d}.pt"
                save_lora(model, lora_ckpt_path)
                logger.info(f"LoRA checkpoint saved: {lora_ckpt_path}")
            else:
                checkpoint_path = output_dir / f"full_duplex_dit_epoch_{epoch + 1:04d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    if use_lora:
        final_lora_path = output_dir / "lora_adapter.pt"
        save_lora(model, final_lora_path)
        logger.info(f"LoRA adapter saved to {final_lora_path}")
    else:
        checkpoint_path = output_dir / "full_duplex_dit.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Full-Duplex DiT")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/motion")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_params", type=int, default=45)
    parser.add_argument("--hidden_dim", type=int, default=320)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--dataset_type", type=str, default="preprocessed",
                        choices=["preprocessed", "mead", "biwi", "vocaset"],
                        help="Dataset format type")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (0=no validation)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine-tuning (freezes base model)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA decomposition rank (default: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA scaling factor (default: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="*", default=None,
                        help="Module name patterns for LoRA (e.g. dit_blocks output_head)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        num_params=args.num_params,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
        use_amp=not args.no_amp,
        dataset_type=args.dataset_type,
        val_split=args.val_split,
        resume_from=args.resume,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
