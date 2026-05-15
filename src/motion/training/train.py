from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.motion.model import FullDuplexDiT
from src.motion.training.dataset import MotionDataset


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
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info(f"Training Full-Duplex DiT on {device}")

    dataset = MotionDataset(data_dir)
    if len(dataset) == 0:
        logger.error(f"No samples found in {data_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = FullDuplexDiT(
        num_params=num_params,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_gradient_checkpointing=(device in ("cuda", "mps")),
    ).to(device)

    trainable = model.get_trainable_param_count()
    total = model.get_total_param_count()
    logger.info(f"Model: {trainable:,} trainable / {total:,} total parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device) if use_amp and device == "cuda" else None

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (audio, motion) in enumerate(dataloader):
            audio = audio.to(device)
            motion = motion.to(device)
            B, _ = audio.shape[0], motion.shape[1]
            tts_audio = torch.zeros_like(audio)
            visual = torch.zeros(B, 5, 3, 224, 224, device=device)
            identity = torch.zeros(B, dtype=torch.long, device=device)
            prompts = [""] * B

            t = torch.randint(0, num_diffusion_steps, (B,), device=device)
            noise = torch.randn_like(motion)
            noisy = motion + noise * (1.0 - 1e-4) * (t.float() / num_diffusion_steps).unsqueeze(
                -1
            ).unsqueeze(-1)

            with torch.amp.autocast(device_type=device, enabled=scaler is not None):
                pred = model(audio, tts_audio, visual, prompts, identity, t, noisy)
                loss = criterion(pred, noise)

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
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    checkpoint_path = output_dir / "full_duplex_dit.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")


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
    )
