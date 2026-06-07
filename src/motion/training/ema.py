"""Exponential Moving Average (EMA) for diffusion model weights.

Standard practice for diffusion model training: maintain a shadow copy of
the model parameters that is updated as an exponential moving average of
the training parameters. EMA weights are generally smoother and produce
higher-quality samples than the final training weights.

This module is self-contained and depends only on torch + a copy of the
model's state_dict. It intentionally avoids hard dependency on
`torch_ema` (which is a third-party package) so the training pipeline
has no extra installation requirement.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class EMAModel:
    """Track an exponential moving average of a model's parameters.

    Usage:
        ema = EMAModel(model, decay=0.999)
        for step in training:
            ...
            optimizer.step()
            ema.update(model)

        # at evaluation / inference:
        ema.copy_to(model)
        # ... or persist
        torch.save(ema.state_dict(), "ema.pt")

    The EMA only tracks parameters that have `requires_grad=True` so the
    frozen encoders (Hubert, MobileNet, BERT) are not duplicated.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # Snapshot of the (trainable) parameters on the same device as the
        # original module. We do not require them to be contiguous.
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow weights using the current model parameters.

        shadow ← decay · shadow + (1 − decay) · param
        """
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            shadow = self.shadow[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module, store_original: bool = False) -> dict[str, torch.Tensor] | None:
        """Replace the model's trainable parameters with the EMA weights.

        If `store_original=True`, the original parameters are returned so
        the caller can restore them later (e.g. for the next optimizer
        step). Otherwise the original weights are lost.
        """
        if not store_original:
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))
            return None

        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Inverse of copy_to(store_original=True)."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device).to(param.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("shadow", {})
        for k, v in loaded.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)

    def to(self, device: torch.device) -> "EMAModel":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    def keys(self) -> Iterable[str]:
        return self.shadow.keys()
