"""LoRA (Low-Rank Adaptation) for FullDuplexDiT character-specific fine-tuning.

Each character gets its own LoRA adapter (~MB per character) that modifies
attention projections, FFN layers, AdaLN projections, and Conv1d output
head — all while keeping the base model frozen.

Usage:
    from src.motion.training.lora import apply_lora, save_lora, load_lora

    model = FullDuplexDiT(...)
    lora_info = apply_lora(model, {"lora_rank": 8, "lora_alpha": 16})
    # ... train only LoRA params ...
    save_lora(model, "models/lora/kurisu.pt")
    # At inference: load base model + LoRA adapter
    load_lora(model, "models/lora/kurisu.pt")
    merge_lora(model)  # optional: merge for faster inference
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# LoRA Modules
# ═══════════════════════════════════════════════════════════════


class LoRALinear(nn.Module):
    """Drop-in LoRA replacement for nn.Linear.

    Original weight is frozen; low-rank A/B matrices are trainable.
    Forward: y = W_original @ x + (B @ A @ x^T)^T * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Store original weight (frozen)
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=original.weight.device))

        # Dropout on input before LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zeros for B (starts as zero LoRA contribution)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # Store original dtype for consistent output
        self._dtype = original.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear path (frozen)
        result = nn.functional.linear(x, self.weight, self.bias)

        # LoRA path
        dropped = self.lora_dropout(x)
        lora_out = (dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        result = result + lora_out.to(result.dtype)

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weight (in-place). After this, forward
        only uses the merged weight — call once before deployment."""
        with torch.no_grad():
            self.weight.data += (self.scaling * self.lora_B.data @ self.lora_A.data).to(self.weight.dtype)
            # Zero out LoRA so forward is a no-op
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


class LoRAConv1d(nn.Module):
    """Drop-in LoRA replacement for nn.Conv1d.

    LoRA is applied as a rank-decomposition on the channel dimension,
    equivalent to modifying the (out_channels, in_channels) part of
    the kernel while keeping groups and kernel_size unchanged.
    """

    def __init__(
        self,
        original: nn.Conv1d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size[0] if isinstance(original.kernel_size, tuple) else original.kernel_size
        self.stride = original.stride[0] if isinstance(original.stride, tuple) else original.stride
        self.padding = original.padding[0] if isinstance(original.padding, tuple) else original.padding
        self.dilation = original.dilation[0] if isinstance(original.dilation, tuple) else original.dilation
        self.groups = original.groups
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Store original weight/bias (frozen)
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices: factorize the channel mixing
        # Original weight shape: (out_channels, in_channels/groups, kernel_size)
        # We decompose along channel dims: (out_channels, rank) x (rank, in_channels/groups * kernel_size)
        fan_in = self.in_channels // self.groups * self.kernel_size
        self.lora_A = nn.Parameter(torch.empty(rank, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.view(rank, -1), a=5 ** 0.5)
        self._dtype = original.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base conv path (frozen)
        result = nn.functional.conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        # LoRA path: lora_B @ lora_A reshaped to (out_channels, in_channels/groups, kernel_size)
        dropped = self.lora_dropout(x)
        lora_kernel = (self.lora_B @ self.lora_A).view(
            self.out_channels, self.in_channels // self.groups, self.kernel_size
        ).to(self.weight.device)
        lora_out = nn.functional.conv1d(
            dropped.to(lora_kernel.device), lora_kernel * self.scaling, None, self.stride, self.padding, self.dilation, self.groups
        )
        result = result + lora_out.to(result.device)

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weight."""
        with torch.no_grad():
            lora_kernel = (self.lora_B @ self.lora_A).view(
                self.out_channels, self.in_channels // self.groups, self.kernel_size
            )
            self.weight.data += (self.scaling * lora_kernel).to(self.weight.device).to(self.weight.dtype)
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


# ═══════════════════════════════════════════════════════════════
# LoRA Application & Management
# ═══════════════════════════════════════════════════════════════


def _match_module_name(name: str, patterns: list[str]) -> bool:
    """Check if a module name matches any of the glob-like patterns.

    Patterns support * as wildcard. Examples:
      - "dit_blocks.*.attn" matches "dit_blocks.0.attn"
      - "output_head" matches "output_head"
      - "dit_blocks" matches "dit_blocks.0.ffn" (parent match)
    """
    for pattern in patterns:
        # Convert glob pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]*")
        # Full match: either exact or as prefix
        if re.fullmatch(regex_pattern, name):
            return True
        # Parent match: pattern is a prefix of the module path
        if name.startswith(pattern.replace("*", "") ) and pattern.endswith("*"):
            if re.fullmatch(regex_pattern + r"\..*", name):
                return True
        # Simple prefix match: "dit_blocks" matches "dit_blocks.0.ffn"
        if name.startswith(pattern.rstrip("*") + ".") or name == pattern.rstrip("*"):
            return True
    return False


def get_lora_target_modules(model: nn.Module, target_types: list[type] | None = None) -> list[str]:
    """Return qualified module names that are candidates for LoRA replacement.

    Args:
        model: The model to scan.
        target_types: Module types to target. Defaults to [nn.Linear, nn.Conv1d].

    Returns:
        List of fully-qualified module names (e.g., "dit_blocks.0.ffn.0").
    """
    if target_types is None:
        target_types = [nn.Linear, nn.Conv1d]

    targets = []
    for name, module in model.named_modules():
        if type(module) in target_types:
            targets.append(name)
    return targets


def apply_lora(model: nn.Module, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply LoRA adapters to a FullDuplexDiT model.

    Replaces target nn.Linear and nn.Conv1d modules with LoRA wrappers.
    Original weights are frozen; only LoRA A/B matrices are trainable.

    Args:
        model: The model to apply LoRA to.
        config: LoRA configuration with keys:
            - lora_rank (int, default 8): Rank of the low-rank decomposition.
            - lora_alpha (float, default 16.0): Scaling factor.
            - lora_dropout (float, default 0.0): Dropout probability on LoRA input.
            - target_modules (list[str] | None): Module name patterns to target.
              Examples: ["dit_blocks", "output_head"]. None = auto-detect all
              Linear/Conv1d in DiT blocks and output_head.

    Returns:
        Dict with applied LoRA info: count of replaced modules, trainable params, etc.
    """
    if config is None:
        config = {}

    rank = config.get("lora_rank", 8)
    alpha = config.get("lora_alpha", 16.0)
    dropout = config.get("lora_dropout", 0.0)
    target_patterns = config.get("target_modules", None)

    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Identify target modules
    all_linear_conv = get_lora_target_modules(model, [nn.Linear, nn.Conv1d])

    if target_patterns:
        # Filter by user-specified patterns
        target_names = [n for n in all_linear_conv if _match_module_name(n, target_patterns)]
    else:
        # Default: target all Linear/Conv1d in dit_blocks, output_head, and projection layers
        # Skip encoder internals (audio_encoder, visual_encoder, text_encoder)
        default_prefixes = ["dit_blocks.", "output_head.", "audio_proj.", "cross_proj."]
        target_names = [n for n in all_linear_conv if any(n.startswith(p) for p in default_prefixes)]

    replaced = {}
    for name in target_names:
        module = dict(model.named_modules())[name]

        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        elif isinstance(module, nn.Conv1d):
            lora_module = LoRAConv1d(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            continue

        # Replace module in model hierarchy
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)
        replaced[name] = type(module).__name__

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logger.info(
        f"LoRA applied: {len(replaced)} modules replaced, "
        f"{trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )

    # Store lora config on model for later save/load
    model._lora_config = {  # type: ignore[attr-defined]
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_patterns,
        "replaced_modules": list(replaced.keys()),
    }
    model._lora_applied = True  # type: ignore[attr-defined]

    return {
        "replaced_count": len(replaced),
        "replaced_modules": replaced,
        "trainable_params": trainable,
        "total_params": total,
    }


def remove_lora(model: nn.Module) -> dict[str, Any]:
    """Remove LoRA wrappers and restore original frozen modules.

    LoRA contributions are DISCARDED — only original weights remain.

    Returns:
        Dict with removal info.
    """
    if not getattr(model, "_lora_applied", False):
        logger.warning("No LoRA adapters found on model")
        return {"removed_count": 0}

    lora_config = getattr(model, "_lora_config", {})
    target_names = lora_config.get("replaced_modules", [])

    removed = 0
    for name in target_names:
        module = dict(model.named_modules())[name]
        if isinstance(module, LoRALinear):
            original_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
            )
            original_linear.weight = nn.Parameter(module.weight.data.clone(), requires_grad=False)
            if module.bias is not None:
                original_linear.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original_linear)
            removed += 1
        elif isinstance(module, LoRAConv1d):
            original_conv = nn.Conv1d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups,
                bias=module.bias is not None,
                device=module.weight.device,
            )
            original_conv.weight = nn.Parameter(module.weight.data.clone(), requires_grad=False)
            if module.bias is not None:
                original_conv.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original_conv)
            removed += 1

    model._lora_applied = False  # type: ignore[attr-defined]
    logger.info(f"LoRA removed: {removed} modules restored to original")

    return {"removed_count": removed}


def merge_lora(model: nn.Module) -> dict[str, Any]:
    """Merge LoRA weights into base weights (in-place).

    After merging, LoRA matrices are zeroed out. The model structure remains
    unchanged (LoRA wrappers still present but contribute nothing).
    Call this before deployment for optimized inference.

    Returns:
        Dict with merge info.
    """
    if not getattr(model, "_lora_applied", False):
        logger.warning("No LoRA adapters found on model")
        return {"merged_count": 0}

    lora_config = getattr(model, "_lora_config", {})
    target_names = lora_config.get("replaced_modules", [])

    merged = 0
    for name in target_names:
        module = dict(model.named_modules())[name]
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            module.merge_weights()
            merged += 1

    # After merge, all effective computation is in base weights
    # LoRA A/B are zeroed, so forward() has zero LoRA contribution
    logger.info(f"LoRA merged: {merged} modules (weights absorbed into base)")
    return {"merged_count": merged}


def save_lora(model: nn.Module, path: str | Path) -> None:
    """Save LoRA parameters (A/B matrices + config) to a .pt file.

    Only saves the LoRA-specific parameters, not the base model weights.
    This produces small files (~MB per character).

    Args:
        model: Model with LoRA applied.
        path: Output file path (e.g., "models/lora/kurisu.pt").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not getattr(model, "_lora_applied", False):
        raise ValueError("Model has no LoRA adapters applied. Call apply_lora() first.")

    lora_config = getattr(model, "_lora_config", {})
    lora_state = {}
    lora_config_copy = dict(lora_config)

    for name in lora_config.get("replaced_modules", []):
        module = dict(model.named_modules())[name]
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        elif isinstance(module, LoRAConv1d):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()

    save_dict = {
        "lora_config": lora_config_copy,
        "lora_state_dict": lora_state,
    }

    torch.save(save_dict, path)
    logger.info(f"LoRA adapter saved to {path} ({len(lora_state)} tensors)")


def load_lora(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load LoRA parameters from a .pt file into an existing model.

    The model must already have LoRA applied with matching configuration
    (same rank, alpha, target modules).

    Args:
        model: Model with LoRA already applied (via apply_lora).
        path: Path to LoRA .pt file.

    Returns:
        Dict with load info.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    lora_config = checkpoint.get("lora_config", {})
    lora_state = checkpoint.get("lora_state_dict", {})

    if not getattr(model, "_lora_applied", False):
        # Auto-apply LoRA with saved config before loading weights
        apply_lora(model, lora_config)

    loaded = 0
    for name in lora_config.get("replaced_modules", []):
        module = dict(model.named_modules())[name]
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in lora_state and b_key in lora_state:
            if isinstance(module, LoRALinear):
                module.lora_A.data.copy_(lora_state[a_key].to(module.lora_A.device))
                module.lora_B.data.copy_(lora_state[b_key].to(module.lora_B.device))
            elif isinstance(module, LoRAConv1d):
                module.lora_A.data.copy_(lora_state[a_key].to(module.lora_A.device))
                module.lora_B.data.copy_(lora_state[b_key].to(module.lora_B.device))
            loaded += 1

    logger.info(f"LoRA adapter loaded from {path} ({loaded} modules)")
    return {
        "loaded_count": loaded,
        "lora_config": lora_config,
    }


def get_lora_param_count(model: nn.Module) -> tuple[int, int]:
    """Count LoRA-specific trainable parameters.

    Returns:
        (trainable_lora_params, total_lora_params) — both A and B matrices.
    """
    trainable = 0
    total = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            total += module.lora_A.numel() + module.lora_B.numel()
            if module.lora_A.requires_grad:
                trainable += module.lora_A.numel() + module.lora_B.numel()
    return trainable, total