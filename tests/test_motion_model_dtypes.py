"""Layer B: the fp16 (half-precision) path must actually run.

``config/default.yaml`` sets ``motion.use_fp16: true``, and
``DiffusionMotionInference.load_model()`` calls ``model.to_half()`` whenever the
device is not CPU. So this path is taken on *every* CUDA or MPS deployment.

A partially-converted model is not a degraded model - it fails at the first
forward pass with "expected scalar type Half but found Float". These tests pin
that down without needing real encoder weights (conftest forces the offline
stubs), because dtype consistency is independent of the weights.
"""

from __future__ import annotations

import torch

from src.motion.model import FullDuplexDiT

TINY = dict(
    num_params=45,
    hidden_dim=32,
    num_layers=1,
    use_gradient_checkpointing=False,
)


def _tiny_model() -> FullDuplexDiT:
    return FullDuplexDiT(**TINY)


def test_to_half_converts_every_parameter() -> None:
    """Half conversion must cover the whole model, not a hand-listed subset.

    ``input_proj`` was absent from the explicit conversion list, so it stayed
    float32 while every DiT block became float16 - the first forward pass then
    raised a dtype error. A hand-maintained list is the underlying defect, so
    this asserts the model as a whole rather than the one module.
    """
    model = _tiny_model().to_half()
    unconverted = [
        (name, str(param.dtype))
        for name, param in model.named_parameters()
        if param.dtype != torch.float16
    ]
    assert not unconverted, (
        f"to_half() left {len(unconverted)} parameter(s) unconverted: "
        f"{unconverted[:5]}"
    )


def test_half_model_forward_is_dtype_consistent() -> None:
    """A half model must accept half inputs and return half output.

    Catches dtype inconsistencies that are invisible in the parameter list -
    e.g. a submodule that computes in float32 internally and then feeds a
    float16 layer.
    """
    model = _tiny_model().to_half().eval()
    batch, frames = 1, 49
    with torch.no_grad():
        out = model(
            torch.zeros(batch, 16_000, dtype=torch.float16),
            torch.zeros(batch, 16_000, dtype=torch.float16),
            torch.zeros(batch, 5, 3, 224, 224, dtype=torch.float16),
            [""],
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, frames, 45, dtype=torch.float16),
        )
    assert out.dtype == torch.float16, f"expected half output, got {out.dtype}"
    assert out.shape == (batch, frames, 45)
