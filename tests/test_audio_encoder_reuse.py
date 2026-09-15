"""Inference must not encode the same waveform twice per diffusion step.

``_diffusion_infer`` encoded ``user_audio`` once purely to discover the encoder's
output length (T) before sizing the latent, and ``FullDuplexDiT.forward`` then
encoded the identical waveform again. Hubert is the most expensive part of a step,
so this was a full extra encoder pass per DDIM step for no benefit — roughly a 2x
cost on the audio path the whole threading design is organised around.
"""

from __future__ import annotations

import numpy as np

from src.motion.inference import DiffusionMotionInference

CHUNK = 16_000


def _engine_with_real_model(config: dict) -> tuple[DiffusionMotionInference, list[int]]:
    """An engine whose model is a real (stub-encoder) FullDuplexDiT.

    Returns the engine plus a mutable counter of audio-encoder invocations.
    """
    engine = DiffusionMotionInference(config)
    model = engine._model if engine._model is not None else None
    if model is None:
        from src.motion.model import FullDuplexDiT

        model = FullDuplexDiT(num_params=45, hidden_dim=32, num_layers=1,
                             use_gradient_checkpointing=False)
    model.eval()
    engine._model = model
    engine._loaded = True
    engine._device = __import__("torch").device("cpu")

    calls: list[int] = []
    original = model.audio_encoder.forward

    def counting_forward(waveform):
        calls.append(1)
        return original(waveform)

    model.audio_encoder.forward = counting_forward  # type: ignore[method-assign]
    return engine, calls


def test_each_diffusion_step_encodes_user_audio_once(config: dict) -> None:
    """With one DDIM step, the user waveform must be encoded exactly once.

    The forward pass also encodes tts_audio, so the expected total is two: one for
    the user waveform (shared with inference) and one for the TTS waveform.
    """
    engine, calls = _engine_with_real_model(config)
    engine.num_inference_steps = 1
    engine._diffusion_infer(
        np.zeros(CHUNK, dtype=np.float32),
        np.zeros(CHUNK, dtype=np.float32),
        np.zeros((5, 3, 224, 224), dtype=np.float32),
    )

    assert calls, "the audio encoder was never invoked"
    assert len(calls) == 2, (
        f"audio encoder invoked {len(calls)} times for a single DDIM step; "
        f"expected 2 (user waveform once, TTS waveform once). More than 2 means the "
        f"user waveform is being encoded redundantly"
    )


def test_listen_features_can_be_supplied_by_the_caller(config: dict) -> None:
    """forward() must accept a precomputed listen encoding and skip its own."""
    import torch

    from src.motion.model import FullDuplexDiT

    model = FullDuplexDiT(
        num_params=45, hidden_dim=32, num_layers=1, use_gradient_checkpointing=False
    ).eval()

    calls: list[int] = []
    original = model.audio_encoder.forward

    def counting_forward(waveform):
        calls.append(1)
        return original(waveform)

    model.audio_encoder.forward = counting_forward  # type: ignore[method-assign]

    batch, frames = 1, 49
    with torch.no_grad():
        listen_feat = model.audio_proj(
            model.audio_encoder(torch.zeros(batch, CHUNK))
        )
        calls.clear()
        model(
            torch.zeros(batch, CHUNK),
            torch.zeros(batch, CHUNK),
            torch.zeros(batch, 5, 3, 224, 224),
            [""],
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, frames, 45),
            listen_feat=listen_feat,
        )

    assert len(calls) == 1, (
        f"audio encoder invoked {len(calls)} times when listen_feat was supplied; "
        f"expected only the TTS encode"
    )
