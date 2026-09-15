"""Runtime contract tests for the P0 defects in docs/PROJECT_REVIEW_REPORT.md.

Every test here documents a defect that is reachable from the real runtime
path. They are expected to be RED until the corresponding P0 fix lands; each
one states the contract in its docstring so the fix has an unambiguous target.

The tests deliberately stop at the model boundary (``_diffusion_infer``) rather
than mocking internals: the diffusion model is the component whose weights are
irrelevant to these contracts, and constructing it with stub encoders is fast
(<0.5 s), so replacing it keeps the tests hermetic without hiding the wiring
under test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

CHUNK_SAMPLES = 16_000  # 1 second at the audio pipeline's 16 kHz


# ── helpers ──────────────────────────────────────────────────────────────────


def _stubbed_engine(config: dict, infer: Callable[..., np.ndarray | None]):
    """Build a DiffusionMotionInference whose model boundary is replaced.

    ``process_user_audio`` / ``process_tts_audio`` gate on ``_loaded`` and
    ``_model`` before doing any work, so both are set to non-None sentinels.
    The sentinel model is never dereferenced because ``_diffusion_infer`` is
    the only code that would touch it.
    """
    from src.motion.inference import DiffusionMotionInference

    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = infer  # type: ignore[method-assign]
    return engine


def _capture_visual(config: dict, num_frames: int) -> np.ndarray:
    """Run one inference step and return the visual tensor handed to the model."""
    captured: dict[str, np.ndarray] = {}

    def _infer(_user_audio, _tts_audio, visual_frames):
        captured["visual"] = visual_frames
        return None

    engine = _stubbed_engine(config, _infer)
    # The camera hands on_frame exactly this: a (224, 224, 3) RGB float frame.
    for _ in range(num_frames):
        engine.process_visual_frame(np.zeros((224, 224, 3), dtype=np.float32))
    engine.process_user_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
    return captured["visual"]


# ── C-4: camera → model tensor layout ────────────────────────────────────────


def test_model_receives_channel_first_visual_frames(config: dict) -> None:
    """The camera→model boundary must be channel-first, matching training.

    ``CameraPerception`` emits HWC frames ``(224, 224, 3)``. The model's
    ``VisualEncoder`` unpacks ``(B, T, C, H, W)`` and training feeds
    ``(B, 5, 3, 224, 224)``. The visual tensor must therefore reach the model
    as ``(T, C, H, W)`` — otherwise the first conv sees 224 "channels".
    """
    visual = _capture_visual(config, num_frames=5)
    assert visual.shape == (5, 3, 224, 224), (
        f"model receives {visual.shape}; expected (T=5, C=3, H=224, W=224). "
        "Camera frames are HWC but the model stack is CHW."
    )


def test_visual_layout_agrees_with_and_without_camera(config: dict) -> None:
    """The no-camera and camera-present branches must not disagree.

    ``_prepare_visual_frames`` has two early paths: a zero-filled placeholder
    when no frames have arrived, and a stacked array once they have. Both feed
    the same conv stack, so both must produce the same layout.
    """
    without_camera = _capture_visual(config, num_frames=0)
    with_camera = _capture_visual(config, num_frames=5)
    assert with_camera.shape[1:] == without_camera.shape[1:], (
        f"no-camera path yields {without_camera.shape}, "
        f"camera path yields {with_camera.shape}; the two branches disagree."
    )


# ── C-5: TTS sample rate vs motion consumption rate ──────────────────────────


def test_one_second_of_tts_audio_occupies_one_second_of_motion_time(config: dict) -> None:
    """TTS audio must be buffered at the rate the motion engine consumes it.

    The engine consumes ``chunk_size`` seconds of audio per inference step,
    counting at ``config['audio']['sample_rate']``. TTS synthesises at
    ``config['tts']['sample_rate']``. If the two differ, the character's mouth
    moves at the wrong speed relative to the speech.
    """
    engine = _stubbed_engine(config, lambda *_a, **_k: None)

    tts_rate = int(config["tts"]["sample_rate"])
    motion_rate = int(engine.sample_rate)

    engine.process_tts_audio(np.zeros(tts_rate, dtype=np.float32))  # exactly 1 s of speech
    engine.process_user_audio(np.zeros(motion_rate, dtype=np.float32))  # exactly 1 s of motion

    remaining = len(engine._tts_buffer)
    assert remaining == 0, (
        f"{tts_rate} samples of TTS audio (1 s at the TTS rate) left {remaining} "
        f"samples un-consumed after 1 s of motion time at {motion_rate} Hz — "
        "TTS and the motion engine disagree on sample rate."
    )


# ── C-6: Speak state must be drivable by TTS audio alone ─────────────────────


def test_tts_audio_alone_drives_motion_inference(config: dict) -> None:
    """The engine must animate while speaking, not only while listening.

    Speak state is driven by TTS audio. If feeding TTS audio does not advance
    the model, motion depends entirely on the microphone callback — so the
    character freezes the instant the user stops talking.
    """
    steps: list[int] = []
    engine = _stubbed_engine(config, lambda *_a, **_k: (steps.append(1), None)[1])

    engine.process_tts_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))

    assert steps, (
        "feeding 1 s of TTS audio produced no inference step: Speak state cannot "
        "be driven by TTS, so motion depends entirely on microphone input."
    )


# ── C-3: resuming from a full snapshot ───────────────────────────────────────


@pytest.fixture
def tiny_preprocessed_dataset(tmp_path: Path) -> Path:
    """A 2-second 25 fps ``.npz`` sample plus its 16 kHz companion ``.wav``."""
    import soundfile as sf

    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    num_frames = 50  # 2 s at 25 fps
    np.savez(
        data_dir / "sample.npz",
        live2d_params=rng.random((num_frames, 45), dtype=np.float32),
        blendshapes=rng.random((num_frames, 52), dtype=np.float32),
        head_angles=rng.random((num_frames, 3), dtype=np.float32),
        bad_frames=np.array([], dtype=np.int64),
        fps=np.float32(25.0),
        duration_sec=np.float32(2.0),
        identity_id=np.int64(0),
        source_video="synthetic",
    )
    sf.write(data_dir / "sample.wav", np.zeros(32_000, dtype=np.float32), 16_000)
    return data_dir


_TINY_TRAIN_KWARGS = dict(
    num_params=45,
    hidden_dim=32,
    num_layers=1,
    device="cpu",
    use_amp=False,
    val_split=0.0,
    num_epochs=1,
)


def test_training_resumes_from_a_full_snapshot(
    tiny_preprocessed_dataset: Path, tmp_path: Path
) -> None:
    """A run must be resumable from the full-snapshot checkpoint it produced.

    README promises "resume is lossless". A full snapshot stores model +
    optimizer + scheduler + epoch, so ``train()`` must be able to consume it
    without raising.
    """
    from src.motion.training.train import train

    first_run = tmp_path / "run1"
    train(data_dir=tiny_preprocessed_dataset, output_dir=first_run, **_TINY_TRAIN_KWARGS)

    checkpoint = first_run / "full_duplex_dit.pt"
    assert checkpoint.exists(), (
        "the first run did not produce a full snapshot; cannot test resume"
    )

    # The behaviour under test: resuming must not raise.
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=tmp_path / "run2",
        resume_from=str(checkpoint),
        **_TINY_TRAIN_KWARGS,
    )


# ── C-8: persona performance parameters must reach the output ────────────────


def test_persona_performance_params_change_emitted_motion(
    config: dict, project_root: Path
) -> None:
    """``config['performance']`` must actually affect the emitted parameters.

    gesture_scale / expressiveness / react_speed / ... are documented as
    runtime persona knobs. If the post-processing engine is never wired into
    the inference path, changing them has no effect and they are dead config.
    """
    from src.config import load_config

    motion = np.full((5, 45), 0.9, dtype=np.float32)  # far from the 0.5 neutral

    def emitted(gesture_scale: float) -> np.ndarray:
        cfg = load_config(project_root / "config" / "default.yaml")
        cfg["performance"]["gesture_scale"] = gesture_scale
        engine = _stubbed_engine(cfg, lambda *_a, **_k: motion)
        frames: list[dict[str, float]] = []
        engine.on_params(frames.append)
        engine.process_user_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        assert frames, "no parameter frames were emitted"
        return np.stack([np.fromiter(f.values(), dtype=np.float32) for f in frames])

    assert not np.allclose(emitted(0.0), emitted(1.0)), (
        "performance.gesture_scale has no effect on the emitted Live2D parameters: "
        "PerformanceEngine is not wired into the inference path."
    )
