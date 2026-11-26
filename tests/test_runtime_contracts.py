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
    """TTS audio must be resampled to the rate the motion engine consumes it at.

    The engine consumes ``chunk_size`` seconds per inference step counting at
    ``config['audio']['sample_rate']``, while TTS synthesises at
    ``config['tts']['sample_rate']``. If the two differ, the character's mouth
    moves at the wrong speed relative to the speech.

    Playback is attached with nothing played and no lookahead, so the utterance is
    retained un-consumed and its resampled length can be measured directly.
    """

    class _IdlePlayback:
        played_samples = 0

    engine = _stubbed_engine(config, lambda *_a, **_k: None)
    engine.attach_playback(_IdlePlayback(), playback_rate=int(config["tts"]["sample_rate"]))
    engine.SPEECH_LOOKAHEAD_SECONDS = 0.0  # nothing may be inferred yet

    tts_rate = int(config["tts"]["sample_rate"])
    motion_rate = int(engine.sample_rate)
    engine.process_tts_audio(np.zeros(tts_rate, dtype=np.float32))  # exactly 1 s of speech

    assert engine._speech is not None, "the utterance was not retained for the Speak state"
    _stream_id, audio = engine._speech
    assert audio.size == motion_rate, (
        f"{tts_rate} samples of TTS audio (1 s at the TTS rate) became {audio.size} "
        f"samples at the pipeline rate {motion_rate} Hz — TTS and the motion engine "
        "disagree on sample rate."
    )


# ── C-6: Speak state must be drivable by TTS audio alone ─────────────────────


def test_tts_audio_alone_drives_motion_inference(config: dict) -> None:
    """The engine must animate while speaking, not only while listening.

    Speak state is driven by TTS audio. If feeding TTS audio does not advance
    the model, motion depends entirely on the microphone callback — so the
    character freezes the instant the user stops talking.

    Note this feeds one second *at the TTS rate*, not ``CHUNK_SAMPLES``: TTS
    audio is resampled to the pipeline rate on ingest (C-5), so 16000 samples
    declared as 22050 Hz audio is only ~0.73 s of speech.
    """
    steps: list[int] = []
    engine = _stubbed_engine(config, lambda *_a, **_k: (steps.append(1), None)[1])

    one_second_of_tts = int(config["tts"]["sample_rate"])
    engine.process_tts_audio(np.zeros(one_second_of_tts, dtype=np.float32))

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
    # With the default of 4 the 1-sample dataset never fills an accumulation
    # group, so a run would complete without a single optimizer step.
    grad_accum_steps=1,
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
    assert checkpoint.exists(), "the first run did not produce a full snapshot; cannot test resume"

    # The behaviour under test: resuming must not raise.
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=tmp_path / "run2",
        resume_from=str(checkpoint),
        **_TINY_TRAIN_KWARGS,
    )


def test_the_final_snapshot_records_the_last_completed_epoch(
    tiny_preprocessed_dataset: Path, tmp_path: Path
) -> None:
    """Resuming must continue the schedule, not skip past its end.

    The snapshot's ``epoch`` is the resume cursor (``start_epoch = epoch + 1``).
    Recording ``num_epochs`` instead of the last completed epoch makes a
    resumed early-stopped run start past the end of the schedule - silently
    discarding every remaining epoch - and makes extending a finished run skip
    one epoch.
    """
    import torch

    from src.motion.training.train import train

    run = tmp_path / "run"
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=run,
        **{**_TINY_TRAIN_KWARGS, "num_epochs": 2},
    )

    checkpoint = run / "full_duplex_dit.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1, (
        f"the final snapshot claims epoch {payload['epoch']}; a 2-epoch run completed "
        f"epochs 0..1 and resume must start at 2, not past the end of the schedule"
    )


def test_gradient_accumulation_does_not_drop_the_tail_of_each_epoch(
    tiny_preprocessed_dataset: Path, tmp_path: Path, monkeypatch
) -> None:
    """Batches that never fill an accumulation group must still train.

    The step ran only on ``(batch_idx + 1) % grad_accum_steps == 0``, so the
    leftover batches of every epoch were discarded: their gradients were
    zeroed by the next epoch's ``zero_grad()`` without ever stepping. With the
    training guide's own config (270 train batches, accumulation of 4) that
    silently dropped two batches of every epoch.
    """
    import torch

    from src.motion.training.train import train

    steps = 0
    original_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    # The 1-sample dataset gives 1 batch per epoch; a group of 2 makes it a tail.
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=tmp_path / "run",
        **{**_TINY_TRAIN_KWARGS, "grad_accum_steps": 2},
    )
    assert steps == 1, (
        f"an epoch of batches that never filled an accumulation group produced "
        f"{steps} optimizer steps; the tail group must not be dropped"
    )


def test_sample_every_renders_a_gif_per_interval(
    tiny_preprocessed_dataset: Path, tmp_path: Path
) -> None:
    """``--sample_every`` is advertised as the way to watch learning happen.

    The flag existed and its help promised "Save a sample GIF every N epochs",
    but nothing read it: a 200-epoch overnight run produced zero samples and
    no error.
    """
    from src.motion.training.train import train

    run = tmp_path / "run"
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=run,
        **{**_TINY_TRAIN_KWARGS, "num_epochs": 2, "sample_every": 1},
    )

    samples = (
        sorted((run / "samples").glob("epoch_*.gif")) if (run / "samples").exists() else []
    )
    assert [p.name for p in samples] == ["epoch_0001.gif", "epoch_0002.gif"], (
        f"sample_every=1 over 2 epochs should render one GIF per epoch; "
        f"got {[p.name for p in samples]}"
    )
    assert all(p.stat().st_size > 0 for p in samples), "a rendered sample GIF is empty"


def test_an_early_stopped_run_returns_the_best_validation_weights_not_the_last(
    tiny_preprocessed_dataset: Path, tmp_path: Path, monkeypatch
) -> None:
    """An early-stopped run must save the best-val epoch's weights.

    ``best_val_loss`` was tracked but nothing kept its weights: the final save
    wrote whatever the loop happened to stop on, so a weekend run could return
    a model worse than the best epoch it flew past - defeating the feature.
    """
    import soundfile as sf
    import torch

    from src.motion.training.train import train

    # The 1-sample fixture cannot split; add a second clip for 1 train / 1 val.
    rng = np.random.default_rng(1)
    np.savez(
        tiny_preprocessed_dataset / "sample2.npz",
        live2d_params=rng.random((50, 45), dtype=np.float32),
        blendshapes=rng.random((50, 52), dtype=np.float32),
        head_angles=rng.random((50, 3), dtype=np.float32),
        bad_frames=np.array([], dtype=np.int64),
        fps=np.float32(25.0),
        duration_sec=np.float32(2.0),
        identity_id=np.int64(0),
        source_video="synthetic",
    )
    sf.write(tiny_preprocessed_dataset / "sample2.wav", np.zeros(32_000, dtype=np.float32), 16_000)

    # Val losses per epoch: 1.0 (best), 0.5 (best), 0.9 (patience fires).
    scripted_vals = iter([1.0, 0.5, 0.9])

    class _ScriptedLoss:
        def __call__(self, pred, target):
            value = 0.0 if torch.is_grad_enabled() else next(scripted_vals)
            return (pred * 0).sum() + value

    # Every optimizer step stamps the first parameter tensor with
    # 100 + step_count, so the saved weights encode which epoch they are from.
    original_step = torch.optim.AdamW.step
    count = {"n": 0}

    def stamping_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        count["n"] += 1
        self.param_groups[0]["params"][0].data.fill_(100.0 + count["n"])
        return result

    monkeypatch.setattr(torch.nn, "MSELoss", lambda: _ScriptedLoss())
    monkeypatch.setattr(torch.optim.AdamW, "step", stamping_step)

    run = tmp_path / "run"
    train(
        data_dir=tiny_preprocessed_dataset,
        output_dir=run,
        **{
            **_TINY_TRAIN_KWARGS,
            "num_epochs": 5,
            "val_split": 0.5,
            "early_stopping_patience": 1,
        },
    )

    payload = torch.load(run / "full_duplex_dit.pt", map_location="cpu", weights_only=False)
    values = [t for t in payload["model"].values() if torch.is_tensor(t)]

    def has_marker(value: float) -> bool:
        return any(bool(torch.all(t == value)) for t in values)

    assert has_marker(102.0), (
        "the final snapshot does not carry the best-validation epoch's weights "
        "(epoch 2 of the scripted 1.0/0.5/0.9 run)"
    )
    assert not has_marker(103.0), (
        "the final snapshot carries the weights of the last epoch trained, not "
        "the best-validation one: early stopping returned a worse model"
    )


def test_motion_to_gif_renders_low_parameter_values(tmp_path: Path) -> None:
    """A bar for a small parameter value must render, not crash the GIF writer.

    The bar rectangle was drawn as [x0, H*(1-v), x1, H-30]: any value below
    about 0.09 put the top edge below the bottom edge, and Pillow raises
    "y1 must be greater than or equal to y0".
    """
    from src.motion.training.visualize import motion_to_gif

    out = tmp_path / "low.gif"
    motion_to_gif(np.zeros((3, 45), dtype=np.float32), out)
    assert out.exists() and out.stat().st_size > 0, "the GIF was not rendered"


# ── C-8: persona performance parameters must reach the output ────────────────


def test_persona_performance_params_change_emitted_motion(config: dict, project_root: Path) -> None:
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


def test_listen_step_consumes_exactly_one_chunk(config: dict) -> None:
    """Each Listen step must consume exactly one chunk, leaving no residue.

    The removed "overlap" code popped `keep_n` samples from the front of the
    buffer and re-inserted them in the same order: a no-op that neither carried
    audio across chunks nor left the buffer short, while implying continuity that
    did not exist. This guards against a half-implemented overlap being
    reintroduced, which would silently change how much audio a step consumes.
    """
    seen: list[int] = []

    def _infer(user_chunk, _tts_chunk, _visual):
        seen.append(int(user_chunk.size))
        return None

    engine = _stubbed_engine(config, _infer)
    for _ in range(3):
        engine._audio_buffer.extend([0.1] * CHUNK_SAMPLES)

    engine.process_user_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))

    assert seen, "no inference step ran"
    assert all(size == CHUNK_SAMPLES for size in seen), (
        f"a step consumed an unexpected amount of audio: {seen}"
    )
    assert len(seen) == 4, f"expected 4 consuming steps, got {len(seen)}"
    assert len(engine._audio_buffer) == 0, (
        f"{len(engine._audio_buffer)} samples were left over after draining four chunks"
    )


# ── C-8: the category mapping must come from the mapping file ────────────────


def test_the_default_mapping_file_loads_instead_of_falling_back_to_the_heuristic(
    project_root: Path,
) -> None:
    """The shipped mapping must drive the categories, not the fallback.

    ``_load_mapping`` expected a ``params``/``target_index`` schema the shipped
    YAML does not have (it names the parameters in output order under
    ``live2d_param_order``), so every run fell back to a heuristic whose
    categories contradict the real 45-slot order: head/body angles were scaled
    and clipped as mouth (halving head motion), the mouth scaled as eyes, and
    the tear/blush block as head.
    """
    from src.motion.performance import PerformanceEngine

    engine = PerformanceEngine(
        mapping_path=project_root / "src" / "motion" / "preprocess" / "mappings" / "default.yaml"
    )
    assert engine._param_indices, "the mapping file did not load; the heuristic fallback ran"
    assert sorted(engine._head_indices) == [0, 1, 2, 3, 4, 5], (
        f"head/body angles live in slots 0-5, classified as {sorted(engine._head_indices)}"
    )
    assert 14 in engine._mouth_indices and 15 in engine._mouth_indices, (
        f"ParamMouthOpenY/ParamMouthForm live in slots 14-15, classified as "
        f"{sorted(engine._mouth_indices)}"
    )
    for slot in (0, 1, 2, 3, 4, 5):
        assert slot not in engine._mouth_indices, (
            f"head/body angle slot {slot} is classified as mouth; it would be "
            f"clipped to mouth_open_max"
        )
    for slot in (6, 7, 10, 13):  # eyes and brows
        assert slot in engine._eye_indices, f"eye/brow slot {slot} is not classified as eye"


def test_the_heuristic_fallback_categories_match_the_real_parameter_order() -> None:
    """The fallback must mirror the default 45-slot order it stands in for.

    It claimed "first 6: mouth-related", but slots 0-5 are the head/body
    angles and the mouth lives at 14-15.
    """
    from src.motion.performance import PerformanceEngine

    engine = PerformanceEngine()
    assert sorted(engine._mouth_indices) == [14, 15]
    assert sorted(engine._head_indices) == [0, 1, 2, 3, 4, 5]
    assert sorted(engine._eye_indices) == [6, 7, 8, 9, 10, 11, 12, 13]
