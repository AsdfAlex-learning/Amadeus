from __future__ import annotations

import copy

import numpy as np

from src.motion.baseline import Phase0Baseline


def _config(**overrides):
    baseline = {
        "enabled": True,
        "fps": 50.0,
        "seed": 1,
        "expression_smoothing": 0.0,
        "blink_interval_seconds": [100.0, 100.0],
    }
    baseline.update(overrides)
    return {"motion": {"baseline": baseline}}


def _sine(frequency: float, amplitude: float, sample_rate: int = 16000) -> np.ndarray:
    times = np.arange(sample_rate) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def test_shape_and_bounds():
    baseline = Phase0Baseline(_config(), sample_rate=16000)
    out = baseline.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    assert out.shape == (50, 45)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_mouth_follows_audio_energy():
    baseline = Phase0Baseline(_config(), sample_rate=16000)
    silent = baseline.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    spoken = baseline.generate(
        _sine(220.0, 0.5),
        np.zeros(0, dtype=np.float32),
        mode="listen",
        num_frames=50,
    )
    assert spoken[:, 14].mean() > silent[:, 14].mean() + 0.05
    assert spoken[:, 14].max() > 0.2


def test_tts_drives_speak_mouth():
    baseline = Phase0Baseline(_config(), sample_rate=16000)
    out = baseline.generate(
        np.zeros(0, dtype=np.float32),
        _sine(180.0, 0.6),
        mode="speak",
        num_frames=50,
    )
    assert out[:, 14].max() > 0.2


def test_blink_happens_and_eyes_reopen():
    baseline = Phase0Baseline(
        _config(blink_interval_seconds=[0.1, 0.1], blink_duration_seconds=0.2),
        sample_rate=16000,
    )
    out = baseline.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    assert out[:, 6].min() < 0.5
    assert out[:, 6].max() > 0.9


def test_breathing_varies():
    baseline = Phase0Baseline(_config(), sample_rate=16000)
    out = baseline.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    assert float(np.std(out[:, 17])) > 0.01


def test_deterministic_with_seed():
    first = Phase0Baseline(_config(seed=42), sample_rate=16000)
    second = Phase0Baseline(_config(seed=42), sample_rate=16000)
    left = first.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    right = second.generate(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        mode="silence",
        num_frames=50,
    )
    assert np.array_equal(left, right)


def test_engine_falls_back_without_checkpoint(config, tmp_path):
    from src.motion.inference import DiffusionMotionInference

    cfg = copy.deepcopy(config)
    cfg["motion"]["model_path"] = str(tmp_path)
    cfg["motion"]["baseline"] = {
        "enabled": True,
        "seed": 1,
        "expression_smoothing": 0.0,
    }
    engine = DiffusionMotionInference(cfg)
    try:
        assert engine.load_model() is True
        assert engine._use_baseline is True
        out = engine._diffusion_infer(
            np.zeros(16000, dtype=np.float32),
            _sine(180.0, 0.5),
            np.zeros((5, 3, 224, 224), dtype=np.float32),
        )
        assert out is not None
        assert out.shape == (50, 45)
    finally:
        engine.cleanup()
