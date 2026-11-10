"""The camera→motion handoff must be synchronised, timestamped and rate-gated.

Three separate defects:

* `DiffusionMotionInference._visual_buffer` was a plain ``list`` mutated by the
  camera thread (``append``/``pop(0)``) and read by the motion worker
  (``np.stack`` over indices). A read during a prune raises ``IndexError``, or
  silently pairs shifted frames with the current audio.
* `_prepare_visual_frames` sampled across the whole buffer with
  ``range(0, num, step)``, biasing the model toward frames up to ~2 s old while
  pairing them with audio from now. With a 10-frame buffer it should take the
  **newest 5**.
* The camera loop published at the device rate (30 fps) while the model consumes
  5 frames per ~1 s step.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from src.runtime.concurrency import FrameBuffer

# ── FrameBuffer ──────────────────────────────────────────────────────────────


def test_snapshot_returns_the_newest_frames_oldest_first() -> None:
    buf = FrameBuffer(capacity=10)
    for value in (1.0, 2.0, 3.0):
        buf.put(np.full((2, 2, 3), value, dtype=np.float32))
    newest = buf.snapshot(2)
    assert [float(f[0, 0, 0]) for f in newest] == [2.0, 3.0]


def test_snapshot_is_empty_when_nothing_was_published() -> None:
    assert FrameBuffer(capacity=4).snapshot(5) == []


def test_buffer_is_bounded_and_counts_drops() -> None:
    buf = FrameBuffer(capacity=4)
    for i in range(7):
        buf.put(np.full(1, i, dtype=np.float32))
    assert len(buf) == 4
    assert buf.dropped == 3


def test_frames_are_evicted_by_age() -> None:
    """A stalled camera must not leave stale frames paired with current audio."""
    buf = FrameBuffer(capacity=10, max_age_s=1.0)
    buf.put(np.zeros(1, dtype=np.float32), timestamp=100.0)  # becomes 1.5 s old
    buf.put(np.ones(1, dtype=np.float32), timestamp=100.8)  # becomes 0.7 s old

    # At now=101.5 the first frame has expired and the second has not.
    fresh = buf.snapshot(5, now=101.5)
    assert len(fresh) == 1, f"expected only the fresh frame, got {len(fresh)}"
    assert float(fresh[0][0]) == 1.0


def test_snapshot_never_returns_more_than_available() -> None:
    buf = FrameBuffer(capacity=10)
    buf.put(np.zeros(1, dtype=np.float32))
    assert len(buf.snapshot(5)) == 1


def test_concurrent_writer_and_reader_do_not_race() -> None:
    """The original list raised IndexError (or shifted frames) under this load."""
    buf = FrameBuffer(capacity=10, max_age_s=60.0)
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for i in range(5_000):
                buf.put(np.full((4, 4, 3), float(i % 255), dtype=np.float32))
        except BaseException as e:  # noqa: BLE001 - recorded, then asserted
            errors.append(e)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                frames = buf.snapshot(5)
                for frame in frames:
                    assert frame.shape == (4, 4, 3)
        except BaseException as e:  # noqa: BLE001 - recorded, then asserted
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not errors, f"cross-thread access failed: {errors[:3]}"
    assert all(not t.is_alive() for t in threads)


# ── inference integration ────────────────────────────────────────────────────


def _captured_visual(config: dict, frames: list[np.ndarray]) -> np.ndarray:
    """Run one inference step and return the visual tensor handed to the model."""
    from src.motion.inference import DiffusionMotionInference

    captured: dict[str, np.ndarray] = {}

    def _infer(_user_audio, _tts_audio, visual_frames):
        captured["visual"] = visual_frames
        return None

    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = _infer  # type: ignore[method-assign]

    for frame in frames:
        engine.process_visual_frame(frame)
    engine.process_user_audio(np.zeros(16_000, dtype=np.float32))
    return captured["visual"]


def test_prepare_visual_frames_uses_the_newest_frames(config: dict) -> None:
    """With 10 frames buffered, the newest 5 must be the ones sent to the model.

    The old sampler spread its picks across the entire buffer, so with a 10-frame
    buffer the oldest pick was ~2 s stale.
    """
    frames = [np.full((224, 224, 3), i / 10.0, dtype=np.float32) for i in range(10)]
    visual = _captured_visual(config, frames)

    assert visual.shape == (5, 3, 224, 224)
    # Newest five are 0.5 .. 0.9, oldest first.
    values = [float(visual[i][0, 0, 0]) for i in range(5)]
    assert values == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9]), (
        f"expected the newest 5 frames, got {values}"
    )


def test_prepare_visual_frames_pads_by_repeating_the_newest(config: dict) -> None:
    """With fewer than 5 frames the newest must be repeated, not an old frame."""
    frames = [np.full((224, 224, 3), 0.2, dtype=np.float32)]
    visual = _captured_visual(config, frames)

    assert visual.shape == (5, 3, 224, 224)
    assert visual == pytest.approx(np.full((5, 3, 224, 224), 0.2))


# ── camera rate gate ─────────────────────────────────────────────────────────


def test_camera_publishes_at_the_configured_rate(config: dict) -> None:
    from src.perception.camera import CameraPerception

    cam = CameraPerception(config)
    try:
        cam.frame_fps = 4.0
        cam._last_published_at = None
        # First call always publishes, then one per 250 ms.
        assert cam._rate_gate(1000.0) is True
        assert cam._rate_gate(1000.1) is False, "published twice inside one interval"
        assert cam._rate_gate(1000.24) is False
        assert cam._rate_gate(1000.25) is True
        assert cam._rate_gate(1000.5) is True
    finally:
        cam.cleanup()


def test_camera_rate_gate_can_be_disabled(config: dict) -> None:
    from src.perception.camera import CameraPerception

    cam = CameraPerception(config)
    try:
        cam.frame_fps = 0.0
        assert cam._rate_gate(1.0) is True
        assert cam._rate_gate(1.0) is True
    finally:
        cam.cleanup()
