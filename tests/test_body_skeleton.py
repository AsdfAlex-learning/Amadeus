from __future__ import annotations

import numpy as np

from src.motion.preprocess.body_skeleton import BODY_PARAM_KEYS, BodySkeletonExtractor
from src.motion.preprocess.pipeline import PreprocessPipeline


class _FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = xy
        self.conf = conf


class _FakeResult:
    def __init__(self, keypoints: _FakeKeypoints) -> None:
        self.keypoints = keypoints


class _FakeModel:
    def __init__(self, keypoints: _FakeKeypoints) -> None:
        self.keypoints = keypoints
        self.calls: list[np.ndarray] = []

    def predict(self, frame: np.ndarray, verbose: bool = False, conf: float = 0.25):
        self.calls.append(frame)
        return [_FakeResult(self.keypoints)]


def _pose(conf_value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    xy[0, 0] = (320, 120)
    xy[0, 5] = (200, 200)
    xy[0, 6] = (400, 200)
    xy[0, 9] = (150, 300)
    xy[0, 10] = (450, 300)
    xy[0, 11] = (250, 400)
    xy[0, 12] = (350, 400)
    conf = np.full((1, 17), conf_value, dtype=np.float32)
    return xy, conf


def test_maps_pose_keypoints_to_body_params():
    xy, conf = _pose()
    model = _FakeModel(_FakeKeypoints(xy, conf))
    extractor = BodySkeletonExtractor(model=model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    out = extractor.process_frame(frame)

    assert out is not None
    assert set(out) == set(BODY_PARAM_KEYS)
    assert all(0.0 <= value <= 1.0 for value in out.values())
    assert abs(out["arm_l_x"] - (0.5 + (150 - 200) / 640)) < 1e-3
    assert abs(out["arm_l_y"] - (0.5 + (300 - 200) / 480)) < 1e-3
    assert abs(out["body_angle_x"] - 0.5) < 1e-3
    assert abs(out["body_angle_z"] - 0.5) < 1e-3
    assert model.calls


def test_missing_wrist_uses_neutral_arm():
    xy, conf = _pose()
    conf[0, 9] = 0.0
    model = _FakeModel(_FakeKeypoints(xy, conf))
    extractor = BodySkeletonExtractor(model=model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    out = extractor.process_frame(frame)

    assert out is not None
    assert out["arm_l_x"] == 0.5
    assert out["arm_l_y"] == 0.5


def test_returns_none_without_a_person():
    model = _FakeModel(
        _FakeKeypoints(
            np.zeros((0, 17, 2), dtype=np.float32),
            np.zeros((0, 17), dtype=np.float32),
        )
    )
    extractor = BodySkeletonExtractor(model=model)

    assert extractor.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_returns_none_when_unavailable():
    extractor = BodySkeletonExtractor(enabled=False)

    assert extractor.process_frame(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    out = extractor.process_video("missing.mp4")
    assert out["body_params"].shape == (0, len(BODY_PARAM_KEYS))


def test_invalid_frame_returns_none():
    model = _FakeModel(_FakeKeypoints(*_pose()))
    extractor = BodySkeletonExtractor(model=model)

    assert extractor.process_frame(None) is None
    assert extractor.process_frame(np.zeros((1, 1, 3), dtype=np.uint8)) is None


def test_align_body_params_pads_truncates_and_fills():
    assert PreprocessPipeline._align_body_params(None, 4).shape == (4, len(BODY_PARAM_KEYS))

    short = np.arange(14, dtype=np.float32).reshape(2, 7)
    padded = PreprocessPipeline._align_body_params(short, 4)
    assert padded.shape == (4, 7)
    assert np.array_equal(padded[0], short[0])
    assert np.array_equal(padded[3], short[1])

    long = np.arange(35, dtype=np.float32).reshape(5, 7)
    truncated = PreprocessPipeline._align_body_params(long, 4)
    assert np.array_equal(truncated, long[:4])
