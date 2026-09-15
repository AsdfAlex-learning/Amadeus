"""The render drain must consume at the model's rate, not reset to neutral.

Two independent defects lived in `Live2DWidget`:

* `paintGL` popped one frame per paint and reset to `_default_params` on
  underflow. At `update_interval: 0.016` that is ~62.5 consumes/s against ~50
  produced/s, so the underflow is structural and the character snapped to a
  neutral pose between chunks.
* Draining inside `paintGL` stops entirely when the widget is hidden or occluded.

These tests drive the drain with an injected clock, so they assert the policy
rather than Qt's timer behaviour.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def widget(config: dict, qapp):
    """A Live2DWidget with no GL surface.

    `initializeGL` is never invoked, so `renderer` stays None and nothing touches
    OpenGL; the drain policy under test is independent of rendering.
    """
    from src.app.live2d_widget import Live2DWidget

    w = Live2DWidget(config)
    # The drain timers are started in initializeGL, which we skip.
    assert w.renderer is None
    yield w
    w.cleanup()


def _frame(mouth: float) -> dict[str, float]:
    return {"param_14": mouth}


def test_drain_applies_the_last_frame_popped(widget) -> None:
    """Consuming several frames in one tick must not teleport to queue[-1].

    Applying the *last popped* frame is sequential consumption; applying the
    newest queued frame would skip the intermediate poses entirely.
    """
    for mouth in (0.1, 0.2, 0.3):
        widget.push_params(_frame(mouth))

    # 60 ms at 50 fps -> 3 frames of credit, so all three are consumed.
    widget._drain(now=1000.0)
    widget._drain(now=1000.1)  # 100 ms -> 5 frames of credit (>= the 3 queued)

    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.3)
    assert widget.queued_frames == 0


def test_drain_is_rate_matched(widget) -> None:
    """A sub-frame interval must not consume a whole frame."""
    widget.push_params(_frame(0.9))
    widget._drain(now=2000.0)
    widget._drain(now=2000.005)  # 5 ms -> 0.25 frames of credit

    # Nothing applied yet beyond the initial neutral pose.
    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.0)
    assert widget.queued_frames == 1


def test_underflow_holds_the_last_applied_pose(widget) -> None:
    """A brief gap between bursts must not reset the character to neutral."""
    widget.push_params(_frame(0.8))
    widget._drain(now=3000.0)
    widget._drain(now=3000.1)  # consume the frame
    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.8)

    widget._drain(now=3000.15)  # 50 ms later: still within the idle timeout
    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.8), (
        "a short underflow reset the pose; it must hold the last applied frame"
    )


def test_sustained_underflow_decays_to_the_neutral_pose(widget) -> None:
    """Holding the final speaking frame forever would freeze the character.

    It must decay to neutral once the underflow outlasts the idle timeout.
    """
    widget.push_params(_frame(0.8))
    widget._drain(now=4000.0)
    widget._drain(now=4000.1)
    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.8)

    widget._drain(now=4000.1 + widget.idle_timeout_ms / 1000.0 + 0.01)
    assert widget.applied_params["ParamMouthOpenY"] == pytest.approx(0.0), (
        "the character stayed frozen on its last speaking pose during idle"
    )


def test_buffer_is_bounded_and_counts_drops(widget) -> None:
    """Overflow must be bounded *and* visible, not silent."""
    for i in range(widget.capacity + 7):
        widget.push_params(_frame(i / 1000.0))

    assert widget.queued_frames == widget.capacity
    assert widget.dropped_frames == 7, (
        f"expected 7 counted drops, got {widget.dropped_frames}"
    )


def test_parameter_names_are_mapped_to_live2d(widget) -> None:
    """Generic model indices must arrive as Cubism parameter names."""
    widget.push_params({"param_14": 0.42, "param_6": 0.9})
    widget.push_params(_frame(0.5))  # second frame supplies the both-frame interval
    widget._drain(now=5000.0)
    widget._drain(now=5000.1)

    assert "ParamMouthOpenY" in widget.applied_params
    assert "param_14" not in widget.applied_params


def test_credit_is_clamped_after_a_long_stall(widget) -> None:
    """A long pause must not accumulate unlimited credit.

    Accumulated credit is clamped to the buffer capacity, so a stall cannot cause
    the drain to consume more frames than could ever be queued.
    """
    widget.push_params(_frame(0.7))
    widget._drain(now=6000.0)
    widget._drain(now=6000.0 + 3600.0)  # an hour of accumulated credit

    assert widget._credit <= float(widget.capacity) + 1e-6
