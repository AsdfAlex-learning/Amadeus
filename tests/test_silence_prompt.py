from __future__ import annotations

import random
import time

from src.motion.silence_prompt import SilenceContext, SilencePromptGenerator


def _config(**overrides):
    prompt = {
        "enabled": True,
        "min_interval_seconds": 0.0,
        "context_chars": 40,
        "micro_behaviors": ["slowly blink"],
    }
    prompt.update(overrides)
    return {"motion": {"silence_prompt": prompt}}


def _generator(clock=lambda: 100.0, **overrides):
    return SilencePromptGenerator(_config(**overrides), rng=random.Random(7), clock=clock)


def test_describes_quiet_duration_and_behavior():
    prompt = _generator().generate(SilenceContext(seconds_since_interaction=15.0))
    assert "15" in prompt
    assert "quiet" in prompt.lower()
    assert "slowly blink" in prompt


def test_includes_truncated_exchange():
    prompt = _generator().generate(
        SilenceContext(
            seconds_since_interaction=15.0,
            last_user_text="hello " * 30,
            last_assistant_text="hi there",
        )
    )
    assert "Last exchange" in prompt
    assert "hello" in prompt
    assert len(prompt) < 400


def test_describes_perception():
    perception = {
        "face_detected": True,
        "gaze_direction": "center",
        "smile": 0.9,
        "brow_raise": 0.8,
        "eye_openness_l": 0.9,
        "eye_openness_r": 0.9,
    }
    prompt = _generator().generate(SilenceContext(20.0, perception=perception))
    assert "smil" in prompt.lower()
    assert "curious" in prompt.lower()


def test_describes_missing_face():
    prompt = _generator().generate(SilenceContext(20.0, perception={"face_detected": False}))
    assert "No face" in prompt


def test_min_interval_reuses_prompt():
    now = [100.0]
    generator = _generator(clock=lambda: now[0], min_interval_seconds=10.0)
    first = generator.generate(SilenceContext(15.0))
    now[0] = 105.0
    assert generator.generate(SilenceContext(20.0)) == first
    now[0] = 111.0
    assert generator.generate(SilenceContext(26.0)) != first


def test_disabled_returns_empty():
    assert _generator(enabled=False).generate(SilenceContext(15.0)) == ""


def test_emotion_is_included():
    prompt = _generator().generate(
        SilenceContext(15.0, character_emotion={"happiness": 0.7, "sadness": 0.1})
    )
    assert "happiness" in prompt


def test_window_wires_generated_prompt_into_motion_engine():
    from src.app.window import AmadeusWindow
    from src.dialogue.context import ConversationContext

    class _Motion:
        def __init__(self):
            self.prompts = []

        def set_silence_prompt(self, prompt):
            self.prompts.append(prompt)

    class _Camera:
        def latest_perception(self):
            return {"face_detected": True, "gaze_direction": "center"}

    class _FakeWindow:
        _shutting_down = False
        context = ConversationContext()
        camera = _Camera()
        motion_model = _Motion()
        silence_prompt_generator = _generator()
        _last_interaction_at = time.monotonic() - 15.0
        dialogue_model = object()

    fake = _FakeWindow()
    fake.context.add_user_message("how was your day?")
    AmadeusWindow._update_silence_prompt(fake)

    assert fake.motion_model.prompts
    prompt = fake.motion_model.prompts[0]
    assert "quiet" in prompt.lower()
    assert "how was your day?" in prompt
