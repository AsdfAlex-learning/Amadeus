"""Generate the text prompt used by the Silence state.

The Performance Engine is always-on, but the text prompt for an idle step is not
the last thing the user said. It is a description of the current social context:
how long the user has been quiet, whether they are still visible, what the last
exchange was about, and a small idle behaviour to keep the character alive.

This module is deliberately pure: it takes a snapshot (``SilenceContext``) and
returns a string. The window owns the snapshot and calls it on a timer; the
motion engine only receives the resulting prompt through
``set_silence_prompt()``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SilenceContext:
    """Snapshot of the state used to describe a Silence-state step."""

    seconds_since_interaction: float
    last_user_text: str | None = None
    last_assistant_text: str | None = None
    perception: dict[str, Any] | None = None
    character_emotion: dict[str, float] | None = None


class SilencePromptGenerator:
    """Build a short, human-readable prompt for idle motion.

    Config (``motion.silence_prompt``):

    - ``enabled`` (default ``True``)
    - ``min_interval_seconds`` (default ``2.0``): reuse the previous prompt while
      it is still fresh, so the model does not see a new description every tick
    - ``context_chars`` (default ``120``): truncation limit for quoted messages
    - ``micro_behaviors``: list of small actions, one is chosen per prompt
    - ``update_interval_ms``: used by the window timer, not by this class
    """

    DEFAULT_MICRO_BEHAVIORS = (
        "slightly tilt the head",
        "slowly blink",
        "shift gaze toward the user",
        "take a small breath",
        "relax the shoulders",
    )

    def __init__(
        self,
        config: dict | None = None,
        *,
        rng: random.Random | None = None,
        clock=time.monotonic,
    ) -> None:
        cfg: dict[str, Any] = {}
        if isinstance(config, dict):
            cfg = (config.get("motion", {}) or {}).get("silence_prompt", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_interval_s = float(cfg.get("min_interval_seconds", 2.0))
        self.context_chars = max(16, int(cfg.get("context_chars", 120)))
        behaviors = cfg.get("micro_behaviors")
        if not isinstance(behaviors, (list, tuple)) or not behaviors:
            behaviors = self.DEFAULT_MICRO_BEHAVIORS
        self._behaviors = [str(b).strip() for b in behaviors if str(b).strip()]
        self._rng = rng or random.Random()
        self._clock = clock
        self._last_at: float | None = None
        self._last_prompt = ""

    def generate(self, context: SilenceContext) -> str:
        """Return the prompt for this idle step (possibly the cached one)."""
        if not self.enabled:
            return ""

        now = self._clock()
        if self._last_at is not None and now - self._last_at < self.min_interval_s:
            return self._last_prompt

        parts = [self._quiet_phrase(context.seconds_since_interaction)]
        perception = self._perception_phrase(context.perception)
        if perception:
            parts.append(perception)
        exchange = self._exchange_phrase(context.last_user_text, context.last_assistant_text)
        if exchange:
            parts.append(exchange)
        emotion = self._emotion_phrase(context.character_emotion)
        if emotion:
            parts.append(emotion)
        if self._behaviors:
            parts.append(f"Subtle idle behavior: {self._rng.choice(self._behaviors)}.")

        prompt = " ".join(part for part in parts if part).strip()
        self._last_at = now
        self._last_prompt = prompt
        return prompt

    @staticmethod
    def _quiet_phrase(seconds_since_interaction: float) -> str:
        seconds = max(0.0, float(seconds_since_interaction))
        if seconds < 5:
            return "The user has just paused for a moment."
        if seconds < 15:
            return f"The user has been quiet for {int(seconds)} seconds."
        if seconds < 60:
            return f"The user has been quiet for about {int(seconds // 5) * 5} seconds."
        if seconds < 300:
            return "The user has been quiet for a few minutes."
        return "The user has been quiet for a long while."

    def _truncate(self, text: str) -> str:
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= self.context_chars:
            return cleaned
        return cleaned[: self.context_chars - 3].rstrip() + "..."

    def _exchange_phrase(self, user_text: str | None, assistant_text: str | None) -> str:
        user = self._truncate(user_text) if user_text else ""
        assistant = self._truncate(assistant_text) if assistant_text else ""
        if user and assistant:
            return (
                f'Last exchange: the user said "{user}", and the character replied "{assistant}".'
            )
        if user:
            return f'Last user message: "{user}".'
        if assistant:
            return f'Last character reply: "{assistant}".'
        return ""

    @staticmethod
    def _perception_phrase(perception: dict[str, Any] | None) -> str:
        if not isinstance(perception, dict):
            return ""
        if not perception.get("face_detected", False):
            return "No face is visible in the camera."

        parts = ["The user is present."]
        gaze = str(perception.get("gaze_direction", "center") or "center")
        if gaze == "center":
            parts.append("Their gaze is centered on the character.")
        else:
            parts.append("Their gaze is slightly away from the character.")
        if float(perception.get("smile", 0.0) or 0.0) > 0.5:
            parts.append("They are smiling.")
        if float(perception.get("brow_raise", 0.0) or 0.0) > 0.5:
            parts.append("They look curious.")
        eye_l = float(perception.get("eye_openness_l", 1.0) or 1.0)
        eye_r = float(perception.get("eye_openness_r", 1.0) or 1.0)
        if (eye_l + eye_r) * 0.5 < 0.35:
            parts.append("They look relaxed.")
        return " ".join(parts)

    @staticmethod
    def _emotion_phrase(emotion: dict[str, float] | None) -> str:
        if not isinstance(emotion, dict) or not emotion:
            return ""
        try:
            label, value = max(emotion.items(), key=lambda item: float(item[1]))
        except (TypeError, ValueError):
            return ""
        if float(value) < 0.4:
            return ""
        return f"The character's current mood is {label}."
