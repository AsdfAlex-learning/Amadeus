import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.motion.model import FullDuplexDiT
from src.runtime.concurrency import BoundedDropOldest, FrameBuffer, Mailbox


class DiffusionMotionInference:
    #: Published microphone chunks awaiting inference. Bounded and drop-oldest:
    #: a backlog of stale audio is worse than a gap, because it adds latency
    #: without improving sync.
    AUDIO_QUEUE_CAPACITY = 2
    #: Consecutive inference failures before the worker stops itself.
    MAX_CONSECUTIVE_FAILURES = 5
    #: How far ahead of the playback position Speak state may infer. One chunk of
    #: lookahead lets the first step fire before playback has advanced at all.
    SPEECH_LOOKAHEAD_SECONDS = 1.0
    #: If the cursor falls further than this behind playback, skip forward (and
    #: count it). Without a bound the lag grows relative to the audio the user
    #: actually hears, and the mouth animates audio that has already finished.
    DEFAULT_MAX_SPEECH_LAG_SECONDS = 1.5
    _WORKER_JOIN_TIMEOUT_S = 2.0
    _WORKER_POLL_S = 0.1

    def __init__(self, config: dict):
        motion_cfg = config["motion"]
        self.num_params = int(motion_cfg["output_params"])
        self.chunk_size = float(motion_cfg["chunk_size"])
        self.sample_rate = config["audio"]["sample_rate"]
        # C-5: TTS synthesises at its own rate (config tts.sample_rate, 22050 by
        # default) while this engine consumes at self.sample_rate (16000). Feeding
        # one as though it were the other compresses speech by ~1.38x, so TTS audio
        # is resampled to the pipeline rate once, at ingest.
        self._tts_sample_rate = int(config.get("tts", {}).get("sample_rate", self.sample_rate))

        # C-8: persona performance parameters were dead config. The engine
        # (src/motion/performance.py) existed and was exported, but nothing ever
        # instantiated it, so gesture_scale / expressiveness / react_speed /
        # mouth_open_max / head_motion_range / idle_energy had no effect at all.
        self._performance = self._build_performance_engine(config)

        # Stage 1.4: inference gets an owner thread so it can never run on the
        # capture drain worker, which must keep running VAD.
        self._motion_worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._motion_wake = threading.Event()
        self._audio_in_q: BoundedDropOldest[np.ndarray] = BoundedDropOldest(
            self.AUDIO_QUEUE_CAPACITY
        )
        self._consecutive_failures = 0
        self.model_path = Path(motion_cfg.get("model_path", "models/motion"))
        self.num_inference_steps = int(motion_cfg.get("num_inference_steps", 4))
        self.diffusion_beta_start = float(motion_cfg.get("diffusion_beta_start", 1e-4))
        self.diffusion_beta_end = float(motion_cfg.get("diffusion_beta_end", 0.02))
        self.identity_vocab_size = int(motion_cfg.get("identity_vocab_size", 16))

        # H2 fix: character lora directory (per-character adapter files).
        # When set, set_character_id() loads models/lora/<character_id>/lora_adapter.pt
        # after the base model is loaded. The default is "lora" under model_path.
        lora_cfg = motion_cfg.get("lora", {}) or {}
        self.lora_root = Path(
            lora_cfg.get("root", self.model_path / "lora")
        )
        self.lora_default_rank = int(lora_cfg.get("rank", 8))
        self.lora_default_alpha = float(lora_cfg.get("alpha", 16.0))
        self.lora_merge_on_load = bool(lora_cfg.get("merge_on_load", True))
        self._current_lora_id: int | None = None

        self._model: FullDuplexDiT | None = None
        self._device = torch.device("cpu")
        self._use_fp16 = bool(motion_cfg.get("use_fp16", True))
        self._loaded = False
        self._param_callbacks: list[Callable[[dict[str, float]], None]] = []
        self._audio_buffer: deque[float] = deque()
        # Speak state (S2.4). The current utterance is RETAINED and read through a
        # cursor owned by this (the motion) side; it is not drained from a queue and
        # discarded, because a position-indexed read needs the audio to still exist.
        self._speech: tuple[int, np.ndarray] | None = None
        self._speech_inferred = 0
        self._speech_skipped = 0
        self._speech_seq = 0
        # Single slot: a new utterance supersedes a *pending* one, never the one
        # already being read (dropping the active stream would jump mid-utterance).
        self._tts_pending: Mailbox[tuple[int, np.ndarray]] = Mailbox()
        # Playback clock (T7). Optional: without it the gate opens at the cursor.
        self._playback = None
        self._playback_rate = self._tts_sample_rate
        self._max_lag_s = float(
            motion_cfg.get("max_speech_lag_seconds", self.DEFAULT_MAX_SPEECH_LAG_SECONDS)
        )

        # Silence state. Without it the engine is a two-state implementation:
        # `_try_infer` requires a full chunk in some buffer, so with nothing to
        # listen to and nothing to perform there were no inference steps at all -
        # the "always-on" engine was on only while someone was talking, and idle
        # behaviour came solely from the renderer's underflow decay.
        self._silence_enabled = bool(motion_cfg.get("silence_enabled", True))
        self._silence_idle_s = float(motion_cfg.get("silence_idle_seconds", 0.5))
        self._silence_interval_s = float(motion_cfg.get("silence_interval_seconds", 1.0))
        self._silence_prompt = ""
        self._last_activity_at = time.monotonic()
        self._last_silence_at = 0.0
        # Thread-safe: written by the camera thread, read by the motion worker.
        # A plain list was shared between them, so the writer's prune raced the
        # reader's indexed snapshot. (S2.2)
        perception_cfg = config.get("perception", {}) or {}
        self._frame_fps = float(perception_cfg.get("fps", 4.0))
        self._visual_buffer = FrameBuffer(
            capacity=int(motion_cfg.get("visual_buffer_frames", 10)),
            max_age_s=max(2.5, 5.0 / max(self._frame_fps, 1e-6)),
        )
        self._overlap = 0.2
        self._prev_params: np.ndarray | None = None
        self._current_prompt = ""
        self._character_id = 0

        # Precompute diffusion schedule
        self._betas: np.ndarray | None = None
        self._alphas_cumprod: np.ndarray | None = None
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        betas = np.linspace(self.diffusion_beta_start, self.diffusion_beta_end, 1000)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self._betas = betas
        self._alphas_cumprod = alphas_cumprod

    def _detect_device(self) -> torch.device:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> bool:
        try:
            self._device = self._detect_device()
            self._model = FullDuplexDiT(
                num_params=self.num_params,
                hidden_dim=320,
                num_layers=4,
                identity_vocab_size=self.identity_vocab_size,
                use_gradient_checkpointing=False,
            )
            checkpoint_path = self.model_path / "full_duplex_dit.pt"
            if checkpoint_path.exists():
                # Support both the new full snapshot format (L4) and the legacy
                # raw state_dict. The full snapshot stores weights under "model".
                state = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
                if isinstance(state, dict) and "model" in state:
                    self._model.load_state_dict(state["model"])
                else:
                    self._model.load_state_dict(state)
                logger.info(f"Loaded Full-Duplex DiT from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint at {checkpoint_path}, using random weights")
            self._model = self._model.to(self._device)
            if self._use_fp16 and self._device.type != "cpu":
                self._model = self._model.to_half()
            self._model.eval()
            self._model.warmup(self._device)

            # H2 fix: if the current character has a LoRA adapter on disk,
            # apply it after the base model is loaded. The character_id was
            # set via set_character_id() before load_model() was called.
            if self._current_lora_id is not None:
                self._load_lora_for(self._current_lora_id)

            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_lora_for(self, character_id: int) -> None:
        """Apply the LoRA adapter for the given character onto the loaded model.

        Idempotent: if a different LoRA was previously applied, it is removed
        first so the base weights are restored before the new adapter is
        loaded. Files looked up:
            <lora_root>/<character_id>/lora_adapter.pt
            <lora_root>/<character_id>/lora_config.pt
        """
        if self._model is None:
            logger.warning("Cannot load LoRA: base model not loaded yet")
            return

        # Restore the base weights if a previous LoRA is still attached.
        if self._current_lora_id is not None and self._current_lora_id != character_id:
            try:
                from src.motion.training.lora import remove_lora
                remove_lora(self._model)
            except Exception as e:
                logger.warning(f"Could not remove previous LoRA: {e}")

        adapter_path = self.lora_root / str(character_id) / "lora_adapter.pt"
        if not adapter_path.exists():
            logger.info(
                f"No LoRA adapter at {adapter_path} — using base model for character {character_id}"
            )
            self._current_lora_id = character_id
            return

        # Load LoRA config if available; otherwise use defaults.
        from src.motion.training.lora import apply_lora, load_lora, merge_lora

        config_path = self.lora_root / str(character_id) / "lora_config.pt"
        lora_config: dict = {
            "lora_rank": self.lora_default_rank,
            "lora_alpha": self.lora_default_alpha,
        }
        if config_path.exists():
            try:
                saved_cfg = torch.load(config_path, map_location="cpu", weights_only=False)
                if isinstance(saved_cfg, dict):
                    lora_config.update(
                        {
                            k: v
                            for k, v in saved_cfg.items()
                            if k in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules")
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not read LoRA config {config_path}: {e}")

        try:
            apply_lora(self._model, lora_config)
            load_lora(self._model, adapter_path)
            if self.lora_merge_on_load:
                merge_lora(self._model)
                logger.info(
                    f"LoRA merged into base weights for character {character_id}"
                )
            self._current_lora_id = character_id
        except Exception as e:
            logger.error(f"Failed to load LoRA for character {character_id}: {e}")
            self._current_lora_id = character_id

    def on_params(self, callback: Callable[[dict[str, float]], None]):
        self._param_callbacks.append(callback)

    def set_text_prompt(self, prompt: str):
        self._current_prompt = prompt

    def set_silence_prompt(self, prompt: str):
        """Text prompt used for Silence-state motion.

        Kept separate from the dialogue prompt: idle behaviour is described by
        perception and elapsed time ("user has been quiet for 15 s"), not by what
        was last said. `docs/MODEL_TRAINING_ROADMAP.md` proposes a
        SilencePromptGenerator to populate this.
        """
        self._silence_prompt = prompt

    def set_character_id(self, char_id: int):
        """Set the active character and hot-swap LoRA adapters when possible.

        Called both before load_model() (initial selection) and at runtime
        (character swap). If the model is already loaded we attempt to swap
        the LoRA in place; otherwise the choice is remembered and applied
        when load_model() runs.
        """
        if self._character_id == char_id:
            return
        self._character_id = char_id
        if self._loaded and self._model is not None:
            self._load_lora_for(char_id)

    def process_user_audio(self, audio: np.ndarray):
        if not self._loaded or self._model is None:
            return
        self._last_activity_at = time.monotonic()
        if self._motion_worker is not None:
            # The worker owns the audio buffer and inference; this thread only
            # publishes. Runs on the capture drain worker, so it must stay fast.
            self._audio_in_q.put(np.asarray(audio, dtype=np.float32))
            self._motion_wake.set()
            return
        self._audio_buffer.extend(audio.tolist())
        self._try_infer()

    def process_tts_audio(self, audio: np.ndarray, sample_rate: int | None = None):
        """Buffer synthesized speech for the Speak state, at the pipeline rate.

        ``sample_rate`` defaults to the configured TTS rate. Callers should pass
        it explicitly when the backend's true rate differs from
        ``config.tts.sample_rate``; otherwise the duration is silently wrong.
        """
        if not self._loaded or self._model is None:
            return
        self._last_activity_at = time.monotonic()
        source_rate = int(sample_rate) if sample_rate is not None else self._tts_sample_rate
        resampled = self._resample_audio(
            np.asarray(audio, dtype=np.float32), source_rate, self.sample_rate
        )
        # Publish as a *retained* stream, already at the pipeline rate, so the
        # reader never has to mix sample rates. A single slot: a newer utterance
        # supersedes a pending one, never the one currently being read. (S2.4)
        self._speech_seq += 1
        self._tts_pending.put((self._speech_seq, resampled))
        # C-6: TTS must be able to advance the engine on its own, otherwise the
        # character stands still for the whole spoken response.
        if self._motion_worker is not None:
            self._motion_wake.set()
            return
        self._try_infer()

    # ── owner thread (ADR 0003 Stage 1.4) ────────────────────────────────────

    def start_worker(self):
        """Start the inference owner thread.

        Once running, ``process_user_audio`` and ``process_tts_audio`` only
        publish, and this thread becomes the sole mutator of the model and of
        ``_audio_buffer`` / ``_tts_buffer``.
        """
        if self._motion_worker is not None and self._motion_worker.is_alive():
            return
        self._worker_stop.clear()
        self._motion_worker = threading.Thread(
            target=self._motion_loop, name="amadeus-motion", daemon=True
        )
        self._motion_worker.start()

    def stop_worker(self):
        worker, self._motion_worker = self._motion_worker, None
        if worker is None:
            return
        self._worker_stop.set()
        self._motion_wake.set()
        worker.join(timeout=self._WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            # Escalation policy (ADR 0003): a 4-step DDIM pass can exceed the join
            # timeout on CPU, so a timeout here is expected rather than exceptional.
            # Leak the daemon instead of releasing a model it is still using.
            logger.warning(
                "Motion worker did not stop within %.0f s; leaking it (daemon)",
                self._WORKER_JOIN_TIMEOUT_S,
            )
            self._motion_worker = worker
        self._motion_wake.clear()

    def _maybe_run_silence(self) -> None:
        """Emit idle motion when nothing else is driving the engine.

        CONTEXT.md's third state had no driver at all: `_try_infer` needs a full
        chunk in some buffer, so with no speech to listen to and nothing to perform
        there were zero inference steps and idle behaviour came only from the
        renderer decaying to its neutral pose. This is what makes the engine
        actually always-on.

        Only the owner thread runs this, so it needs no locking.
        """
        if not self._silence_enabled or self._speech is not None:
            return
        chunk_samples = int(self.chunk_size * self.sample_rate)
        if len(self._audio_buffer) >= chunk_samples:
            return  # the Listen state already has work queued

        now = time.monotonic()
        if now - self._last_activity_at < self._silence_idle_s:
            return  # still within the tail of a real interaction
        if now - self._last_silence_at < self._silence_interval_s:
            return
        self._last_silence_at = now

        zeros = np.zeros(chunk_samples, dtype=np.float32)
        visual_frames = self._prepare_visual_frames(chunk_samples)
        params = self._diffusion_infer(
            zeros, zeros, visual_frames, prompt=self._silence_prompt or self._current_prompt
        )
        if params is not None:
            # "silence" mode, so PerformanceEngine uses idle_energy rather than
            # speech expressiveness.
            params = self._apply_performance(params, "silence")
            self._prev_params = params
            self._emit_params(params)

    def _motion_loop(self):
        """Sole consumer of the audio queue and sole caller of ``_try_infer``.

        Ticks on a timer rather than only on a wake, so the Silence state runs
        while nothing is published.
        """
        while not self._worker_stop.is_set():
            self._motion_wake.wait(timeout=self._WORKER_POLL_S)
            self._motion_wake.clear()
            if self._worker_stop.is_set():
                break

            # The worker owns _audio_buffer, so published audio is folded in here.
            while True:
                item = self._audio_in_q.get_nowait()
                if item is None:
                    break
                self._audio_buffer.extend(item.tolist())

            try:
                self._try_infer()
                self._maybe_run_silence()
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Motion inference failed "
                    f"({self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "Motion inference failed repeatedly; stopping the worker so the "
                        "failure is visible rather than silently disabling motion"
                    )
                    break
            else:
                self._consecutive_failures = 0

    # ── Speak-state clock (S2.4) ─────────────────────────────────────────────

    def attach_playback(self, playback, playback_rate: int | None = None) -> None:
        """Attach the playback worker (T7) so Speak state can follow the audio.

        `playback_rate` is the rate at which ``played_samples`` is counted; the
        position is converted to the pipeline rate here, so the two rates can
        never be mixed (the C-5 bug class).
        """
        self._playback = playback
        if playback_rate is not None:
            self._playback_rate = int(playback_rate)

    def _played_pipeline_samples(self) -> int:
        if self._playback is None:
            # No clock attached: behave as before, with the gate opening at the cursor.
            return self._speech_inferred
        played = int(self._playback.played_samples)
        return int(played * self.sample_rate / max(self._playback_rate, 1))

    def _speech_ready(self) -> bool:
        """Promote a pending utterance and report whether one chunk may be inferred.

        Playback *gates* the advance rather than *indexing* it: reading at the
        playback position would skip every window between this cursor and
        playback, which is the mid-utterance skip the design exists to avoid.
        """
        chunk_samples = int(self.chunk_size * self.sample_rate)

        # Degradation (S2.6): if playback has given up, its position will never
        # advance again, so a gated Speak state would stall forever and - combined
        # with the renderer holding the last frame - freeze the character on its
        # final pose. Drop the utterance so motion falls back to Listen/Silence.
        if self._playback is not None and getattr(self._playback, "failed", False):
            if self._speech is not None:
                logger.warning(
                    "Playback has failed; dropping the active utterance so motion "
                    "falls back to Listen/Silence instead of holding its last frame"
                )
                self.supersede_speech()
            return False

        if self._speech is None:
            promoted = self._tts_pending.get()
            if promoted is None:
                return False
            self._speech = promoted
            self._speech_inferred = 0
            logger.debug(f"Speak state started for stream {promoted[0]}")

        _stream_id, audio = self._speech
        total = int(audio.size)
        if self._speech_inferred >= total:
            self._speech = None
            return False

        played = self._played_pipeline_samples()
        max_lag = int(self._max_lag_s * self.sample_rate)
        if played - self._speech_inferred > max_lag:
            # Skip forward, and count it. Bounded lag relative to what the user
            # hears is the point; unbounded lag would animate finished audio.
            target = max(0, played - max_lag)
            self._speech_skipped += target - self._speech_inferred
            self._speech_inferred = target

        lookahead = int(self.SPEECH_LOOKAHEAD_SECONDS * self.sample_rate)
        gate = min(played + lookahead, total)
        return self._speech_inferred + chunk_samples <= gate

    def _pop_speech_chunk(self, chunk_samples: int) -> np.ndarray:
        _stream_id, audio = self._speech
        start = self._speech_inferred
        end = start + chunk_samples
        chunk = np.zeros(chunk_samples, dtype=np.float32)
        piece = audio[start:end]
        chunk[: piece.size] = piece
        self._speech_inferred = end
        return chunk

    @property
    def speech_active(self) -> bool:
        """True while an utterance is being performed (the model's Speak state)."""
        return self._speech is not None

    def supersede_speech(self) -> None:
        """Drop the active utterance and any pending one (interruption).

        The engine then falls back to Listen/Silence rather than continuing to
        perform audio the user has stopped hearing. The next utterance is picked
        up from the pending slot as usual.
        """
        self._speech = None
        self._speech_inferred = 0
        self._tts_pending.clear()

    @property
    def speech_skipped_samples(self) -> int:
        """Samples skipped to bound the lag behind playback."""
        return self._speech_skipped

    @staticmethod
    def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Linear resample of a mono float32 signal from ``src_rate`` to ``dst_rate``.

        Naive linear interpolation is adequate here: the result drives mouth
        motion, not playback (playback uses the original audio at its native rate).
        """
        if audio.size == 0 or src_rate == dst_rate:
            return audio
        duration = audio.size / float(src_rate)
        dst_count = max(1, int(round(duration * dst_rate)))
        src_positions = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        dst_positions = np.linspace(0.0, duration, num=dst_count, endpoint=False)
        return np.interp(dst_positions, src_positions, audio).astype(np.float32)

    def process_visual_frame(self, frame: np.ndarray):
        """Publish a camera frame. Called from the camera thread."""
        self._visual_buffer.put(frame)

    def _try_infer(self):
        """Run one inference step per available second of audio, from either input.

        Speak state must advance on TTS audio alone. If a step were only produced
        when a full *microphone* chunk existed, the character would freeze the
        moment the user stopped talking - the engine is specified as always-on.
        So a step runs when either input has a chunk, and the absent side is fed
        zeros, which is precisely the model's Speak (user_audio=0) and Listen
        (tts_audio=0) conditioning. (C-6)
        """
        chunk_samples = int(self.chunk_size * self.sample_rate)
        while True:
            speaking = self._speech_ready()
            has_user = len(self._audio_buffer) >= chunk_samples
            if not speaking and not has_user:
                break

            visual_frames = self._prepare_visual_frames(chunk_samples)

            if speaking:
                # Q2: during an active utterance the microphone is zeroed. That IS
                # the model's Speak conditioning (user_audio = 0, tts_audio = speech);
                # feeding both non-zero was a fourth state present in no spec.
                user_chunk = np.zeros(chunk_samples, dtype=np.float32)
                tts_chunk = self._pop_speech_chunk(chunk_samples)
            else:
                user_chunk = np.array(
                    [self._audio_buffer.popleft() for _ in range(chunk_samples)],
                    dtype=np.float32,
                )
                tts_chunk = np.zeros(chunk_samples, dtype=np.float32)

                if self._overlap > 0:
                    keep_n = int(self._overlap * self.sample_rate)
                    keep = [
                        self._audio_buffer.popleft()
                        for _ in range(min(keep_n, len(self._audio_buffer)))
                    ]
                    self._audio_buffer.extendleft(reversed(keep))

            params = self._diffusion_infer(user_chunk, tts_chunk, visual_frames)
            if params is not None:
                # C-8: per-persona multipliers are applied between the model and
                # the renderer. Speak when speaking, otherwise Listen.
                params = self._apply_performance(
                    params, "speak" if speaking else "listen"
                )
                self._prev_params = params
                self._emit_params(params)

    def _prepare_visual_frames(self, audio_samples: int) -> np.ndarray:
        """Return camera frames as ``(T, C, H, W)`` float32, using the **newest** ones.

        The model's ``VisualEncoder`` unpacks ``(B, T, C, H, W)`` and training
        feeds ``(B, 5, 3, 224, 224)``, but ``CameraPerception`` delivers frames as
        HWC ``(224, 224, 3)``. This is the single boundary where that conversion
        belongs, so the empty and populated paths agree on layout. (C-4)

        Taking the newest 5 matters: the previous implementation sampled across
        the whole buffer with ``range(0, num, step)``, biasing the model toward
        frames up to ~2 s old while pairing them with audio from *now*. (S2.2)
        """
        frames = self._visual_buffer.snapshot(5)
        if not frames:
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
        while len(frames) < 5:
            frames.append(frames[-1])  # repeat the newest until the window is full

        # (T, H, W, C) as supplied by CameraPerception -> (T, C, H, W)
        frames_hwc = np.stack(frames)
        return np.ascontiguousarray(frames_hwc.transpose(0, 3, 1, 2))

    def _diffusion_infer(
        self,
        user_audio: np.ndarray,
        tts_audio: np.ndarray,
        visual_frames: np.ndarray,
        prompt: str | None = None,
    ) -> np.ndarray | None:
        if self._model is None:
            return None
        if visual_frames.ndim < 4:
            if visual_frames.ndim == 3:
                visual_frames = np.stack([visual_frames] * 5)
            else:
                visual_frames = np.zeros((5, 3, 224, 224), dtype=np.float32)

        with torch.no_grad():
            user_wav = torch.from_numpy(user_audio).unsqueeze(0).to(self._device)
            tts_wav = torch.from_numpy(tts_audio).unsqueeze(0).to(self._device)
            vis = torch.from_numpy(visual_frames).unsqueeze(0).float().to(self._device)
            id_tensor = torch.tensor([self._character_id], device=self._device)
            active_prompt = prompt if prompt is not None else self._current_prompt
            prompts = [active_prompt] if active_prompt else [""]

            B = 1
            # Derive T from the audio encoder output length instead of hardcoding
            # (M3 fix). Keep the encoding and hand it to the forward pass: it needs
            # the same features, so recomputing them would be a full extra Hubert
            # pass per DDIM step.
            listen_feat = self._model.audio_proj(self._model.audio_encoder(user_wav))
            T = listen_feat.shape[1]
            x = torch.randn(B, T, self.num_params, device=self._device)

            # X-prediction DDIM sampling.
            # Model directly predicts x_0 ∈ [0, 1]. Step formula:
            #   pred_eps = (x_t - sqrt(ᾱ_t) * pred_x0) / sqrt(1 - ᾱ_t)
            #   x_{t-1}  = sqrt(ᾱ_{t-1}) * pred_x0
            #             + sqrt(1 - ᾱ_{t-1} - σ²) * pred_eps
            #             + σ * noise   (η=0 → deterministic)
            timesteps = torch.linspace(
                999, 0, self.num_inference_steps + 1, device=self._device
            ).long()
            for i in range(self.num_inference_steps):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                t_tensor = t.unsqueeze(0).expand(B)
                alpha_t = torch.tensor(self._alphas_cumprod[t], device=self._device)
                pred_x0 = self._model(
                    user_wav, tts_wav, vis, prompts, id_tensor, t_tensor, x,
                    listen_feat=listen_feat,
                )
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                pred_eps = (x - sqrt_alpha_t * pred_x0) / sqrt_one_minus_alpha_t

                alpha_prev = (
                    torch.tensor(self._alphas_cumprod[t_next], device=self._device)
                    if t_next > 0
                    else torch.tensor(1.0, device=self._device)
                )
                # η=0 → deterministic DDIM (no stochastic noise injection)
                sigma_t = torch.tensor(0.0, device=self._device)
                dir_xt = torch.sqrt((1.0 - alpha_prev - sigma_t ** 2).clamp(min=0.0)) * pred_eps
                noise = torch.randn_like(x) if t_next > 0 else torch.zeros_like(x)
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise

            return x[0].cpu().numpy()

    def _emit_params(self, params_seq: np.ndarray):
        num_frames = params_seq.shape[0]
        param_names = [f"param_{i}" for i in range(self.num_params)]
        for frame_idx in range(num_frames):
            frame_params = params_seq[frame_idx]
            param_dict = {name: float(frame_params[i]) for i, name in enumerate(param_names)}
            for cb in self._param_callbacks:
                try:
                    cb(param_dict)
                except Exception as e:
                    logger.error(f"Param callback error: {e}")

    @staticmethod
    def _build_performance_engine(config: dict):
        """Build the persona post-processor from ``config.performance``. (C-8)"""
        from src.motion.performance import PerformanceConfig, PerformanceEngine

        perf_cfg = config.get("performance", {}) or {}
        mapping_path = (
            (config.get("preprocess", {}) or {}).get("arkit_mapping", {}).get("config_path")
        )
        # The config path is project-relative; resolve it so the category mapping
        # (mouth / eye / head indices) is used rather than the heuristic fallback.
        if mapping_path and not Path(mapping_path).is_absolute():
            candidate = Path(__file__).resolve().parents[2] / mapping_path
            if candidate.exists():
                mapping_path = candidate
        return PerformanceEngine(
            PerformanceConfig.from_dict(perf_cfg), mapping_path=mapping_path
        )

    def _apply_performance(self, params: np.ndarray, mode: str) -> np.ndarray:
        """Apply per-persona multipliers to the raw model output. (C-8)"""
        if self._performance is None:
            return params
        try:
            return self._performance.apply(params, mode=mode)
        except Exception as e:
            # Post-processing must never take down motion; fall back to raw params.
            logger.warning(f"Performance post-processing failed ({e}); using raw params")
            return params

    def reset(self):
        self._audio_buffer.clear()
        self._speech = None
        self._speech_inferred = 0
        self._tts_pending.clear()
        self._visual_buffer.clear()
        self._prev_params = None
        if self._performance is not None:
            self._performance.reset()

    def cleanup(self):
        self.stop_worker()
        self._model = None
        self._loaded = False
        self._param_callbacks.clear()
        self._audio_buffer.clear()
        self._speech = None
        self._speech_inferred = 0
        self._tts_pending.clear()
        self._visual_buffer.clear()
        if self._performance is not None:
            self._performance.reset()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
