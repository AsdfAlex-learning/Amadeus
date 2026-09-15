import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.motion.model import FullDuplexDiT
from src.runtime.concurrency import BoundedDropOldest


class DiffusionMotionInference:
    #: Published microphone chunks awaiting inference. Bounded and drop-oldest:
    #: a backlog of stale audio is worse than a gap, because it adds latency
    #: without improving sync.
    AUDIO_QUEUE_CAPACITY = 2
    #: Consecutive inference failures before the worker stops itself.
    MAX_CONSECUTIVE_FAILURES = 5
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
        self._tts_buffer: deque[float] = deque()
        self._visual_buffer: list[np.ndarray] = []
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
        source_rate = int(sample_rate) if sample_rate is not None else self._tts_sample_rate
        resampled = self._resample_audio(
            np.asarray(audio, dtype=np.float32), source_rate, self.sample_rate
        )
        self._tts_buffer.extend(resampled.tolist())
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

    def _motion_loop(self):
        """Sole consumer of the audio queue and sole caller of ``_try_infer``."""
        while not self._worker_stop.is_set():
            if not self._motion_wake.wait(timeout=self._WORKER_POLL_S):
                continue
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
        self._visual_buffer.append(frame)
        if len(self._visual_buffer) > 100:
            self._visual_buffer.pop(0)

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
            has_user = len(self._audio_buffer) >= chunk_samples
            has_tts = len(self._tts_buffer) >= chunk_samples
            if not has_user and not has_tts:
                break

            user_chunk = np.zeros(chunk_samples, dtype=np.float32)
            if has_user:
                user_chunk = np.array(
                    [self._audio_buffer.popleft() for _ in range(chunk_samples)],
                    dtype=np.float32,
                )

            tts_len = min(len(self._tts_buffer), chunk_samples)
            tts_chunk = np.zeros(chunk_samples, dtype=np.float32)
            if tts_len > 0:
                tts_chunk[:tts_len] = np.array(
                    [self._tts_buffer.popleft() for _ in range(tts_len)], dtype=np.float32
                )

            visual_frames = self._prepare_visual_frames(chunk_samples)

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
                # the renderer. Speak when TTS is present, otherwise Listen.
                params = self._apply_performance(params, "speak" if has_tts else "listen")
                self._prev_params = params
                self._emit_params(params)

    def _prepare_visual_frames(self, audio_samples: int) -> np.ndarray:
        """Return camera frames as ``(T, C, H, W)`` float32.

        The model's ``VisualEncoder`` unpacks ``(B, T, C, H, W)`` and training
        feeds ``(B, 5, 3, 224, 224)``, but ``CameraPerception`` delivers frames as
        HWC ``(224, 224, 3)``. This is the single boundary where that conversion
        belongs, so the empty and populated paths agree on layout. (C-4)
        """
        num_visual = min(len(self._visual_buffer), 5)
        if num_visual == 0:
            return np.zeros((1, 3, 224, 224), dtype=np.float32)

        if num_visual < 5:
            indices = [int(i * num_visual / 5) for i in range(5)]
        else:
            step = max(1, num_visual // 5)
            indices = list(range(0, num_visual, step))[:5]

        # (T, H, W, C) as supplied by CameraPerception -> (T, C, H, W)
        frames_hwc = np.stack([self._visual_buffer[i] for i in indices])
        return np.ascontiguousarray(frames_hwc.transpose(0, 3, 1, 2))

    def _diffusion_infer(
        self, user_audio: np.ndarray, tts_audio: np.ndarray, visual_frames: np.ndarray
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
            prompts = [self._current_prompt] if self._current_prompt else [""]

            B = 1
            # Derive T from the audio encoder output length instead of hardcoding
            # (M3 fix). This keeps inference aligned with variable-length audio.
            with torch.no_grad():
                audio_feat = self._model.audio_encoder(user_wav)
            T = audio_feat.shape[1]
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
                pred_x0 = self._model(user_wav, tts_wav, vis, prompts, id_tensor, t_tensor, x)
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
        self._tts_buffer.clear()
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
        self._tts_buffer.clear()
        self._visual_buffer.clear()
        if self._performance is not None:
            self._performance.reset()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
