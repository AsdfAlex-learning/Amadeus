from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.motion.model import FullDuplexDiT


class DiffusionMotionInference:
    def __init__(self, config: dict):
        motion_cfg = config["motion"]
        self.num_params = int(motion_cfg["output_params"])
        self.chunk_size = float(motion_cfg["chunk_size"])
        self.sample_rate = config["audio"]["sample_rate"]
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
        self._audio_buffer.extend(audio.tolist())
        self._try_infer()

    def process_tts_audio(self, audio: np.ndarray):
        if not self._loaded or self._model is None:
            return
        self._tts_buffer.extend(audio.tolist())

    def process_visual_frame(self, frame: np.ndarray):
        self._visual_buffer.append(frame)
        if len(self._visual_buffer) > 100:
            self._visual_buffer.pop(0)

    def _try_infer(self):
        chunk_samples = int(self.chunk_size * self.sample_rate)
        while len(self._audio_buffer) >= chunk_samples:
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
                self._prev_params = params
                self._emit_params(params)

    def _prepare_visual_frames(self, audio_samples: int) -> np.ndarray:
        num_visual = min(len(self._visual_buffer), 5)
        if num_visual == 0:
            return np.zeros((1, 3, 224, 224), dtype=np.float32)
        if num_visual < 5:
            frames = []
            for i in range(5):
                idx = int(i * num_visual / 5)
                frames.append(self._visual_buffer[idx])
            return np.stack(frames)
        step = max(1, num_visual // 5)
        return np.stack([self._visual_buffer[i] for i in range(0, num_visual, step)][:5])

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

    def reset(self):
        self._audio_buffer.clear()
        self._tts_buffer.clear()
        self._visual_buffer.clear()
        self._prev_params = None

    def cleanup(self):
        self._model = None
        self._loaded = False
        self._param_callbacks.clear()
        self._audio_buffer.clear()
        self._tts_buffer.clear()
        self._visual_buffer.clear()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
