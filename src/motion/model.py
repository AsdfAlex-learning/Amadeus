import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from loguru import logger

# ═══════════════════════════════════════════════════════════════
# Encoders (frozen feature extractors)
# ═══════════════════════════════════════════════════════════════


class HubertEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze
        self._loaded = False
        self.hubert = None
        self._try_load()

    def _try_load(self):
        try:
            from transformers import HubertModel

            # Try local cache first (no network)
            try:
                self.hubert = HubertModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                logger.info(f"Hubert not cached, downloading {self.model_name}...")
                self.hubert = HubertModel.from_pretrained(self.model_name)
            self._loaded = True
            if self._freeze:
                for p in self.hubert.parameters():
                    p.requires_grad = False
        except Exception as e:
            logger.warning(f"Hubert unavailable ({e}), using stub")
            self.hubert = None

    def warmup(self, device: torch.device):
        if self._loaded and self.hubert is not None:
            self.hubert = self.hubert.to(device)
            dummy = torch.randn(1, 16000, device=device)
            with torch.no_grad():
                self.forward(dummy)

    def to_half(self):
        if self._loaded and self.hubert is not None:
            self.hubert = self.hubert.half()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._loaded or self.hubert is None:
            return waveform.unsqueeze(-1).expand(-1, -1, 768).to(waveform.dtype)
        with torch.set_grad_enabled(not self._freeze):
            out = self.hubert(waveform, output_hidden_states=True)
            return out.last_hidden_state


class VisualEncoder(nn.Module):
    def __init__(
        self, model_name: str = "mobilenet_v3_small", freeze: bool = True, output_dim: int = 512
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze
        self._loaded = False
        self.output_dim = output_dim
        self.backbone = None
        self._try_load()
        self.proj = nn.Linear(576 if self._loaded else 3, output_dim)

    def _try_load(self):
        try:
            from torchvision.models import mobilenet_v3_small

            backbone = mobilenet_v3_small(weights=None)
            try:
                backbone = mobilenet_v3_small(weights="DEFAULT")
                logger.info("MobileNetV3 loaded with pretrained weights")
            except Exception:
                logger.info("MobileNetV3 using random init (no cached weights)")
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self._loaded = True
            if self._freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        except Exception as e:
            logger.warning(f"MobileNetV3 unavailable ({e}), using stub")

    def warmup(self, device: torch.device):
        if self._loaded and self.backbone is not None:
            self.backbone = self.backbone.to(device)
            dummy = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                self.forward(dummy.unsqueeze(1))

    def to_half(self):
        if self._loaded and self.backbone is not None:
            self.backbone = self.backbone.half()
            self.proj = self.proj.half()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        if self._loaded and self.backbone is not None:
            with torch.set_grad_enabled(not self._freeze):
                feats = self.backbone(frames_flat).squeeze(-1).squeeze(-1)
        else:
            feats = frames_flat.mean(dim=[2, 3]).view(B * T, -1).to(frames.dtype)
        feats = self.proj(feats)
        return feats.view(B, T, -1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2",
        freeze: bool = True,
        output_dim: int = 512,
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze
        self._loaded = False
        self.bert = None
        self.tokenizer = None
        self._try_load()
        self.bert_dim = 128 if self._loaded else 256
        self.proj = nn.Linear(self.bert_dim, output_dim)

    def _try_load(self):
        try:
            from transformers import BertModel, BertTokenizer

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self.bert = BertModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                logger.info(f"BERT not cached, downloading {self.model_name}...")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.bert = BertModel.from_pretrained(self.model_name)
            self._loaded = True
            if self._freeze:
                for p in self.bert.parameters():
                    p.requires_grad = False
        except Exception as e:
            logger.warning(f"BERT-tiny unavailable ({e}), using stub")

    def warmup(self, device: torch.device):
        if self._loaded and self.bert is not None:
            self.bert = self.bert.to(device)

    def to_half(self):
        if self._loaded and self.bert is not None:
            self.bert = self.bert.half()
            self.proj = self.proj.half()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if not self._loaded or self.bert is None or self.tokenizer is None:
            dummy = torch.zeros(len(texts), self.proj.in_features, device=device)
            return self.proj(dummy.to(dummy.dtype))
        with torch.set_grad_enabled(not self._freeze):
            tokens = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=64
            ).to(device)
            out = self.bert(**tokens).pooler_output
            return self.proj(out)


# ═══════════════════════════════════════════════════════════════
# DiT Core Components
# ═══════════════════════════════════════════════════════════════


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        ss = self.scale_shift(cond).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn_norm1 = AdaLN(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm2 = AdaLN(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = AdaLN(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def _forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cross_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (
            x
            + self.attn(
                self.attn_norm1(x, c),
                self.attn_norm1(x, c),
                self.attn_norm1(x, c),
                need_weights=False,
            )[0]
        )
        if cross_kv is not None:
            x = (
                x
                + self.cross_attn(self.attn_norm2(x, c), cross_kv, cross_kv, need_weights=False)[0]
            )
        x = x + self.ffn(self.ffn_norm(x, c))
        return x

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cross_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x, c, cross_kv, mask, use_reentrant=False)
        return self._forward(x, c, cross_kv, mask)


# ═══════════════════════════════════════════════════════════════
# Full-Duplex DiT Model
# ═══════════════════════════════════════════════════════════════


class FullDuplexDiT(nn.Module):
    def __init__(
        self,
        num_params: int = 45,
        hidden_dim: int = 320,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1280,
        dropout: float = 0.1,
        audio_encoder_name: str = "facebook/hubert-base-ls960",
        visual_encoder_name: str = "mobilenet_v3_small",
        text_encoder_name: str = "google/bert_uncased_L-2_H-128_A-2",
        freeze_encoders: bool = True,
        identity_vocab_size: int = 16,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_params = num_params
        self.hidden_dim = hidden_dim

        # ── Shared audio encoder ──
        self.audio_encoder = HubertEncoder(model_name=audio_encoder_name, freeze=freeze_encoders)

        # ── Visual encoder (user's face) ──
        self.visual_encoder = VisualEncoder(
            model_name=visual_encoder_name, freeze=freeze_encoders, output_dim=hidden_dim
        )

        # ── Text encoder (motion control prompts) ──
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name, freeze=freeze_encoders, output_dim=hidden_dim
        )

        # ── Identity embedding ──
        self.identity_embedding = nn.Embedding(identity_vocab_size, hidden_dim)

        # ── Modality projections ──
        self.audio_proj = nn.Linear(768, hidden_dim)
        self.cross_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mode_embedding = nn.Embedding(2, hidden_dim)

        # ── Timestep embedding ──
        self.time_embed = TimestepEmbedding(hidden_dim)

        # ── DiT blocks (interleaved listen/speak) ──
        self.dit_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    use_checkpoint=use_gradient_checkpointing,
                )
                for _ in range(num_layers)
            ]
        )

        # ── Output head ──
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, num_params, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if any(x in name for x in ["audio_encoder", "visual_encoder", "text_encoder"]):
                continue
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def warmup(self, device: torch.device):
        self.audio_encoder.warmup(device)
        self.visual_encoder.warmup(device)
        self.text_encoder.warmup(device)
        logger.info(f"FullDuplexDiT warmed up on {device}")

    def to_half(self):
        self.audio_encoder.to_half()
        self.visual_encoder.to_half()
        self.text_encoder.to_half()
        for module in [
            self.audio_proj,
            self.cross_proj,
            self.mode_embedding,
            self.time_embed,
            self.identity_embedding,
            self.output_head,
        ]:
            module.half()
        for block in self.dit_blocks:
            block.half()
        return self

    def forward(
        self,
        user_audio: torch.Tensor,
        tts_audio: torch.Tensor,
        visual_frames: torch.Tensor,
        text_prompts: list[str],
        identity_ids: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_params: torch.Tensor,
    ) -> torch.Tensor:
        B, T, P = noisy_params.shape
        device = noisy_params.device

        # ── Encode all modalities ──
        listen_feat = self.audio_proj(self.audio_encoder(user_audio))
        speak_feat = self.audio_proj(self.audio_encoder(tts_audio))
        visual_feat = self.visual_encoder(visual_frames)
        text_feat = self.text_encoder(text_prompts, device).unsqueeze(1).expand(-1, T, -1)
        identity_feat = self.identity_embedding(identity_ids).unsqueeze(1).expand(-1, T, -1)

        # ── Time + identity condition ──
        t_emb = self.time_embed(timesteps)
        c = t_emb + identity_feat.mean(dim=1) if identity_feat is not None else t_emb

        # ── Start from noisy params ──
        x = noisy_params

        # ── Interleaved DiT blocks ──
        listen_emb = self.mode_embedding(torch.tensor([0], device=device)).expand(B, T, -1)
        speak_emb = self.mode_embedding(torch.tensor([1], device=device)).expand(B, T, -1)

        for i, block in enumerate(self.dit_blocks):
            is_listen = i % 2 == 0
            x = x + (listen_emb if is_listen else speak_emb)
            audio_feat = listen_feat if is_listen else speak_feat
            cross_kv = torch.cat([audio_feat, visual_feat if is_listen else text_feat], dim=-1)
            cross_kv = self.cross_proj(cross_kv)
            x = block(x, c, cross_kv)

        # ── Decode to parameters ──
        x = x.transpose(1, 2)
        params = self.output_head(x)
        return params.transpose(1, 2)

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
