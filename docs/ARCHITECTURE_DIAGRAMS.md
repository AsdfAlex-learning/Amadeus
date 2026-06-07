# Amadeus 算法架构图 (Mermaid)

> 生成日期: 2026-06-07  
> 基于代码审查，反映当前实现状态（含已知问题）

---

## 1. 系统数据流总览

```mermaid
flowchart TB
    subgraph Input["输入"]
        MIC["🎤 麦克风<br/>16kHz mono"]
        CAM["📷 摄像头<br/>OpenCV + MediaPipe"]
        LLM["💬 对话模型<br/>Qwen2.5-3B"]
    end

    subgraph AudioPipeline["音频管线"]
        MIC --> VAD["VAD<br/>(能量检测)"]
        VAD -->|"语音段"| ASR["Whisper.cpp ASR"]
        ASR --> TEXT["ASR 文本"]
    end

    subgraph Dialogue["对话子系统"]
        TEXT --> CTX["ConversationContext<br/>(滑动窗口)"]
        LLM -->|"系统提示"| CTX
        CTX -->|"生成响应"| TTS["TTS Engine<br/>(CosyVoice2/ChatTTS/...)"]
        TTS --> TTS_AUDIO["TTS 音频流"]
    end

    subgraph Perception["感知子系统"]
        CAM --> FACE["MediaPipe FaceMesh<br/>FaceLandmarker"]
        FACE -->|"gaze, expression<br/>face_detected"| PERCEPT["CameraPerception<br/>结果"]
    end

    subgraph MotionPipeline["运动管线 ⚠️"]
        MIC -->|"原始音频"| USR_AUDIO["user_audio<br/>(16kHz)"]
        TTS_AUDIO --> TTS_AUDIO_BUF["tts_audio<br/>(16kHz)"]
        PERCEPT -->|"text_prompt"| TEXT_PROMPT["情绪描述文本"]

        USR_AUDIO --> DIT["FullDuplexDiT<br/>Mini-LPM"]
        TTS_AUDIO_BUF --> DIT
        CAM_FRAMES["visual_frames<br/>(5×224×224)"] --> DIT
        TEXT_PROMPT --> DIT
        CHAR_ID["identity_id"] --> DIT

        DIT -->|"50 frames × 45 params"| PERF["PerformanceEngine<br/>(Persona 后处理)"]
        PERF -->|"调整后参数"| RENDER["Live2D Renderer<br/>(live2d-py + PySide6)"]
    end

    style DIT fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style PERF fill:#51cf66,stroke:#2b8a3e,color:#fff
```

## 2. FullDuplexDiT 模型架构

```mermaid
flowchart TB
    subgraph Inputs["5-Stream 多模态输入"]
        UA["user_audio<br/>(B, 16000)"]
        TA["tts_audio<br/>(B, 16000)"]
        VF["visual_frames<br/>(B, 5, 3, 224, 224)"]
        TP["text_prompts<br/>(list of str)"]
        ID["identity_ids<br/>(B,)"]
    end

    subgraph Encoders["冻结编码器 (94M+2.5M+4.4M ≈ 101M)"]
        HE["HubertEncoder<br/>facebook/hubert-base-ls960<br/>(768-dim @50Hz)<br/>⚠️ 共享于 Listen/Speak"]
        VE["VisualEncoder<br/>MobileNetV3-Small + proj<br/>(512→320)"]
        TE["TextEncoder<br/>BERT-tiny + proj<br/>(128→320)"]
    end

    UA --> HE
    TA --> HE
    VF --> VE
    TP --> TE

    subgraph Projections["投影层"]
        AP["audio_proj<br/>Linear(768→320)"]
        CP["cross_proj<br/>Linear(640→320)"]
        ME["mode_embedding<br/>Embedding(2→320)"]
    end

    HE --> AP
    AP --> LISTEN_FEAT["listen_feat<br/>(B, T, 320)"]
    HE --> AP --> SPEAK_FEAT["speak_feat<br/>(B, T, 320)"]

    subgraph Conditioning["条件信号"]
        TE --> TEXT_FEAT["text_feat<br/>(B, T, 320)"]
        ID --> ID_EMB["identity_feat<br/>(B, T, 320)"]
        TS["timesteps<br/>(B,)"] --> T_EMB["TimestepEmbedding<br/>(320)"]
    end

    T_EMB --> C["c = t_emb + identity_feat"]
    C --> BLOCKS

    subgraph DiTCore["DiT 交错块 (×4)"]
        direction TB
        BLOCKS["DiT Block 集合"]

        subgraph B0["Block 0 → Listen"]
            direction LR
            LE0["+ listen_emb"] --> SA0["Self-Attn<br/>(8 heads)"]
            SA0 --> CA0["Cross-Attn<br/>key=cat[listen_feat, visual_feat]<br/>proj→320"]
            CA0 --> FFN0["FFN<br/>(320→1280→320)"]
        end

        subgraph B1["Block 1 → Speak"]
            direction LR
            SE1["+ speak_emb"] --> SA1["Self-Attn"]
            SA1 --> CA1["Cross-Attn<br/>key=cat[speak_feat, text_feat]<br/>proj→320"]
            CA1 --> FFN1["FFN"]
        end

        subgraph B2["Block 2 → Listen"]
            SA2["Self-Attn + Cross-Attn<br/>同 Block 0"]
        end

        subgraph B3["Block 3 → Speak"]
            SA3["Self-Attn + Cross-Attn<br/>同 Block 1"]
        end
    end

    LISTEN_FEAT -->|"listen 模式"| CA0
    SPEAK_FEAT -->|"speak 模式"| CA1

    subgraph VisualCross["视觉/文本 Cross-Attn KV"]
        VF_CROSS["visual_feat<br/>(B, T, 320)"]
        TP_CROSS["text_feat<br/>(B, T, 320)"]
    end

    VE --> VF_CROSS
    TE --> TP_CROSS

    VF_CROSS -->|"listen"| CA0
    TP_CROSS -->|"speak"| CA1

    subgraph OutputHead["输出头 ⚠️ CRITICAL BUG"]
        direction TB
        TRANS["Transpose<br/>(B, T, 320) → (B, 320, T)"]
        C1["Conv1d(320, 320, k=5)<br/>+ GELU + Dropout"]
        C2["Conv1d(320, 160, k=5)<br/>+ GELU + Dropout"]
        C3["Conv1d(160, 45, k=5)<br/>+ Sigmoid ⚠️"]
        OUT["output<br/>(B, T, 45) ∈ [0, 1]"]
    end

    BLOCKS --> TRANS --> C1 --> C2 --> C3 --> OUT

    NOISE["noisy_params<br/>(B, T, 45)"] --> X["x = noisy_params"] --> BLOCKS

    style C3 fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style HE fill:#ffd43b,stroke:#e67700,color:#000
    style C2 fill:#d0bfff,stroke:#7950f2,color:#000
```

> ⚠️ **Sigmoid + ε-prediction**: 输出 Sigmoid 范围 [0,1]，但训练目标是标准正态噪声 N(0,1)，数学上不兼容。应切换到 x-prediction 或移除 Sigmoid。

## 3. DiT Block 内部结构

```mermaid
flowchart LR
    X_IN["x"] --> ADA1["AdaLN<br/>(t_emb + identity)"]
    ADA1 --> Q1["Q"]
    ADA1 --> K1["K"]
    ADA1 --> V1["V"]
    Q1 --> SA["Multi-Head<br/>Self-Attention<br/>(8 heads)"]
    K1 --> SA
    V1 --> SA
    SA --> RES1["+ x"]
    RES1 --> ADA2["AdaLN"]
    ADA2 --> Q2["Q"]
    K2["cross_kv"] --> CA["Multi-Head<br/>Cross-Attention<br/>(8 heads)"]
    Q2 --> CA
    CA --> RES2["+ x"]
    RES2 --> ADA3["AdaLN"]
    ADA3 --> FFN_IN["Linear(320→1280)"]
    FFN_IN --> GELU["GELU"]
    GELU --> DROP1["Dropout(0.1)"]
    DROP1 --> FFN_OUT["Linear(1280→320)"]
    FFN_OUT --> DROP2["Dropout(0.1)"]
    DROP2 --> RES3["+ x"]
    RES3 --> X_OUT["x"]
```

> AdaLN 条件信号: `c = TimestepEmbedding(t) + IdentityEmbedding(id).mean(dim=1)`

## 4. 训练管线流程

```mermaid
flowchart TB
    subgraph Preprocess["预处理管线"]
        direction TB
        VIDEO["视频文件<br/>.mp4/.mkv/..."] --> VR["VideoReader<br/>(ffmpeg + cv2 fallback)"]
        VR -->|"提取帧"| FL["FaceLandmarkerExtractor<br/>(MediaPipe Tasks API)"]
        VR -->|"提取音频"| AUDIO_WAV["音频 WAV<br/>(16kHz mono)"]
        FL -->|"52 ARKit blendshapes<br/>+ head_angles"| MAP["ARKitToLive2DMapper<br/>(YAML 配置)"]
        MAP -->|"45 Live2D params"| NPZ[".npz 文件<br/>live2d_params, blendshapes,<br/>head_angles, bad_frames, fps, identity_id"]
        AUDIO_WAV --> NPZ
    end

    subgraph Dataset["数据集 (MotionDataset)"]
        direction TB
        NPZ_DATA[".npz 文件目录"] --> SCAN["_scan_npz()<br/>查找 npz + 对应 wav"]
        SCAN --> LOAD["_get_npz_item()"]
        LOAD --> CHUNk["随机块选择<br/>chunk_duration=1.0s"]
        CHUNk --> PAD["⚠️ 25fps→50fps<br/>零填充 (50% 零!)"]
        PAD --> BATCH["{'user_audio': (16000,),<br/>'tts_audio': zeros,<br/>'visual_frames': zeros ⚠️<br/>'text_prompt': '',<br/>'identity_id': 0,<br/>'motion': (50, 45)}"]
    end

    subgraph TrainingLoop["训练循环 (train.py)"]
        direction TB
        BATCH --> DIFF["DDPM 前向过程<br/>t ~ Uniform(0, 999)<br/>noise ~ N(0,1)<br/>noisy = √ᾱ_t · x_0 + √(1-ᾱ_t) · noise"]
        DIFF --> PRED["model(audio, tts, visual, prompt, id, t, noisy)"]
        PRED --> LOSS["⚠️ loss = MSELoss(pred, noise)<br/>但 pred ∈ [0,1] (Sigmoid)<br/>noise ∈ ℝ (标准正态)<br/>→ 不兼容!"]
        LOSS -->|"backward"| GRAD["梯度累积 × grad_accum_steps"]
        GRAD --> OPT["AdamW<br/>⚠️ 缺 weight_decay"]
        OPT -->|"每 epoch"| SCHED["CosineAnnealingLR<br/>⚠️ 缺 warmup"]
        SCHED -->|"每 10 epoch"| CKPT["保存检查点<br/>⚠️ 不含 optimizer 状态"]
        SCHED --> VAL["验证集评估<br/>(10% random_split)"]
    end

    subgraph LoRA["LoRA 微调 (--use_lora)"]
        direction TB
        BASE_CKPT["基础模型检查点"] --> LORA_APPLY["apply_lora(model, config)<br/>冻结所有参数<br/>替换 Linear/Conv1d → LoRALinear/LoRAConv1d"]
        LORA_APPLY --> LORA_TRAIN["仅 LoRA A/B 矩阵可训练<br/>rank=8, alpha=16<br/>≈500K 参数/角色"]
        LORA_TRAIN --> LORA_SAVE["save_lora() → lora_adapter.pt<br/>仅保存 A/B 矩阵 (~MB)"]
    end

    style PAD fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style LOSS fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style OPT fill:#ffd43b,stroke:#e67700,color:#000
```

## 5. 推理管线流程

```mermaid
flowchart TB
    subgraph Runtime["运行时数据流"]
        direction TB
        MIC_IN["麦克风输入<br/>(16kHz)"] --> BUF["_audio_buffer<br/>(deque)"]
        BUF -->|"chunk_size 采样"| CHUNK["音频块<br/>(16000 samples)"]
        
        TTS_IN["TTS 音频流"] --> TTS_BUF["_tts_buffer<br/>(deque)"]
        
        CAM_IN["摄像头帧"] --> VIS_BUF["_visual_buffer<br/>(list, max 100)"]
    end

    subgraph Inference["DiffusionMotionInference 推理"]
        direction TB
        CHUNK --> DDIFF["4-step DDIM 去噪"]
        TTS_BUF --> DDIFF
        VIS_BUF --> PREP_VIS["_prepare_visual_frames<br/>(5帧采样)"]
        PREP_VIS --> DDIFF

        subgraph DDIM["⚠️ DDIM 步骤 (非标准)"]
            direction TB
            INIT["params ~ N(0,1)<br/>T=50 硬编码 ⚠️"]
            STEP1["t=999→749<br/>pred = model(..., t, params)<br/>params = (params - β/√(1-ᾱ) · pred) / √(1-β) + σ·noise"]
            STEP2["t=749→499<br/>同上"]
            STEP3["t=499→249<br/>同上"]
            STEP4["t=249→0<br/>同上，无 noise"]
            INIT --> STEP1 --> STEP2 --> STEP3 --> STEP4
        end

        DDIFF -->|"50 frames × 45 params"| PERF["PerformanceEngine<br/>(Persona 后处理)"]
        
        subgraph PerfPost["PerformanceEngine 后处理"]
            direction TB
            GS["gesture_scale<br/>整体运动幅度 ×(val-0.5)+0.5"]
            EXPR["expressiveness<br/>面部参数 ×(val-0.5)+0.5"]
            MOMAX["mouth_open_max<br/>嘴部参数 clip(0, val)"]
            HEAD["head_motion_range<br/>头部参数 ×val"]
            SPEED["react_speed<br/>EMA 时序平滑"]
            IDLE["idle_energy<br/>静默模式面部变化幅度"]
            GS --> RESULT["调整后参数 (T, 45)"]
            EXPR --> RESULT
            MOMAX --> RESULT
            HEAD --> RESULT
            SPEED --> RESULT
            IDLE --> RESULT
        end

        RESULT --> CALLBACKS["_param_callbacks<br/>→ Live2DWidget.push_params()"]
    end

    style INIT fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style STEP1 fill:#ffd43b,stroke:#e67700,color:#000
    style STEP2 fill:#ffd43b,stroke:#e67700,color:#000
    style STEP3 fill:#ffd43b,stroke:#e67700,color:#000
    style STEP4 fill:#ffd43b,stroke:#e67700,color:#000
```

## 6. LoRA 微调架构

```mermaid
flowchart TB
    subgraph BaseModel["基础模型 (冻结)"]
        direction TB
        HUB["HubertEncoder<br/>(94M, 冻结)"]
        VIS["VisualEncoder<br/>(2.5M, 冻结)"]
        TXT["TextEncoder<br/>(4.4M, 冻结)"]
        DIT_BASE["DiT Blocks<br/>(~15M)"]
        OUT_BASE["Output Head<br/>(~6M)"]
        PROJ["Projections<br/>(~0.3M)"]
    end

    subgraph LoRALayers["LoRA 适配器 (可训练)"]
        direction TB
        LA1["LoRALinear<br/>attn_norm1.scale_shift<br/>A:(8,320) B:(320,8)"]
        LA2["LoRALinear<br/>attn.in_proj_weight 区域<br/>A:(8,320) B:(320,8)"]
        LA3["LoRALinear<br/>ffn layers<br/>A:(8,d_in) B:(d_out,8)"]
        LC1["LoRAConv1d<br/>output_head Conv1d layers<br/>A:(8, fan_in) B:(d_out,8)"]
    end

    BASE_WEIGHT["原始权重 W<br/>(冻结, requires_grad=False)"] --> MATMUL_BASE["y_base = W @ x"]
    LORA_A["LoRA A 矩阵<br/>(Kaiming 初始化)"] --> LORA_MATMUL["lora_out = (x @ A^T) @ B^T"]
    LORA_MATMUL --> SCALE["× (alpha / rank) = 16/8 = 2.0"]
    SCALE --> ADD["y = y_base + lora_out"]
    ADD --> OUTPUT["输出"]

    MATMUL_BASE --> ADD

    subgraph Lifecycle["LoRA 生命周期"]
        direction LR
        APPLY["apply_lora()<br/>冻结基础模型<br/>替换模块"] --> TRAIN["LoRA 训练<br/>仅 A/B 可训练"]
        TRAIN --> SAVE["save_lora()<br/>仅保存 A/B (~MB)"]
        SAVE --> LOAD["load_lora()<br/>加载到新模型"]
        LOAD --> MERGE["merge_lora()<br/>W += α·B@A<br/>部署加速"]
        APPLY --> REMOVE["remove_lora()<br/>恢复原始模块<br/>丢弃 LoRA"]
    end

    style LoRALayers fill:#d0bfff,stroke:#7950f2,color:#000
    style BASE_WEIGHT fill:#dee2e6,stroke:#868e96,color:#000
```

## 7. 预处理管线数据流

```mermaid
flowchart LR
    subgraph Input["输入"]
        VIDEO["视频文件<br/>.mp4/.mkv/.avi/.mov"]
    end

    VIDEO --> VR["VideoReader"]

    subgraph VideoReaderInternals["VideoReader 内部"]
        direction TB
        VR_A["extract_audio()<br/>ffmpeg → 16kHz WAV<br/>fallback: soundfile"]
        VR_F["iter_frames()<br/>ffmpeg pipe → BGR numpy<br/>fallback: cv2.VideoCapture"]
        VR_META["get_metadata()<br/>ffprobe → fps/duration/codec<br/>fallback: cv2"]
    end

    VR --> VR_A
    VR --> VR_F

    VR_A --> AUDIO_OUT["_audio.wav<br/>(16kHz mono f32)"]
    VR_F --> FRAMES_OUT["BGR 帧流<br/>(H, W, 3) uint8"]

    subgraph FaceProcessing["FaceLandmarkerExtractor"]
        direction TB
        FL_INIT["_init_landmarker()<br/>Tasks API → legacy FaceMesh → zeros fallback"]
        FL_MODEL["模型下载<br/>face_landmarker_v2_with_blendshapes.task<br/>(自动从 GCS 下载)"]
        FL_PROC["process_frame(frame)<br/>→ 52 blendshapes<br/>→ head_pitch/yaw/roll<br/>→ face_detected"]
        FL_LEGACY["_landmarks_to_blendshapes()<br/>478点 → 52 blendshapes 近似<br/>_estimate_head_pose()<br/>landmark 几何 → 角度"]
    end

    FRAMES_OUT --> FL_PROC
    FL_INIT --> FL_PROC
    FL_MODEL --> FL_INIT

    FL_PROC --> BS["blendshapes (T, 52) float32"]
    FL_PROC --> HA["head_angles (T, 3) float32"]
    FL_PROC --> BAD["bad_frames list[int]"]

    subgraph Mapping["ARKitToLive2DMapper"]
        direction TB
        YAML["default.yaml<br/>45 参数映射规则"]
        MAP_LOGIC["map() 逻辑<br/>head_angle: angle × scale + bias<br/>blendshape: Σ(weight × bs) + bias<br/>constant: 固定值"]
    end

    BS --> MAP_LOGIC
    HA --> MAP_LOGIC
    YAML --> MAP_LOGIC

    MAP_LOGIC --> L2D["live2d_params (T, 45) float32"]

    subgraph PostProcess["后处理"]
        INTERP["_interpolate_bad_frames()<br/>坏帧 ← 邻居均值<br/>⚠️ O(n) list 查找"]
        CLIP["np.clip(result, 0, 1)"]
    end

    L2D --> INTERP --> CLIP

    CLIP --> NPZ_OUT[".npz 输出<br/>──────────────<br/>live2d_params (T, 45)<br/>blendshapes (T, 52)<br/>head_angles (T, 3)<br/>bad_frames (N,)<br/>fps, duration_sec<br/>identity_id, source_video"]

    AUDIO_OUT --> AUDIO_META["_audio.wav<br/>伴随 NPZ"]

    NPZ_OUT --> DS["MotionDataset<br/>⚠️ fps 不匹配: 25→50"]
    AUDIO_META --> DS

    style DS fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

## 8. 性能参数后处理详解

```mermaid
flowchart TB
    INPUT["原始参数<br/>(T, 45) ∈ [0, 1]"]

    INPUT --> GS["gesture_scale<br/>new = 0.5 + (val - 0.5) × scale<br/>整体运动幅度缩放"]
    
    GS --> MODE_CHECK{"模式?"}

    MODE_CHECK -->|speak/listen| EXPR["expressiveness<br/>face_params = 0.5 + (val - 0.5) × expr<br/>面部表情夸张度"]
    MODE_CHECK -->|silence| IDLE_ENERGY["idle_energy<br/>face_params = mean + (val - mean) × energy<br/>静默模式能量水平"]

    EXPR --> MOUTH["mouth_open_max<br/>clip(mouth_params, 0, max)<br/>嘴部最大开合度"]
    IDLE_ENERGY --> MOUTH

    MOUTH --> HEAD_M["head_motion_range<br/>head_params × range<br/>头部/身体运动范围"]
    
    HEAD_M --> REACT["react_speed<br/>EMA 时序平滑<br/>val[t] = val[t-1] + α × (val[t] - val[t-1])<br/>α ∈ [0, 1], 默认 0.5"]

    REACT --> FINAL["最终参数<br/>(T, 45) ∈ [0, 1]"]

    subgraph ParamGroups["参数分组映射"]
        MOUTH_IDS["嘴部参数<br/>indices: 0-5<br/>mouthOpenY, mouthForm, ..."]
        EYE_IDS["眼部参数<br/>indices: 6-19<br/>eyeLOpen, browLY, ..."]
        HEAD_IDS["头部/身体参数<br/>indices: 31-39<br/>angleX/Y/Z, bodyAngleX/Y/Z"]
        OTHER_IDS["其他参数<br/>indices: 20-30, 40-44<br/>arms, hair, extras"]
        
        MOUTH_IDS -.-> MOUTH
        EYE_IDS -.-> EXPR
        HEAD_IDS -.-> HEAD_M
    end

    style INPUT fill:#e7f5ff,stroke:#339af0,color:#000
    style FINAL fill:#d3f9d8,stroke:#2b8a3e,color:#000
```

## 9. 训练数据对齐问题详解

```mermaid
flowchart LR
    subgraph Problem["⚠️ FPS 不匹配问题"]
        direction TB
        VIDEO_25FPS["预处理<br/>target_fps = 25"] -->|"25 帧/秒"| NPZ_25["NPZ<br/>live2d_params: (T, 45)<br/>T = duration × 25"]
        NPZ_25 --> DS["MotionDataset<br/>chunk_motion_frames = 50"]
        DS --> PAD["np.pad((0, 25), (0,0))<br/>⚠️ 50% 零填充!"]
        PAD --> MODEL["模型输入<br/>(50, 45)<br/>后半全是零"]
    end

    subgraph Solution["✅ 修复方案"]
        direction TB
        INTERP_FX["scipy.interpolate.interp1d<br/>25fps → 50fps 线性插值"]
        INTERP_FX --> MODEL_FX["模型输入<br/>(50, 45)<br/>全部有效数据"]
    end

    AUDIO_50["Hubert 编码器<br/>stride=320 @ 16kHz<br/>= 50 特征/秒"] -->|"50 帧/秒"| ALIGNED["音频与运动<br/>时间对齐"]

    style PAD fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style INTERP_FX fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style MODEL_FX fill:#d3f9d8,stroke:#2b8a3e,color:#000
```

## 10. 三模态性能引擎状态

```mermaid
flowchart TB
    START["▶ Start"] -->|"用户说话<br/>VAD 检测"| LISTEN

    subgraph LISTEN_STATE["🟢 Listen 模式"]
        direction TB
        L_INPUT["user_audio = 用户语音<br/>(非零张量)"]
        L_TTS["tts_audio = 零张量"]
        L_TEXT["text_prompt = ASR 转录"]
        L_CROSS["Cross-Attn KV =<br/>cat(user_audio_feat, visual_feat)"]
        L_MODE["mode_emb = listen_emb"]
    end

    subgraph SPEAK_STATE["🔵 Speak 模式"]
        direction TB
        S_INPUT["user_audio = 零张量"]
        S_TTS["tts_audio = TTS 合成音频<br/>(非零张量)"]
        S_TEXT["text_prompt = LLM 响应文本"]
        S_CROSS["Cross-Attn KV =<br/>cat(speak_audio_feat, text_feat)"]
        S_MODE["mode_emb = speak_emb"]
    end

    subgraph SILENCE_STATE["⚪ Silence 模式"]
        direction TB
        SI_INPUT["user_audio = 零张量"]
        SI_TTS["tts_audio = 零张量"]
        SI_TEXT["text_prompt = Perception 生成<br/>(用户正注视着...)"]
        SI_CROSS["Cross-Attn KV =<br/>cat(zero_feat, text_feat)"]
        SI_MODE["mode_emb = listen_emb"]
    end

    LISTEN_STATE -->|"用户停止说话<br/>+ LLM 完成响应"| SPEAK_STATE
    LISTEN_STATE -->|"用户静默<br/>超时阈值"| SILENCE_STATE

    SPEAK_STATE -->|"用户打断<br/>VAD 检测新语音"| LISTEN_STATE
    SPEAK_STATE -->|"TTS 播放完成"| SILENCE_STATE

    SILENCE_STATE -->|"用户开始说话<br/>VAD 检测"| LISTEN_STATE
    SILENCE_STATE -->|"LLM 完成响应<br/>+ TTS 音频就绪"| SPEAK_STATE

    style LISTEN_STATE fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style SPEAK_STATE fill:#d0ebff,stroke:#1971c2,color:#000
    style SILENCE_STATE fill:#f8f9fa,stroke:#868e96,color:#000
```

## 11. 问题影响链 (Mermaid)

```mermaid
flowchart TB
    C1["🔴 C1: Sigmoid + ε-prediction<br/>不兼容"] -->|"训练无法收敛"| NO_TRAIN["❌ 无法训练"]
    C2["🔴 C2: FPS 25→50<br/>50% 零填充"] -->|"模型学坏模式"| BAD_MOTION["❌ 运动后半段衰减"]
    C2 -->|"与 C1 叠加"| NO_TRAIN

    H1["🟠 H1: DDIM 推理<br/>公式错误"] -->|"推理质量差"| BAD_INFER["❌ 推理输出无意义"]
    C1 -->|"修复后需重写"| H1_FX["DDIM x-prediction"]
    
    H2["🟠 H2: LoRA<br/>无推理加载"] -->|"角色特化不可用"| NO_CHAR["❌ 无法切换角色"]
    
    M1["🟡 M1: 零视觉帧<br/>视觉通路未训练"] -->|"推理时视觉无效"| NO_VIS["⚠️ 摄像头感知浪费"]
    
    M2["🟡 M2: weight_decay<br/>未传入优化器"] -->|"可能过拟合"| OVERFIT["⚠️ 小数据集过拟合"]
    
    M3["🟡 M3: T=50<br/>硬编码"] -->|"变长音频不同步"| DESYNC["⚠️ 音画不同步"]
    M4["🟡 M4: warmup<br/>未实现"] -->|"训练初期不稳定"| UNSTABLE["⚠️ 早期训练波动"]

    style C1 fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style C2 fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style H1 fill:#ffd43b,stroke:#e67700,color:#000
    style H2 fill:#ffd43b,stroke:#e67700,color:#000
    style M1 fill:#ffe066,stroke:#e67700,color:#000
    style M2 fill:#ffe066,stroke:#e67700,color:#000
    style M3 fill:#ffe066,stroke:#e67700,color:#000
    style M4 fill:#ffe066,stroke:#e67700,color:#000
    style NO_TRAIN fill:#1a1a2e,stroke:#e03131,color:#fff
    style BAD_INFER fill:#1a1a2e,stroke:#e67700,color:#fff
    style NO_CHAR fill:#1a1a2e,stroke:#e67700,color:#fff
```

---

*图中红色 (⚠️ / 🔴) 标记为已知问题，详见 `docs/TRAINING_PIPELINE_REVIEW.md`。*