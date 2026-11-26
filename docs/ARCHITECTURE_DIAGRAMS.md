# Amadeus 算法架构图 (Mermaid)

> **状态**: 反映 `fix/training-pipeline-issues` 分支合并后的当前代码
> **更新日期**: 2026-06-07
> **作用**: 用 Mermaid 图表描述当前系统的算法结构、训练流程和推理路径
>
> **问题历史**: 修复过程中发现的问题及解决方案见 `docs/TRAINING_PIPELINE_REVIEW.md` 和 `docs/adr/0002-x-prediction-and-50hz-alignment.md`

---

## 目录

1. [系统数据流总览](#1-系统数据流总览)
2. [FullDuplexDiT 模型架构](#2-fullduplexdit-模型架构)
3. [DiT Block 内部结构](#3-dit-block-内部结构)
4. [训练管线流程](#4-训练管线流程)
5. [推理管线流程](#5-推理管线流程)
6. [LoRA 微调架构](#6-lora-微调架构)
7. [预处理管线数据流](#7-预处理管线数据流)
8. [性能参数后处理详解](#8-性能参数后处理详解)
9. [训练数据 fps 对齐](#9-训练数据-fps-对齐)
10. [三模态性能引擎状态](#10-三模态性能引擎状态)
11. [x-Prediction 训练全景](#11-x-prediction-训练全景)

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

    subgraph MotionPipeline["运动管线"]
        MIC -->|"原始音频"| USR_AUDIO["user_audio<br/>(16kHz)"]
        TTS_AUDIO --> TTS_AUDIO_BUF["tts_audio<br/>(16kHz)"]
        PERCEPT -->|"text_prompt"| TEXT_PROMPT["情绪描述文本"]

        USR_AUDIO --> DIT["FullDuplexDiT<br/>Mini-LPM<br/>(x-prediction)"]
        TTS_AUDIO_BUF --> DIT
        CAM_FRAMES["visual_frames<br/>(5×224×224)"] --> DIT
        TEXT_PROMPT --> DIT
        CHAR_ID["identity_id"] --> DIT

        DIT -->|"T × 45 params<br/>T = Hubert 推导"| PERF["PerformanceEngine<br/>(Persona 后处理)"]
        PERF -->|"调整后参数"| RENDER["Live2D Renderer<br/>(live2d-py + PySide6)"]
    end

    style DIT fill:#1a1a2e,stroke:#4a90d9,color:#fff
    style PERF fill:#51cf66,stroke:#2b8a3e,color:#fff
    style RENDER fill:#4a90d9,stroke:#2b5d8e,color:#fff
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

    subgraph Encoders["冻结编码器 (~101M)"]
        HE["HubertEncoder<br/>facebook/hubert-base-ls960<br/>768-dim @50Hz<br/>(Listen/Speak 共享)"]
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

    subgraph OutputHead["输出头 (Sigmoid 约束 ∈ [0, 1])"]
        direction TB
        TRANS["Transpose<br/>(B, T, 320) → (B, 320, T)"]
        C1["Conv1d(320, 320, k=5)<br/>+ GELU + Dropout"]
        C2["Conv1d(320, 160, k=5)<br/>+ GELU + Dropout"]
        C3["Conv1d(160, 45, k=5)<br/>+ Sigmoid"]
        OUT["output<br/>(B, T, 45) ∈ [0, 1]<br/>(x_0 预测)"]
    end

    BLOCKS --> TRANS --> C1 --> C2 --> C3 --> OUT

    NOISE["noisy_params<br/>(B, T, 45)"] --> X["x = noisy_params"] --> BLOCKS

    style C3 fill:#d0bfff,stroke:#7950f2,color:#000
    style HE fill:#fff5d6,stroke:#e67700,color:#000
    style C2 fill:#d0bfff,stroke:#7950f2,color:#000
    style OUT fill:#d3f9d8,stroke:#2b8a3e,color:#000
```

> **Sigmoid 含义**: 输出被约束在 `[0, 1]`,与 Live2D 参数范围天然对齐。配合 **x-prediction** 损失 (`loss = MSE(pred, motion)`),Sigmoid 是模型的正确约束,不再是不兼容的 bug。

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
        LOAD --> RESAMPLE["_resample_motion()<br/>按 .npz 中 fps 字段<br/>线性插值 → 50Hz"]
        RESAMPLE --> CHUNk["随机块选择<br/>chunk_duration=1.0s"]
        CHUNk --> BATCH["{'user_audio': (16000,),<br/>'tts_audio': zeros,<br/>'visual_frames': zeros,<br/>'text_prompt': '',<br/>'identity_id': 0,<br/>'motion': (50, 45)}"]
    end

    subgraph TrainingLoop["训练循环 (train.py)"]
        direction TB
        BATCH --> DIFF["DDPM 前向过程<br/>t ~ Uniform(0, 999)<br/>noise ~ N(0,1)<br/>noisy = √ᾱ_t · x_0 + √(1-ᾱ_t) · noise"]
        DIFF --> PRED["model(audio, tts, visual, prompt, id, t, noisy)"]
        PRED --> LOSS["loss = MSELoss(pred, motion)<br/>(x-prediction:<br/>Sigmoid 输出 ∈ [0,1]<br/>与 motion ∈ [0,1] 对齐)"]
        LOSS -->|"backward"| GRAD["梯度累积 × grad_accum_steps<br/>+ grad clip (max_norm=1.0)"]
        GRAD --> OPT["AdamW<br/>weight_decay=0.01"]
        OPT -->|"每 step"| SCHED["SequentialLR<br/>LinearLR (warmup) →<br/>CosineAnnealingLR"]
        SCHED --> EMA["EMA 更新<br/>(可选, decay=0.999)"]
        EMA -->|"每 epoch"| VAL["验证集评估 (10%)<br/>早停 patience=N"]
        VAL -->|"每 10 epoch"| CKPT["完整快照检查点<br/>model + optimizer +<br/>scheduler + scaler +<br/>EMA + epoch"]
    end

    subgraph LoRA["LoRA 微调 (--use_lora)"]
        direction TB
        BASE_CKPT["基础模型检查点"] --> LORA_APPLY["apply_lora(model, config)<br/>冻结所有参数<br/>替换 Linear/Conv1d → LoRALinear/LoRAConv1d"]
        LORA_APPLY --> LORA_TRAIN["仅 LoRA A/B 矩阵可训练<br/>rank=8, alpha=16<br/>≈500K 参数/角色"]
        LORA_TRAIN --> LORA_SAVE["save_lora() → lora_adapter.pt<br/>仅保存 A/B 矩阵 (~MB)"]
    end

    style RESAMPLE fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style LOSS fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style OPT fill:#d0ebff,stroke:#1971c2,color:#000
    style SCHED fill:#d0ebff,stroke:#1971c2,color:#000
    style CKPT fill:#fff5d6,stroke:#e67700,color:#000
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
        CHUNK --> T_DERIVE["从 Hubert 编码器输出<br/>推导 T (动态,非硬编码)"]
        TTS_BUF --> T_DERIVE
        VIS_BUF --> PREP_VIS["_prepare_visual_frames<br/>(5帧采样)"]
        PREP_VIS --> T_DERIVE

        subgraph DDIM["x-Prediction DDIM (4步, η=0)"]
            direction TB
            INIT["x_T ~ N(0,1)<br/>(B, T, 45)"]
            STEP1["t=999→749<br/>pred_x0 = model(...)<br/>pred_eps = (x_t - √ᾱ_t·pred_x0) / √(1-ᾱ_t)"]
            STEP2["t=749→499<br/>x_{t-1} = √ᾱ_{t-1}·pred_x0<br/>+ √(1-ᾱ_{t-1}-σ²)·pred_eps"]
            STEP3["t=499→249 (同 STEP2)"]
            STEP4["t=249→0 (同 STEP2, 无 noise)"]
            INIT --> STEP1 --> STEP2 --> STEP3 --> STEP4
        end

        T_DERIVE --> DDIM
        DDIM -->|"T frames × 45 params"| PERF["PerformanceEngine<br/>(Persona 后处理)"]

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

        LORA_HOT["set_character_id()<br/>(运行时热加载)<br/>→ remove + apply + load_lora<br/>+ merge_lora"]
    end

    style DDIM fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style T_DERIVE fill:#d0ebff,stroke:#1971c2,color:#000
    style LORA_HOT fill:#fff5d6,stroke:#e67700,color:#000
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
        INTERP["_interpolate_bad_frames()<br/>坏帧 ← 邻居均值<br/>O(n) list 查找"]
        CLIP["np.clip(result, 0, 1)"]
    end

    L2D --> INTERP --> CLIP

    CLIP --> NPZ_OUT[".npz 输出<br/>──────────────<br/>live2d_params (T, 45)<br/>blendshapes (T, 52)<br/>head_angles (T, 3)<br/>bad_frames (N,)<br/>fps, duration_sec<br/>identity_id, source_video"]

    AUDIO_OUT --> AUDIO_META["_audio.wav<br/>伴随 NPZ"]

    NPZ_OUT --> DS["MotionDataset<br/>._resample_motion()<br/>按 npz.fps → 50Hz"]
    AUDIO_META --> DS

    style DS fill:#d3f9d8,stroke:#2b8a3e,color:#000
```

> **fps 对齐**: 数据集读取 `.npz` 中的 `fps` 字段(预处理时存下来的源帧率),按需要线性插值到模型期望的 50 Hz 速率。无需修改预处理脚本。

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

## 9. 训练数据 fps 对齐

```mermaid
flowchart LR
    subgraph Source["源数据 (任意 fps)"]
        V["视频文件"]
    end

    V --> PP["PreprocessPipeline<br/>(target_fps=25 / 30 / 自定义)"]
    PP --> NPZ[".npz 文件<br/>live2d_params (T, 45)<br/>T = duration × source_fps<br/>+ source_fps 字段"]

    NPZ --> DS["MotionDataset.__getitem__"]

    subgraph Resample["按需重采样"]
        direction TB
        READ_FPS["读取 npz.fps"]
        RESAMPLE_MOTION["_resample_motion(<br/>arr, src_fps, dst_fps=50)<br/>np.interp 线性插值<br/>新长度 = round(T × 50/src_fps)"]
        READ_FPS --> RESAMPLE_MOTION
    end

    DS --> READ_FPS
    RESAMPLE_MOTION --> ALIGNED["对齐后的运动<br/>(T', 45), T' = 50 × duration<br/>(50Hz)"]

    subgraph Audio["音频对齐"]
        WAVE["_audio.wav<br/>16kHz mono"]
        HUB["Hubert 编码器<br/>stride=320 @ 16kHz<br/>= 50 特征/秒"]
        WAVE --> HUB
    end

    HUB --> AUDIO_FEAT["audio features<br/>(50Hz)"]

    AUDIO_FEAT --> MODEL["FullDuplexDiT<br/>音频 (50Hz) + 运动 (50Hz)<br/>时间轴完全对齐"]

    style RESAMPLE_MOTION fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style ALIGNED fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style MODEL fill:#1a1a2e,stroke:#4a90d9,color:#fff
```

> **数据流保证**: 无论预处理以多少帧率提取,数据集都会在加载时重采样到 50 Hz,与 Hubert 的 stride 完全对齐。模型永远不会收到 zero-padded 输入。

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

## 11. x-Prediction 训练全景

```mermaid
flowchart TB
    subgraph INPUT["训练输入"]
        direction TB
        AUDIO["user_audio<br/>(16000 samples)"]
        MOTION["motion<br/>(50, 45) ∈ [0, 1]"]
        VISUAL["visual_frames<br/>(5, 3, 224, 224)"]
        PROMPT["text_prompt"]
        ID["identity_id"]
    end

    subgraph DIFFUSION["x-Prediction Diffusion"]
        direction TB
        T_SAMPLE["t ~ Uniform(0, 999)"]
        NOISE["ε ~ N(0, I)"]
        FORWARD["noisy = √ᾱ_t · motion<br/>+ √(1-ᾱ_t) · ε"]
        T_SAMPLE --> FORWARD
        NOISE --> FORWARD
    end

    subgraph MODEL["FullDuplexDiT (x-prediction)"]
        direction TB
        ENC["冻结编码器<br/>(Hubert + MobileNet + BERT)"]
        PROJ["投影层 + TimestepEmbedding<br/>+ IdentityEmbedding + ModeEmbedding"]
        DIT["DiT × 4<br/>(Listen/Speak 交错)"]
        HEAD["Conv1d × 3 + Sigmoid<br/>→ pred_x0 ∈ [0, 1]"]
        ENC --> PROJ --> DIT --> HEAD
    end

    subgraph LOSS["损失与优化"]
        direction TB
        X_LOSS["loss = MSE(pred_x0, motion)<br/>(Sigmoid 输出与 [0,1] target 兼容)"]
        BACKWARD["backward + 梯度累积<br/>+ grad clip (max_norm=1.0)"]
        OPT["AdamW(lr=1e-4, weight_decay=0.01)"]
        SCHED["SequentialLR<br/>(LinearLR warmup → Cosine)"]
        EMA["EMA 更新 (可选)<br/>decay=0.999"]
        EARLY["早停 patience=N<br/>(验证损失驱动)"]
        X_LOSS --> BACKWARD --> OPT --> SCHED --> EMA --> EARLY
    end

    subgraph OUTPUT["训练产出"]
        direction TB
        SNAP["完整快照检查点<br/>model + optimizer +<br/>scheduler + scaler +<br/>EMA + epoch"]
        LORA["LoRA 适配器<br/>(若 --use_lora)<br/>~MB/角色"]
        NEXT["恢复训练 / 部署推理"]
        SNAP --> NEXT
        LORA --> NEXT
    end

    subgraph INFER["推理 (x-prediction DDIM)"]
        direction TB
        HOT["set_character_id()<br/>热加载角色 LoRA"]
        DDIM_INF["4步 x-DDIM<br/>T 从 Hubert 推导<br/>η=0 确定性"]
        PERF["PerformanceEngine<br/>(persona 后处理)"]
        LIVE2D["Live2D Renderer<br/>(60fps)"]
        HOT --> DDIM_INF --> PERF --> LIVE2D
    end

    INPUT --> DIFFUSION
    DIFFUSION -->|"noisy + 条件"| MODEL
    MODEL -->|"pred_x0"| LOSS
    LOSS --> OUTPUT
    NEXT -->|"checkpoint"| INFER

    style MOTION fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style X_LOSS fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style EMA fill:#d0ebff,stroke:#1971c2,color:#000
    style SNAP fill:#fff5d6,stroke:#e67700,color:#000
    style LORA fill:#d0bfff,stroke:#7950f2,color:#000
    style DDIM_INF fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style LIVE2D fill:#1a1a2e,stroke:#4a90d9,color:#fff
```

---

*本文件反映 `fix/training-pipeline-issues` 分支合并后的代码状态。所有训练阻断问题 (C1/C2) 与推理缺陷 (H1/H2) 均已修复;质量改进 (M1–M4) 与最佳实践 (L2–L5) 已集成。完整问题史与修复决策见 `docs/TRAINING_PIPELINE_REVIEW.md` 和 `docs/adr/0002-x-prediction-and-50hz-alignment.md`。*