# ADR 0001: Base Model + LoRA Character Weights Architecture

## Status

Accepted

## Context

Amadeus needs a Performance Engine that generates character-specific motion (Live2D parameters) from multimodal input. The user wants:

1. A single base model that learns universal human motion patterns
2. The ability to specialize for specific characters (e.g., Kurisu from Steins;Gate)
3. M2 8G / consumer GPU compatibility for both training and inference
4. Character weights that are small enough to distribute independently (~MB, not ~GB)

## Decision

**Base Model + Character LoRA architecture.**

Base Model (FullDuplexDiT, ~24M trainable params) trained on mixed data:
- Real human datasets (MEAD, BIWI, VOCASET) — foundation for natural, diverse motion
- Anime character data — higher expressiveness for certain motion patterns

Character-specific weights via LoRA (Low-Rank Adaptation):
- Base Model DiT layers are frozen
- Each character gets a small trainable LoRA matrix per DiT layer
- Character LoRA files are ~MB in size — easily distributed and hot-swapped

Training data format: ground truth = Live2D parameter values (45 floats per frame). Skeleton extraction (YOLOv8/MediaPipe) is an intermediate preprocessing step.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| **Identity Embedding only** (320-dim per character) | Cannot capture complex behavioral patterns (e.g., body posture, gesture style) |
| **Full model fine-tuning per character** | Too large (~100MB per character), too slow to swap, expensive to train many characters |
| **Persona YAML only** (text-based control) | Text prompts cannot encode detailed motion characteristics learned from video data |
| **Training directly on skeleton coordinates** | Adds inference-time conversion step; model doesn't optimize directly in Live2D parameter space |

## Consequences

- ✅ One shared Base Model — training cost amortized across all characters
- ✅ Small LoRA files — practical for distribution and runtime swapping
- ✅ Character identity preserved through learned motion style, not just appearance
- ⚠ Training data pipeline is non-trivial: video → skeleton → Live2D parameters requires preprocessing tooling
- ⚠ LoRA training requires Base Model checkpoint first — Base Model must be trained before any character LoRA
