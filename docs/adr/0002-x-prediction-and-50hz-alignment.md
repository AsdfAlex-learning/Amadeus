# ADR 0002: X-Prediction Diffusion and 50Hz Motion Alignment

## Status

Accepted (2026-06-07)

## Context

The FullDuplexDiT model produces 50 Live2D parameters per frame, at a
rate of 50 frames per second — chosen so the motion rate matches
Hubert's stride (320 audio samples at 16 kHz → 50 features/sec). The
first end-to-end training run, however, never converged, and a review
of the training pipeline (`docs/TRAINING_PIPELINE_REVIEW.md`) revealed
two independent training-blocking bugs that share the same goal: keep
the model's output and the training target in a numerically compatible
space.

### Problem 1 — Output head Sigmoid vs. ε-prediction

`model.py` ends the output head with `nn.Sigmoid()`, which constrains
the output to `[0, 1]`. The training loop, however, used the standard
DDPM ε-prediction loss:

```python
pred = model(audio, tts_audio, visual, prompts, identity, t, noisy)
loss = criterion(pred, noise)  # noise ~ N(0, 1)
```

Standard normal noise has unbounded range: about 68% of samples fall
outside `[0, 1]`, ~5% outside `[-2, 2]`. A Sigmoid cannot represent
negative values or values > 1. The model was asked to learn a
function outside its representational range. Gradients on the
saturated ends of the Sigmoid vanish, so the network could not
back-propagate useful signal.

### Problem 2 — 25 fps preprocessing vs. 50 fps model

`pipeline.py` extracts motion at the configurable `target_fps`
(default 25). `MotionDataset` then samples chunks of
`chunk_motion_frames = 50` frames (1 second at the model's 50 Hz rate).
The shorter 25-frame motion segments were **zero-padded** to 50:

```python
if len(motion_chunk) < self.chunk_motion_frames:
    pad_len = self.chunk_motion_frames - len(motion_chunk)
    motion_chunk = np.pad(motion_chunk, ((0, pad_len), (0, 0)))
```

Every training sample was 50% zero at the tail. The model learned
"motion decays to zero in the second half of every chunk", and the
regression-to-zero behaviour was baked into the weights.

## Decision

### C1 — Switch to x-prediction

We switch the diffusion parameterization from ε-prediction to
**x-prediction**: the model is asked to predict the clean sample `x_0`
directly.

- **Loss target**: `motion` (the ground truth, in `[0, 1]`)
- **Model output**: still bounded by Sigmoid, now naturally
  compatible with the target
- **Inference**: x-prediction DDIM step, computed analytically from
  `pred_x0`:

```
pred_eps = (x_t − √ᾱ_t · pred_x0) / √(1 − ᾱ_t)
x_{t-1}  = √ᾱ_{t-1} · pred_x0
          + √(1 − ᾱ_{t-1} − σ²) · pred_eps
          + σ · noise        (η = 0 → deterministic)
```

The Sigmoid is preserved (it's the natural constraint for the output
space) and the loss target is changed to match. Training and inference
are now consistent.

### C2 — Resample motion to 50 Hz in the dataset

The dataset reads the `fps` field from each `.npz` and linearly
resamples the motion (and head_angles) array to
`AUDIO_FEATURES_PER_SEC` (50) before chunk selection. This produces
real 50-frame chunks with no zero padding, regardless of the
preprocessing fps.

The resampling is a simple `np.interp` per parameter dimension; no
external dependency is added.

## Alternatives Considered

### For C1

| Alternative | Rejected Because |
|---|---|
| Keep ε-prediction, remove Sigmoid | The output is no longer bounded, so the model can produce out-of-range values that the Live2D renderer would have to clip — moves the constraint downstream without removing it. |
| Predict `v = √α·ε − √(1−α)·x_0` (v-prediction) | Same unbounded-target problem as ε-prediction. The benefit (better for high-noise steps) does not justify the complexity. |
| Use a tanh output scaled to `[0, 1]` | Adds a scale-and-shift constant that has to be tuned; Sigmoid is the natural choice. |

### For C2

| Alternative | Rejected Because |
|---|---|
| Re-run the preprocessing pipeline at 50 fps | Most user videos do not have motion that benefits from 50 fps capture (human face motion is bandwidth-limited). 25 fps capture is cheaper and produces good results after interpolation. Resampling is also reversible: if 50 fps source data ever becomes available, only the `fps` field needs to change. |
| Re-sample the audio to 25 Hz | Audio at 16 kHz is high-bandwidth; down-sampling would lose phonetic information. Motion is lower-bandwidth and resamples cleanly. |
| Use a learned upsampler network | Adds parameters that must be trained and a separate training pipeline. Linear interpolation is provably sufficient for piecewise-smooth motion at 25→50 fps (Nyquist is satisfied for the 0–12.5 Hz motion bandwidth). |

## Consequences

- ✅ Training becomes numerically well-posed — model can represent its
  target, gradients flow through Sigmoid into a useful region.
- ✅ Inference and training are coupled by the same parameterization;
  no fudge factors or per-step rescaling.
- ✅ All training samples contain real motion at 50 Hz, no zero pads.
- ✅ Changing the preprocessing `fps` no longer requires code changes —
  the dataset reads `fps` from the `.npz` and resamples to the model's
  50 Hz.
- ⚠ Inference required a complete DDIM rewrite (was a non-standard
  custom step in the original code). The new step is the canonical
  x-prediction DDIM formula.
- ⚠ All previously trained weights (none, since training never
  succeeded) would be incompatible. We re-train from scratch.

## Related ADRs

- [ADR 0001: Base Model + LoRA Character Weights](0001-base-model-lora-architecture.md) —
  the per-character LoRA fine-tuning architecture that benefits from a
  working base model.
