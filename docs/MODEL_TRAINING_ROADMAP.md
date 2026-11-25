# Amadeus Model Training Roadmap

> Goal: make the model path practical for a solo developer who wants long-term local deployment, not just a one-off research demo.

## Executive Summary

The current direction is broadly right:

- Generate **Live2D parameters**, not pixels.
- Separate **Base Model** from **Character LoRA**.
- Use **x-prediction** because Live2D parameters naturally live in `[0, 1]`.
- Use HDTF/MEAD-style talking-head data to validate the training pipeline early.

But the current path is too close to the final FullDuplexDiT shape too early. The risky version is:

```text
HDTF -> train FullDuplexDiT base model -> train character LoRA -> realtime Live2D performance
```

That path can show falling loss while still failing the product goal: a character that feels alive during speaking, listening, and silence.

The safer route is:

```text
Speak first -> rule-based Silence/Listen -> character style LoRA -> unified FullDuplexDiT later
```

In one sentence:

> First train a small, verifiable Speak Performance model; use rules to cover Silence and Listen; then gradually replace rules with data-driven modules as the dataset catches up.

## Why The Current Direction Is Still Worth Keeping

### Live2D Parameter Output Is The Right Target

Generating Live2D parameters is the right deployment choice. It keeps inference light, makes consumer hardware realistic, and allows the Renderer to stay deterministic.

Pixel/video generation would make the project much harder to deploy long term:

- larger models
- heavier GPUs
- more latency
- harder identity consistency
- harder asset control

Live2D parameter generation also gives better debugging locality: if mouth motion is wrong, inspect `ParamMouthOpenY`; if head pose drifts, inspect `ParamAngleX/Y/Z`.

### Base Model + Character LoRA Is The Right Split

The Base Model should learn broad human motion patterns:

- mouth movement
- blink timing
- head motion
- facial expression dynamics
- audio-to-motion timing

Character LoRA should learn style:

- gesture amplitude
- expression sharpness
- head movement range
- mouth openness style
- idle energy
- character-specific exaggeration

This split is suitable for a solo developer because each new Character should cost megabytes, not a full retrain.

### x-prediction Is A Good Fit

Because Live2D parameters are bounded, x-prediction is simpler than epsilon-prediction:

```text
model output = clean Live2D parameter sequence x_0
loss = MSE(pred_x0, ground_truth_motion)
```

This keeps training and inference aligned with the actual output space.

## Main Risk

The biggest risk is not the architecture. It is the data.

FullDuplexDiT expects to learn three states:

| State | Desired behavior | Current data fit |
|---|---|---|
| Speak | Character talks with mouth, face, and head motion | Good enough with talking-head data |
| Listen | Character reacts while the user speaks | Weak unless using real conversation listener data |
| Silence | Character idles naturally while waiting | Weak unless separately collected or procedurally generated |

HDTF is useful, but it mainly teaches talking-head speech motion. It does not solve natural listening or silence behavior.

So the model path should avoid asking one model to learn all three states before the data supports them.

## Recommended Training Phases

## Phase 0: Build A Non-neural Baseline

Before training bigger models, create a simple baseline that can drive Live2D parameters from audio energy and state.

Minimum baseline:

- RMS/phoneme-like mouth openness
- blink scheduler
- breathing parameter
- small head noise
- expression smoothing
- simple Listen reaction when user speech starts
- simple Silence idle loop

Why this matters:

- gives a quality floor
- makes demos stable before model quality is high
- exposes whether the neural model is actually better
- keeps the app deployable even when model assets are missing

Acceptance criteria:

- app can run without FullDuplexDiT checkpoint
- mouth opens in sync with speech energy
- idle does not look frozen
- Listen state has visible but subtle reaction

## Phase 1: Train Speak Performance Only

Train the first neural target only on Speak behavior.

Recommended input/output:

```text
input: speech audio or TTS audio
output: Live2D face/head parameters
state: Speak only
```

Do not require user camera, text prompt, identity conditioning, or Listen/Silence behavior in the first useful model.

Use this phase to verify:

- preprocessing correctness
- 50 Hz alignment
- x-prediction loss
- checkpoint save/resume
- inference sampling
- Live2D parameter mapping
- qualitative playback

Datasets:

- HDTF for pipeline smoke tests
- MEAD if accessible for expression diversity
- any talking-head data with usable audio
- later, recorded TTS audio paired with generated/retargeted Live2D motion

Acceptance criteria:

- loss decreases on train and validation
- generated mouth/head motion is visibly aligned with audio
- output does not collapse to neutral
- inference is stable for at least 60 seconds of audio
- model beats the non-neural baseline in side-by-side playback

## Phase 2: Add A Training Run Manifest

Before running many experiments, every run should produce a manifest.

Each run should save:

- git commit
- command line
- config snapshot
- dependency snapshot
- dataset path
- dataset hash or sample manifest
- train/val split seed
- checkpoint paths
- loss curve
- qualitative sample outputs
- resume smoke-test result

This matters more than it sounds. Long-term solo development fails when a good checkpoint exists but nobody remembers how it was produced.

Recommended artifact layout:

```text
models/runs/2026-06-19-speak-base-hdtf-300/
  run.yaml
  config.yaml
  dataset_manifest.json
  train.log
  loss_curve.png
  samples/
  checkpoints/
  final/
```

Acceptance criteria:

- every model can be traced back to data and config
- resume is tested on at least one checkpoint
- a future run can compare itself to the previous best model

## Phase 3: Keep Silence Procedural First

Do not train Silence first.

Silence should initially be generated by a small deterministic module:

```text
Conversation state + elapsed time + persona performance params -> idle Live2D params
```

Good Silence behavior can be mostly procedural:

- breathing
- blinking
- eye movement
- small head movement
- attention direction
- idle energy
- emotion decay from previous response

This is more controllable than asking FullDuplexDiT to infer Silence from weak or missing data.

Acceptance criteria:

- character does not freeze while waiting
- idle is subtle and not distracting
- persona performance parameters visibly affect idle style
- Silence can be disabled or made deterministic for tests

## Phase 4: Treat Listen As A Data Problem

Listen is the hardest state. It needs examples where one person is speaking and the visible person is listening.

HDTF-style talking-head clips are not enough.

Useful Listen data would include:

- real two-person conversation video
- listener face visible while speaker audio is present
- speaker/listener turn labels
- natural nods, gaze shifts, micro-expressions
- enough silence gaps to learn transitions

Until that exists, Listen should be a hybrid:

```text
AudioPipeline state + user speech energy + recent dialogue context -> rule/small-model listener reaction
```

Possible first implementation:

- speech onset -> tiny attention reaction
- sustained speech -> low-amplitude nod/head drift
- long user speech -> occasional blink/gaze change
- strong user emotion -> expression bias if Perception detects it

Acceptance criteria:

- Listen behavior is visible but restrained
- user interruption does not break TTS or motion state
- rules produce better behavior than random FullDuplexDiT output

## Phase 5: Train Character LoRA After Speak Is Stable

Character LoRA should not be the first thing trained. It should specialize a working Base Model.

Start with a narrow LoRA goal:

- expression amplitude
- mouth openness style
- head movement range
- gesture energy
- idle style, if procedural parameters are included

Avoid expecting LoRA to learn complete character personality or full-body acting early.

Data sources:

- character clips
- anime/game footage where the target character is visible
- manually curated clips with clean face/body framing
- generated Live2D motion clips if retargeting is reliable

Acceptance criteria:

- same audio produces visibly different style with LoRA vs base
- LoRA does not destroy mouth timing
- adapter can hot-swap at runtime
- adapter file remains small enough for per-character deployment

## Phase 6: Unify Into FullDuplexDiT Later

Only after Speak, Silence, and Listen each have a reliable source should FullDuplexDiT become the unified Performance Engine.

At that point the model can reasonably consume:

- user audio
- TTS audio
- visual frames
- text prompt
- character ID
- previous motion state

The unified model should replace rules gradually, not all at once.

Suggested migration:

1. FullDuplexDiT handles Speak.
2. Procedural module handles Silence and Listen.
3. Add learned Listen when listener data exists.
4. Add learned Silence when idle data exists.
5. Keep procedural fallback for deployment safety.

## Dataset Strategy

## HDTF

Use HDTF for:

- preprocessing smoke tests
- audio/face alignment checks
- early Speak training
- regression tests for training pipeline

Do not treat HDTF as sufficient for:

- natural listening
- silence behavior
- character-specific style
- full-body performance

## MEAD / BIWI / VOCASET

Use these to increase expression and identity diversity if access is practical.

They are more valuable after the pipeline can already train and evaluate Speak.

## Character Data

Character data should be curated later and used mainly for LoRA.

Important constraints:

- clips must match the visible Character reasonably well
- face/body framing must be stable
- extracted Live2D parameters must be inspected visually
- bad clips should be rejected aggressively

Poor character data will teach worse style than no character data.

## Synthetic / Procedural Data

Procedural data is useful for:

- Silence
- blink timing
- breathing
- head drift
- regression tests
- stabilizing output ranges

It should not be hidden as if it were real data. Mark it explicitly in run manifests.

## Evaluation Plan

Loss alone is not enough.

Each training stage should have three evaluation layers:

### 1. Numeric

- train loss
- validation loss
- per-parameter MSE
- mouth/head/expression subgroup loss
- output range violations
- temporal smoothness

### 2. Visual

- parameter curve plots
- ground truth vs prediction overlays
- generated GIF/video previews
- side-by-side baseline comparison

### 3. Runtime

- 60-second inference stability
- TTS audio alignment
- no parameter explosions
- no frozen output
- no visible jitter
- acceptable latency on target hardware

## Recommended Milestones

| Milestone | Output | Done when |
|---|---|---|
| M0 Baseline | Rule-based Performance module | App looks alive without neural model |
| M1 Speak Dataset | Clean training subset | Samples render correctly as Live2D curves/GIFs |
| M2 Speak Base | First useful checkpoint | Beats baseline on speaking playback |
| M3 Run Manifest | Reproducible training artifacts | Every checkpoint has config/data lineage |
| M4 Procedural Silence | Stable idle behavior | Character no longer freezes |
| M5 Hybrid Listen | Rule/small-model listener reactions | User speech produces natural restrained response |
| M6 Character LoRA | First style adapter | Same audio has character-specific motion |
| M7 Unified Engine | FullDuplexDiT covers more states | Learned modules replace rules where data supports it |

## What To Avoid

Avoid these traps:

- training the full multimodal model before Speak is validated
- treating HDTF as proof that Listen/Silence will work
- judging success only by loss
- using random weights as a silent runtime fallback
- training LoRA before the Base Model is visually acceptable
- mixing real and synthetic data without manifest labels
- keeping good checkpoints without run metadata
- making FullDuplexDiT responsible for missing data problems

## Recommended Next Action

The next practical step is:

1. Add or improve a non-neural Performance baseline.
2. Fix checkpoint resume reliability.
3. Train and evaluate a Speak-only model.
4. Add run manifests before scaling experiments.
5. Keep Silence and Listen procedural until the right data exists.

This preserves the LPM-inspired long-term vision, while keeping the project usable and maintainable for one developer.
