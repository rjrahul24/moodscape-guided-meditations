<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/heart_mula/engine.py`
**Class:** `HeartMulaEngine` — `load_model()` / `generate()` / `_generate_mlx()` / `_generate_mps()`
**Constants:** `_LM_CFG_SCALE=1.8` · `_LM_TEMPERATURE=0.75` · `_LM_TOP_K=30` · `_CODEC_GUIDANCE_SCALE=1.25` · `_CODEC_NUM_STEPS=12` · `MAX_SEGMENT_SEC=240.0` · `CROSSFADE_SEC=8.0`
**Contract:** Output — 48 kHz mono float32 · MPS ckpts `./ckpt/` · MLX ckpts `./ckpt-mlx/`
**CRITICAL:** HeartCodec always fp32 (bf16 → metallic artifacts) · `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`
**Simultaneous loading (MLX):** Both LM (bf16) + codec (fp32) loaded together (~12 GB) → generate → decode → unload all
**Tasks:**
- Tune generation params → module-level constants at top of `engine.py`
- Change prompt enhancement → `core/pipeline.py :: _enhance_heartmula_prompt()` (Eight Pillars tags)
- Adjust FX chain → `core/audio_processor.py :: make_heartmula_music_chain()`
- Memory issues → check `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and simultaneous loading budget
- Best-of-N quality mode → `quality_mode=True` in `generate()`
**See also:** `docs/ARCHITECTURE.md#heartmulaengine` · `docs/prompting_guides/heartmula_instructions.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# HeartMuLa Implementation Guide

## Overview

HeartMuLa is a two-stage autoregressive text-to-music pipeline used in MoodScape as a high-quality local music generation engine alongside ACE-Step 1.5 and Lyria RealTime.

| Property | Value |
|----------|-------|
| **Architecture** | HeartMuLa LM (3B params) + HeartCodec (12.5 Hz neural audio codec) |
| **License** | Apache 2.0 |
| **Output** | 48,000 Hz stereo (downmixed to mono in the engine) |
| **Backend** | MLX primary (heartlib-mlx, ~2x faster), PyTorch MPS fallback (heartlib) |
| **Max single pass** | 240 seconds (4 minutes) |
| **Long-form** | Segment-and-crossfade with tonal key anchoring (8s equal-power cosine crossfades) |
| **Memory (M1 Max)** | ~12 GB total (LM bf16 + codec fp32 loaded simultaneously) |

## Two-Stage Pipeline

1. **HeartMuLa LM (3B params)** — A causal language model that autoregressively generates discrete acoustic token sequences conditioned on style tags and optional lyrics. Default dtype: `bf16` (safe for MPS/MLX inference).

2. **HeartCodec** — A 12.5 Hz frame-rate neural audio codec that decodes those tokens into waveform audio. Default dtype: `fp32`. **Never use `bf16` for HeartCodec — it degrades audio quality.**

Both stages are loaded simultaneously on MLX (~12 GB total). After generation and decoding, all weights are unloaded. Memory limits (`mx.set_memory_limit(30GB)`, `mx.set_cache_limit(4GB)`) keep usage within the 36 GB unified memory budget.

## Hardware Requirements

| Hardware | Support | Notes |
|----------|---------|-------|
| Apple Silicon (M1/M2/M3) | Full | MLX backend for ~2x speed; MPS fallback available |
| CUDA GPU | Supported | Change device to `"cuda"` in engine |
| CPU | Fallback | Very slow; not recommended |

**Minimum RAM:** 16 GB unified memory (model weights ~6 GB bf16 + KV cache)
**Recommended:** 36 GB unified memory (Apple Silicon) for comfortable headroom

## Installation

### 1. Install heartlib

```bash
# Official PyTorch backend
pip install git+https://github.com/HeartMuLa/heartlib.git

# Optional: MLX backend (2x faster on Apple Silicon)
pip install git+https://github.com/Acelogic/heartlib-mlx.git
```

### 2. Download model weights

```bash
# PyTorch MPS path (./ckpt/)
huggingface-cli download HeartMuLa/HeartMuLa-RL-oss-3B-20260123 \
    --local-dir ./ckpt/HeartMuLa-oss-3B

huggingface-cli download HeartMuLa/HeartCodec-oss-20260123 \
    --local-dir ./ckpt/HeartCodec-oss

huggingface-cli download HeartMuLa/HeartMuLaGen \
    tokenizer.json gen_config.json \
    --local-dir ./ckpt
```

### 3. (Optional) Convert to MLX format

```bash
python -m heartlib_mlx.utils.convert
# Stores converted weights in ./ckpt-mlx/
```

### Checkpoint Directory Structure

```
./ckpt/                         <- PyTorch MPS path
  HeartMuLa-oss-3B/             <- LM weights
  HeartCodec-oss/               <- Codec weights
  gen_config.json
  tokenizer.json

./ckpt-mlx/                     <- MLX path (after conversion)
  heartmula/                    <- Converted LM weights
  heartcodec/                   <- Converted codec weights
```

## Device & Dtype Strategy

| Component | Dtype | Reason |
|-----------|-------|--------|
| HeartMuLa LM | `bf16` | Reduces memory by ~50%; no impact on token prediction quality |
| HeartCodec | `fp32` | **CRITICAL**: `bf16` causes metallic artifacts in decoded audio |

### Backend Selection (automatic)

The engine tries MLX first, then falls back to PyTorch MPS:

```python
# In HeartMulaEngine.load_model():
try:
    import heartlib_mlx  # MLX backend — ~2x faster
    self._backend = "mlx"
except ImportError:
    import heartlib      # PyTorch MPS fallback
    self._backend = "mps"
```

## Generation Parameters

Optimised for meditation/sleep music with moderate constraints:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cfg_scale` | **1.8** | DPO-trained model follows tags at lower CFG. 3.0 over-constrains → repetitive/robotic. |
| `temperature` | **0.75** | More stable token distributions for sustained drones/pads. |
| `top_k` | **30** | Tighter pool prevents repetitive loops in ambient territory. |
| `codec_guidance_scale` | **1.25** | Paper Table 2: 1.25 = "smoother and less harsh." Reduces metallic artifacts. |
| `codec_num_steps` | **12** | Reflow-distilled for 10 steps; 12 gives marginal improvement, 16 was wasteful. |

## Prompt System — Eight Pillars

The pipeline enhances user prompts via `_enhance_heartmula_prompt()` following the **Eight Pillars** tag hierarchy from HeartMuLa research. The key principle is **"Less is More"** — excessive tags cause probability interference.

See `docs/prompting_guides/heartmula_instructions.md` for the full prompting guide.

**Tag structure (5-6 tags total):**
1. Core anchors: `deep ambient` genre + `60bpm` tempo (2 tags)
2. User tags (deduplicated, appended verbatim)
3. Negative floor: `no drums, instrumental` (2 tags)

**Structural lyrics** use `[inst-medium]` markers (not `[verse]`/`[bridge]`) to suppress vocal bias.

## Generation Pipeline

### Single Segment (<=240s)

```
tags + lyrics -> HeartMuLa LM (bf16) -> discrete tokens
                                         |
discrete tokens -> HeartCodec (fp32) -> stereo 48kHz waveform
                                         |
stereo -> mono -> peak normalize (-1 dBFS) -> safety clip -> float32 array
```

### Long-Form (>240s)

For 20-minute meditation tracks:

```
Total: 1200s -> ceil(1200/240) = 5 segments x 240s each

Tonal anchor: "key of D minor" (randomly selected, applied to ALL segments)
Token-level continuation: last 250 tokens of segment N fed as prefix to segment N+1

Segment 0: [intro-medium] + [inst-medium]     <- opening arrival
Segment 1: [inst-medium] + [inst-medium]      <- sustaining
Segment 2: [inst-medium] + [inst-medium]      <- sustaining
Segment 3: [inst-medium] + [inst-medium]      <- sustaining
Segment 4: [inst-medium] + [outro-medium]     <- closing dissolution

Each segment: generate -> postprocess -> collect
Join with STFT crossfade in log-magnitude domain (cosine² fallback)

Total output: 5x240s - 4x8s = 1168s ~ 19m28s
```

**Key anchoring** is the primary cross-segment coherence mechanism. A musical key tag constrains the LM to generate in a consistent key across all segments. **Token-level continuation** feeds the last 250 tokens of each segment as a prefix to the next, providing tonal context across boundaries. Segments are joined using **STFT crossfade in log-magnitude domain** for seamless spectral transitions (falls back to cosine² if STFT fails).

### Story Mode

One HeartMuLa call per stage. Each stage can have different tags for tonal evolution (e.g., centering -> depth -> integration). Stages exceeding 240s are automatically delegated to the long-form pipeline with key anchoring.

## Quality Mode (Best-of-N)

When `quality_mode=True` is passed to `generate()` (or the UI checkbox "High Quality (Best-of-3)" is enabled), the engine generates **N=3 candidate** audio segments using different random seeds. Each candidate is scored using `compute_composite_score()` from `core/qa_monitor.py` (weighted composite of all QA checks, range 0-1). The highest-scoring candidate is selected as the final output.

This triples generation time but significantly improves consistency, especially for longer segments where the autoregressive LM is more likely to drift.

## Temperature & CFG Scheduling

Rather than using fixed values throughout generation, the engine applies phase-aware scheduling:

| Phase | Temperature | CFG Scale |
|-------|-------------|-----------|
| Intro | 0.80 | 2.0 |
| Development | 0.85 | linear decay toward 1.0 |
| Resolution | 0.65 | 1.0 |

CFG follows a **2.0 → 1.0 linear decay** over the generation steps. This provides stronger tag adherence at the start (establishing the tonal character) while allowing organic evolution in later steps.

Implemented via monkey-patching heartlib-mlx's `generate()` method to inject per-step parameter overrides.

## Neural Post-Processing

An optional **Apollo GAN** (ICASSP 2025) pass removes codec artifacts from HeartMuLa output. Apollo loads after the music engine unloads (~7 GB) and applies neural upsampling / artifact removal. If `apollo` is not installed, the engine falls back gracefully to the standard FX chain without error.

## Post-Processing Chain

The `make_heartmula_music_chain()` in `core/audio_processor.py` applies:

| Step | Effect | Purpose |
|------|--------|---------|
| 1 | NoiseGate (-52 dB, 2:1) | Suppress codec quantization noise |
| 2 | HighpassFilter (60 Hz) | Remove sub-60 Hz codec rumble |
| 3 | LowShelfFilter (150 Hz, +2.0 dB) | Fletcher-Munson bass compensation for headphone listening |
| 4 | PeakFilter (220 Hz, -1.0 dB, Q=0.7) | Clarity cut; removes low-mid mud from HeartCodec |
| 5 | PeakFilter (4000 Hz, -2.0 dB, Q=0.5) | Tame upper-mid brightness; wider Q for transparency |
| 6 | HighShelfFilter (9500 Hz, -2.0 dB) | Warm HF rolloff for headphone listening |
| 7 | LowpassFilter (14000 Hz) | HF ceiling before compression |
| 8 | Convolution reverb (stone_chapel, 15% wet) | Spatial depth and natural decay |
| 9 | Compressor (-20 dB, 2:1, 100ms/900ms) | Slow meditative dynamics; prevents pumping |
| 10 | Limiter (-0.5 dBFS) | Safety limiter |

Pre-mix LUFS target: **-16.0 LUFS** (HeartMuLa output levels are consistent).

## Memory Management

Both LM and codec are loaded simultaneously (~12 GB total on a 36 GB system). Memory limits are configured at load time:

```python
# MLX backend — set before loading:
mx.set_memory_limit(30 * 1024**3)  # 30 GB hard ceiling
mx.set_cache_limit(4 * 1024**3)    # 4 GB evaluation cache

# After generation:
# 1. Reset KV cache
# 2. Clear activations
# 3. Decode audio
# 4. Unload all weights
gc.collect()
mx.clear_cache()
```

Peak memory usage is ~12 GB during generation, well within the 36 GB budget.

## Expected Performance on M1 Max

| Duration | Segments | Est. Time | Notes |
|----------|----------|-----------|-------|
| 1 min | 1 | 2-5 min | Quick test |
| 4 min | 1 | 8-20 min | Single segment max |
| 10 min | 3 | 25-60 min | Long-form |
| 20 min | 5 | 40-100 min | Full meditation |

MLX backend is approximately 2x faster than PyTorch MPS.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` for checkpoints | Download weights — see Installation above |
| `RuntimeError: MPS device not found` | Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set; check native ARM64 Python |
| Generation is extremely slow | Expected on MPS (~5-10x slower than CUDA). Use MLX backend or ACE-Step for faster results |
| Audio has quality degradation | Verify `codec_dtype="fp32"` — never use `bf16` for HeartCodec |
| Out of memory crash | Check `mx.set_memory_limit()` and `mx.set_cache_limit()` values. Close other GPU-intensive apps |
| MLX backend import error | Install: `pip install git+https://github.com/Acelogic/heartlib-mlx.git` |

## Integration with Pipeline

The engine is instantiated in `core/pipeline.py` when `music_model == "heartmula"`:

```python
from core.heart_mula.engine import HeartMulaEngine
music_engine = HeartMulaEngine()
music_engine.load_model()
audio = music_engine.generate(tags, duration, lyrics=lyrics, ...)
music_engine.unload_model()
```

The pipeline enhances user prompts via `_enhance_heartmula_prompt()` which builds concise Eight Pillars tags and `[inst-medium]`-based structural lyrics.
