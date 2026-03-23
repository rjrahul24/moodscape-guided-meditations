<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/heart_mula/engine.py`
**Class:** `HeartMulaEngine` — `load_model()` / `generate()` / `_generate_mlx()` / `_generate_mps()`
**Constants:** `_LM_CFG_SCALE=3.0` · `_LM_TEMPERATURE=0.9` · `_LM_TOP_K=45` · `MAX_SEGMENT_SEC=240.0` · `CROSSFADE_SEC=8.0`
**Contract:** Output — 48 kHz mono float32 · MPS ckpts `./ckpt/` · MLX ckpts `./ckpt-mlx/`
**CRITICAL:** HeartCodec always fp32 (bf16 → metallic artifacts) · `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`
**Lazy-load sequence (MLX):** Load LM (bf16) → generate → `del + gc + mx.clear_cache()` → load codec (fp32) → decode → unload
**Tasks:**
- Tune generation params → module-level constants at top of `engine.py`
- Change prompt enhancement → `core/pipeline.py :: _enhance_heartmula_prompt()` (Eight Pillars tags)
- Adjust FX chain → `core/audio_processor.py :: make_heartmula_music_chain()`
- Memory issues → check `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and lazy-load sequence
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
| **Memory (M1 Max)** | ~6-8 GB LM (bf16) + ~2-4 GB codec (fp32) per phase (lazy loaded) |

## Two-Stage Pipeline

1. **HeartMuLa LM (3B params)** — A causal language model that autoregressively generates discrete acoustic token sequences conditioned on style tags and optional lyrics. Default dtype: `bf16` (safe for MPS/MLX inference).

2. **HeartCodec** — A 12.5 Hz frame-rate neural audio codec that decodes those tokens into waveform audio. Default dtype: `fp32`. **Never use `bf16` for HeartCodec — it degrades audio quality.**

Both stages are lazy-loaded: the LM loads, generates tokens, then unloads. Then HeartCodec loads, decodes audio, then unloads. This is critical for the 36 GB unified memory budget.

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
| `cfg_scale` | **3.0** | Moderate tag adherence (default 1.5 too loose). Values above 4.0 risk early EOS / mode collapse. |
| `temperature` | **0.9** | Slightly below default (1.0) for more grounded ambient output. |
| `top_k` | **45** | Slight restriction from default (50) keeps the LM in ambient territory. |
| `codec_guidance_scale` | **1.5** | Slightly above default (1.25) for better sustained pad fidelity. |
| `codec_num_steps` | **16** | Above default (10) for cleaner reconstruction. |

## Prompt System — Eight Pillars

The pipeline enhances user prompts via `_enhance_heartmula_prompt()` following the **Eight Pillars** tag hierarchy from HeartMuLa research. The key principle is **"Less is More"** — excessive tags cause probability interference.

See `docs/prompting_guides/heartmula_instructions.md` for the full prompting guide.

**Tag structure (7-9 tags total):**
1. Core anchors: Genre + Timbre + Mood + Instrument (4 tags, pillar-ordered)
2. Temporal descriptor scaled to duration (1 tag)
3. User tags (appended verbatim)
4. Negative floor: `no drums, instrumental` (2 tags)

**Structural lyrics** use `[interlude]` markers (not `[verse]`/`[bridge]`) to suppress vocal bias.

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

Segment 0: [intro] + [interlude]     <- opening arrival
Segment 1: [interlude] + [interlude] <- sustaining
Segment 2: [interlude] + [interlude] <- sustaining
Segment 3: [interlude] + [interlude] <- sustaining
Segment 4: [interlude] + [outro]     <- closing dissolution

Each segment: generate -> postprocess -> collect
Join with 8s equal-power cosine-squared crossfades

Total output: 5x240s - 4x8s = 1168s ~ 19m28s
```

**Key anchoring** is the primary cross-segment coherence mechanism. Since the heartlib API does not support latent context feedback (`ref_audio` raises `NotImplementedError`, MuQ `continuous_segments` hardcoded to zeros), a musical key tag constrains the LM to generate in a consistent key across all segments.

### Story Mode

One HeartMuLa call per stage. Each stage can have different tags for tonal evolution (e.g., centering -> depth -> integration). Stages exceeding 240s are automatically delegated to the long-form pipeline with key anchoring.

## Post-Processing Chain

The `make_heartmula_music_chain()` in `core/audio_processor.py` applies:

| Step | Effect | Purpose |
|------|--------|---------|
| 1 | NoiseGate (-55 dB, 2:1) | Suppress codec quantization noise; threshold lowered for higher-CFG output |
| 2 | HighpassFilter (60 Hz) | Remove sub-60 Hz codec rumble |
| 3 | LowShelfFilter (100 Hz, +1.5 dB) | Grounding warmth for drone-heavy content |
| 4 | PeakFilter (220 Hz, -1.0 dB, Q=0.7) | Clarity cut; removes low-mid mud from HeartCodec |
| 5 | PeakFilter (4000 Hz, -2.0 dB, Q=0.6) | Tame upper-mid brightness; wide Q for transparency |
| 6 | HighShelfFilter (9500 Hz, -2.0 dB) | Warm HF rolloff for headphone listening |
| 7 | Compressor (-20 dB, 2:1, 100ms/900ms) | Slow meditative dynamics; prevents pumping |
| 8 | Limiter (-0.5 dBFS) | Safety limiter |

Pre-mix LUFS target: **-17.0 LUFS** (HeartMuLa output levels are consistent).

## Memory Management

```python
# After each generation phase:
gc.collect()

# MPS backend:
torch.mps.empty_cache()

# MLX backend:
mx.set_cache_limit(0)
mx.clear_cache()
```

The `lazy_load=True` parameter ensures that the LM and codec are never loaded simultaneously, keeping peak memory within ~8-10 GB per phase.

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
| Out of memory crash | Confirm `lazy_load=True`. Close other GPU-intensive apps |
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

The pipeline enhances user prompts via `_enhance_heartmula_prompt()` which builds concise Eight Pillars tags and `[interlude]`-based structural lyrics.
