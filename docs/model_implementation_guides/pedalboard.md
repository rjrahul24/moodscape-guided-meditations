<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/audio_processor.py` · `core/mixer.py` · `core/kokoro_tts/postprocessor.py` · `core/f5_tts/postprocessor.py`
**Key functions:** `make__music_chain()` · `make_acestep_music_chain()` · `make_lyria_music_chain()` · `make_vocal_pocket_chain()` · `make_master_chain()` · `build_voice_chain()` · `apply_fx()`
**Mix defaults:** `music_volume_db=−16.0` · `duck_amount_db=−16.0` · `target_lufs=−16.0` · export streamed in 20s chunks
**Active ducking:** `mixer.mix()` calls `apply_breathing_duck()` — a script/VAD-aware sidechain duck (predictive S-curve descent, deep hold during speech, gradual release, pause lift). Applied fullband.
**IR files:** `assets/impulse_responses/{warm_studio,wooden_hall,stone_chapel}.wav` · default: `warm_studio`
**Tasks:**
- See full plugin-by-plugin parameter tables → `docs/ARCHITECTURE.md#fx-chains--full-parameter-tables`
- Change ducking params → `mixer.py :: compute_breathing_gain_db()` (pre_descent/attack_ramp/release/lift/VAD threshold)
- Change LUFS target → `pipeline.py` (`target_lufs`) + `mixer.py :: export_audio()`
- Change master limiter → `audio_processor.py :: make_master_chain()`
**See also:** `docs/ARCHITECTURE.md#fx-chains--full-parameter-tables` · `docs/optimization_and_processing/audio_processing.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# Pedalboard Implementation Guide for MoodScape
### Complete reference for MoodScape — audio processing, ducking, mixing, and normalization

---

## Table of Contents

1. [What Is Pedalboard](#1-what-is-pedalboard)
2. [MoodScape Audio Stack Overview](#2-moodscape-audio-stack-overview)
3. [Sample Rate Alignment (Standardized 44.1kHz)](#3-sample-rate-alignment)
4. [Voice FX Chain — Full Implementation](#4-voice-fx-chain)
5. [Music FX Chain — Spectral Masking & Ambient Preservation](#5-music-fx-chain)
6. [Master Chain — Full Implementation](#6-master-chain)
7. [Auto-Ducking — Mask-Based Voice Activity Ducking](#7-auto-ducking)
8. [Track Overlay & Alignment](#8-track-overlay--alignment)
9. [Chunk Streaming, LUFS & Export](#9-chunk-streaming-lufs--export)
10. [Performance & Memory Best Practices](#10-performance--memory-best-practices)

---

## 1. What Is Pedalboard

Pedalboard is Spotify's open-source Python library for studio-quality audio processing. It releases the Python GIL and processes in C++ via JUCE — the same framework inside Ableton Live and Logic Pro. For MoodScape, this provides near-instant FX rendering.

---

## 2. MoodScape Audio Stack Overview

The full signal flow through Pedalboard in MoodScape:

```
┌─────────────────────────────────────────────────────────────┐
│  Kokoro TTS Output                 Output          │
│  float32, mono, 24000 Hz          float32, 44100 Hz         │
└──────────┬──────────────────────────────────────────────────┘
           │                              │
           │  ┌───────────────────────┐    │
           │  │  ARTIFACT TRIMMER     │    │
           │  │  (per-chunk)          │    │
           │  │  1. Hard-clip guard   │    │
           │  │  2. Trailing silence  │    │
           │  │  3. Spectral flatness │    │
           │  └──────────┬────────────┘    │
           │             │                │
  ┌────────▼─────────┐         ┌──────────▼──────────┐
  │  VOICE FX CHAIN  │         │  MUSIC FX CHAIN     │
  │  NoiseGate       │         │  PeakFilter (EQ)    │
  │  HighpassFilter  │         │  HighShelfFilter    │
  │  Compressor      │         │  Limiter            │
  │  Reverb          │         └──────────┬──────────┘
  │  Limiter         │                    │
  └────────┬─────────┘                    │
           │                              │
           │  ┌──────────────────────────┐ │
           │  │  MASK-BASED DUCKING     │ │
           │  │  1. voice_activity mask │ │
           │  │  2. 100ms lookahead     │ │
           │  │  3. Attack/Release EMA  │ │
           │  │  4. dB → linear gain    │ │
           │  └──────────────────────────┘ │
           │                              │
           └──────────────┬───────────────┘
                          │
                 ┌────────▼──────────┐
                 │  OVERLAY + FADES  │
                 │  Pre-roll music   │
                 │  Fade in/out      │
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │  EXPORT STREAMER  │
                 │  (20 sec chunks)  │
                 │  * LUFS Gain      │
                 │  * 35Hz HPF       │
                 │  * Bus Compressor │
                 │  * Master Limiter │
                 │  * AudioFile Write│
                 └───────────────────┘
```

---

## 3. Sample Rate Alignment

**Pre-Mix Standardization:**
Audio from multiple models (Kokoro at 24kHzat 44.1kHz) must be standardized to a common studio rate to prevent pitch-shifting "chipmunk" artifacts. MoodScape standardizes aggressively on **44.1kHz** immediately after generation using `torchaudio.functional.resample`.

```python
def resample_to_44100(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == 44100: return audio
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    res_tensor = torchaudio.functional.resample(tensor, orig_sr, 44100)
    return res_tensor.squeeze(0).numpy().astype(np.float32)
```

---

## 4. Voice FX Chain

The strict ordering of Voice FX is critical to prepare raw TTS properly.
1. `NoiseGate`: Cleans underlying digital "hiss" before dynamic boosts.
2. `HighpassFilter (80Hz)`: Eliminates sub-bass rumble often generated by model phoneme artifacts.
3. `Compressor`: Steadies loud variances to create a confident, whispering "meditation coach" tone.
4. `Reverb & Limiter`: Immersive room styling and final peak clipping protection.

```python
def make_voice_chain(reverb_amount: float = 0.09) -> Pedalboard:
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        NoiseGate(threshold_db=-35, ratio=20.0, attack_ms=1.0, release_ms=50),
        HighpassFilter(cutoff_frequency_hz=80.0),
        Compressor(threshold_db=-19.0, ratio=3.5, attack_ms=10.0, release_ms=200.0),
        Reverb(room_size=0.17, damping=0.7, wet_level=reverb_amount, dry_level=1.0 - reverb_amount),
        Limiter(threshold_db=-1.0),
    ])
```

---

## 5. Music FX Chain

Background music receives spectral masking to preserve vocal clarity while maintaining natural ambient timbre.

- `PeakFilter (300Hz, +2dB)`: Adds low-end warmth to ambient pads.
- `PeakFilter (1800Hz, -3dB)`: Carves a vocal pocket notch — deeper and slightly higher than before for better speech clarity.
- `HighShelfFilter (10kHz, -4dB)`: Gently softens 's high-frequency content above 10kHz while preserving all ambient harmonic content between 3–10kHz.

> **Note:** The previous implementation used a `LowpassFilter(3000Hz)` which was a brick-wall cut that destroyed all harmonic content above 3kHz. The `HighShelfFilter` replaces this with a gentle slope that preserves the natural timbre of ambient pads.

```python
def make__music_chain() -> Pedalboard:
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.0, q=0.7),      # Low-end warmth
        PeakFilter(cutoff_frequency_hz=1800, gain_db=-3.0, q=0.6),   # Vocal pocket notch
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=-4.0),   # Gentle HF rolloff
        Limiter(threshold_db=-1.0),
    ])
```

- **Equal-Power Crossfade Stitching (2 seconds)**: When music needs to be looped to cover the full meditation duration, MoodScape applies a 2-second equal-power cosine crossfade at loop boundaries. This maintains constant perceived loudness throughout the transition, unlike linear fades which dip by ~3dB at the midpoint.

---

## 6. Master Chain

The master chain applies subsonic filtering, gentle glue compression, and a touch of air. Peak control is deliberately **not** done here: pedalboard 0.9.23's `Limiter` inflates sub-threshold signals (~+4.75 dB) and adds broadband distortion, so true-peak limiting happens at export instead.

```python
def make_master_chain() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),   # Remove sub-bass rumble
        Compressor(threshold_db=-12.0, ratio=1.5, attack_ms=50.0, release_ms=200.0),  # glue (~1-2 dB GR)
        HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=1.0),  # air
    ])
```

Export target: **−16 LUFS** (Apple Music / Spotify standard). `export_audio()` LUFS-normalizes first, then applies `mixer.true_peak_limit()` to a −1.0 dBTP ceiling (oversampled brickwall, clamped release), then streams to file in 20 s chunks.

---

## 7. Auto-Ducking

MoodScape's ducking is the **breathing sidechain duck** (`apply_breathing_duck` in `mixer.py`), built from two combined gain curves:

1. **Script curve** (`_script_gain_db`): phrases are detected from the voice via RMS-envelope VAD (`detect_phrases`, −40 dB threshold, merge gaps <250 ms, drop phrases <150 ms). A predictive cubic-S-curve descent starts ~600 ms *before* each phrase, holds at `duck_amount_db` during speech, releases over ~1.5 s, and lifts +1.5 dB during pauses ≥1.5 s so the bed "breathes". Zero-phase smoothed (~6 Hz).
2. **Reactive curve** (`_reactive_gain_db`): a vectorized envelope-follower safety net (200 Hz–4 kHz speech-band detector) that catches off-script breaths.

The curves are combined by `combine_script_with_reactive()` — where the script lifts, the script wins; elsewhere the deeper of the two applies. The result is applied **fullband** so the whole bed drops under speech.

**Key parameter values:**
| Parameter | Value | Rationale |
|---|---|---|
| `duck_amount_db` | −16 dB | Pipeline default; combined with `music_volume_db=−16` → ~28 LU voice-music separation during speech |
| `pre_descent_ms` | 600 ms | Bed starts falling before the first syllable |
| `attack_ramp_ms` | 700 ms | S-curve descent duration |
| `release_ms` | 1500 ms | Gradual recovery matches meditation pacing |
| `lift_db` / `lift_pause_s` | +1.5 dB / 1.5 s | Bed rises slightly during long pauses |
| `vad_threshold_db` | −40 dB | Phrase-detection threshold on the voice RMS envelope |

---

## 8. Track Overlay & Alignment

The music naturally pre-rolls for 8 seconds (and post-rolls for 15 seconds) to establish the environment before the narrator begins. Fade-in and fade-out functionality creates smooth 0→1 ramps natively in NumPy to taper the combined experience gracefully.

---

## 9. Chunk Streaming, LUFS & Export

To prevent massive RAM spikes (up to 14GB for a 12-hour track) caused by processing 30+ minute long `float32` arrays simultaneously through multiple pedals:

1. **Unified LUFS target of −14 LUFS** (streaming distribution standard — Spotify, Apple Music, YouTube). The gain scalar is pre-calculated via `pyloudnorm.Meter`. This ensures the meditation output matches the loudness of surrounding content on streaming platforms.
2. **AudioFile Streaming**: The `export_audio` function processes the combined waveform iteratively in 20-second chunk block boundaries. It applies the target LUFS linear gain, executes the `make_master_chain` FX (`HighpassFilter(35Hz)` → `Gain(-3.0)` → `Compressor(-18dB, 2:1, 30ms/300ms)` → `Limiter(-0.1)`) directly on the tiny chunk, and safely dumps to the disk using Pedalboard's `AudioFile` protocol. This streams infinite duration meditations indefinitely on low-memory Apple Silicon pipelines.

---

## 10. Performance & Memory Best Practices
- **Never rely on pure single-array manipulation at output stages.** For high-length generations, `np.float32` expands dramatically in volatile RAM. Always map Pedalboard exports using the Context Manager Audio Streaming paradigm. 
- Ensure to rigorously check sample rates matching. torchaudio `functional.resample` handles cross-platform 44.1kHz standards cleanly across PyTorch tensors.