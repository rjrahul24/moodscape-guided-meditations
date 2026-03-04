# Pedalboard Implementation Guide for MoodScape
### Complete reference for Claude Code — audio processing, ducking, mixing, and normalization

---

## Table of Contents

1. [What Is Pedalboard](#1-what-is-pedalboard)
2. [Core Architecture & How It Works](#2-core-architecture--how-it-works)
3. [Critical Constraints & Gotchas](#3-critical-constraints--gotchas)
4. [MoodScape Audio Stack Overview](#4-moodscape-audio-stack-overview)
5. [Sample Rate Alignment (Kokoro + MusicGen)](#5-sample-rate-alignment-kokoro--musicgen)
6. [Audio Array Formats & Conventions](#6-audio-array-formats--conventions)
7. [Voice FX Chain — Full Implementation](#7-voice-fx-chain--full-implementation)
8. [Music FX Chain — Full Implementation](#8-music-fx-chain--full-implementation)
9. [Master Chain — Full Implementation](#9-master-chain--full-implementation)
10. [Auto-Ducking — The Core Algorithm](#10-auto-ducking--the-core-algorithm)
11. [Track Overlay & Alignment](#11-track-overlay--alignment)
12. [Fades & Transitions](#12-fades--transitions)
13. [LUFS Normalization](#13-lufs-normalization)
14. [WAV & MP3 Export](#14-wav--mp3-export)
15. [Full `audio_processor.py` Reference Implementation](#15-full-audio_processorpy-reference-implementation)
16. [Full `mixer.py` Reference Implementation](#16-full-mixerpy-reference-implementation)
17. [Performance & Memory Best Practices](#17-performance--memory-best-practices)
18. [Troubleshooting & Edge Cases](#18-troubleshooting--edge-cases)

---

## 1. What Is Pedalboard

Pedalboard is Spotify's open-source Python library for studio-quality audio processing. It was built by the Audio Intelligence Lab to power features like Spotify's AI DJ and AI Voice Translation — it is actively maintained, production-proven, and the right tool for MoodScape.

**Why Pedalboard over alternatives:**

| Library | Speed vs Pedalboard | Thread-Safe | Studio FX | Notes |
|---------|--------------------|-----------:|:---------:|-------|
| **Pedalboard** | 1× (baseline) | ✅ | ✅ | Wraps JUCE C++ engine |
| pydub | ~300× slower | ❌ | ❌ | No GIL release |
| librosa | ~4× slower for I/O | ❌ | ❌ | Analysis-focused |
| pySoX | ~300× slower | ❌ | ❌ | Shell-out dependency |
| ffmpeg (subprocess) | ~10× slower | ❌ | ❌ | Complex, no Python native |

Pedalboard releases the Python GIL and processes in C++ via JUCE — the same framework inside Ableton Live and Logic Pro. For MoodScape, this means near-instant FX rendering even on consumer hardware.

**Current version:** v0.9.22 (supports Python 3.10–3.14, macOS/Linux/Windows, ARM64)

---

## 2. Core Architecture & How It Works

### The Pedalboard Object

A `Pedalboard` is simply an ordered list of `Plugin` objects. Audio flows through them left to right, just like guitar pedals:

```python
from pedalboard import Pedalboard, Compressor, Reverb, Limiter

board = Pedalboard([
    Compressor(threshold_db=-20, ratio=3.0),
    Reverb(room_size=0.3),
    Limiter(threshold_db=-1.0),
])

# Process: input_audio must be float32, shape (channels, samples)
output_audio = board(input_audio, sample_rate)
```

### Input/Output Contract

**This is the most important rule:** Pedalboard always expects `(channels, samples)` shape — NOT `(samples,)` or `(samples, channels)`.

```python
# ✅ Correct: (1, N) for mono
audio_2d = audio.reshape(1, -1)            # shape: (1, 48000)
processed = board(audio_2d, 24000)         # shape: (1, 48012) — may tail-extend
result = processed.squeeze(0)              # shape: (48012,)

# ❌ Wrong: will raise or produce garbage
board(audio, 24000)                        # audio.shape = (48000,) — BAD
```

### The `reset` Parameter

When processing audio in **chunks**, pass `reset=False` to preserve reverb/delay tails between calls:

```python
# Streaming chunk-by-chunk
for chunk in audio_chunks:
    processed_chunk = board(chunk, sample_rate, reset=False)

# Full-file processing (default reset=True is fine)
processed = board(full_audio, sample_rate)
```

### Available Built-in Effects

| Category | Plugins |
|----------|---------|
| **Dynamics** | `Compressor`, `Gain`, `Limiter`, `NoiseGate` |
| **EQ / Filters** | `LowShelfFilter`, `HighShelfFilter`, `LowpassFilter`, `HighpassFilter`, `PeakFilter`, `LadderFilter` |
| **Spatial** | `Reverb`, `Delay`, `Convolution` |
| **Pitch/Time** | `PitchShift`, `Resample` |
| **Color/Texture** | `Chorus`, `Phaser`, `Distortion`, `Bitcrush` |
| **Codec Simulation** | `MP3Compressor`, `GSMFullRateCompressor` |

For MoodScape, the relevant ones are: `LowShelfFilter`, `HighShelfFilter`, `Compressor`, `Reverb`, `Limiter`, `Gain`.

---

## 3. Critical Constraints & Gotchas

### ⚠️ No Built-in Time Automation

Pedalboard does **not** support automating parameters over time in a single call. You cannot ramp a `Gain` from -8 dB to 0 dB through the plugin itself. This is the key limitation.

**For MoodScape's ducking:** This means you build your own gain curve in NumPy and multiply the audio array directly — you do NOT route the voice signal through a sidechain compressor plugin. See Section 10 for the full algorithm.

### ⚠️ Output May Be Longer Than Input

Reverb has a tail. After your audio ends, reverb keeps ringing out. Pedalboard may return an array that is slightly longer than the input. Always trim or handle this:

```python
processed = board(audio_2d, sample_rate)
processed = processed[:, :audio_2d.shape[1]]  # Trim to original length
```

### ⚠️ dtype Must Be float32

Always convert to `float32` before processing. Pedalboard will error or produce wrong results with `float64` or integer arrays:

```python
audio = audio.astype(np.float32)
```

### ⚠️ Values Must Be in [-1.0, 1.0]

Pedalboard assumes a normalized float range. AI-generated audio from MusicGen and Kokoro may occasionally produce values outside this range. Always clip before processing:

```python
audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
```

---

## 4. MoodScape Audio Stack Overview

The full signal flow through Pedalboard in MoodScape:

```
┌─────────────────────────────────────────────────────────────┐
│  Kokoro TTS Output                MusicGen Output           │
│  float32, mono, 24000 Hz          float32, (possibly stereo)│
│                                   32000 Hz                  │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           │                    ┌─────────▼──────────┐
           │                    │  RESAMPLE to 24000  │
           │                    │  torchaudio          │
           │                    │  Stereo→Mono average │
           │                    └─────────┬────────────┘
           │                              │
  ┌────────▼─────────┐         ┌──────────▼──────────┐
  │  VOICE FX CHAIN  │         │  MUSIC FX CHAIN     │
  │  LowShelfFilter  │         │  LowShelfFilter     │
  │  Compressor      │         │  HighShelfFilter    │
  │  Reverb          │         │  Limiter            │
  │  Limiter         │         └──────────┬──────────┘
  └────────┬─────────┘                    │
           │                              │
           │  ┌──────────────────────────┐│
           │  │  NUMPY DUCKING ENGINE    ││
           │  │  1. Detect voice activity││
           │  │  2. Butterworth smooth   ││
           │  │  3. Build gain curve     ││
           │  │  4. music *= gain_curve  ││
           │  └──────────────────────────┘│
           │                              │
           └──────────────┬───────────────┘
                          │
                 ┌────────▼──────────┐
                 │  OVERLAY + FADES  │
                 │  Pre-roll music   │
                 │  Fade in/out      │
                 │  Sum arrays       │
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │  MASTER CHAIN     │
                 │  Limiter -0.5 dB  │
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │  LUFS NORMALIZE   │
                 │  pyloudnorm -16   │
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │  EXPORT           │
                 │  WAV / MP3        │
                 └───────────────────┘
```

---

## 5. Sample Rate Alignment (Kokoro + MusicGen)

**This must happen before any Pedalboard processing.** Mixing audio at different sample rates produces pitch-shifted or time-warped results.

| Model | Native Sample Rate |
|-------|--------------------|
| Kokoro TTS | **24,000 Hz** |
| MusicGen (all variants) | **32,000 Hz** |
| Target for MoodScape | **24,000 Hz** (Kokoro's rate) |

**Resample MusicGen output to 24,000 Hz:**

```python
import torch
import torchaudio
import numpy as np

def resample_music(audio_np: np.ndarray, orig_sr: int = 32000, target_sr: int = 24000) -> np.ndarray:
    """Resample music from 32kHz to 24kHz. Handles mono or stereo input."""
    audio_t = torch.from_numpy(audio_np.astype(np.float32))
    
    # Ensure shape is (channels, samples) for torchaudio
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)
    
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(audio_t)
    
    # Convert stereo → mono by averaging channels
    if resampled.shape[0] > 1:
        resampled = resampled.mean(dim=0, keepdim=True)
    
    return resampled.squeeze(0).numpy().astype(np.float32)
```

**Why NOT use `AudioFile(...).resampled_to(44100)`:**  
Some research docs suggest resampling to 44100 Hz "studio standard." For MoodScape, this is unnecessary and wastes memory and compute. 24,000 Hz is broadcast-quality for voice content and perfectly fine for a meditation app. Keep everything at 24,000 Hz throughout.

---

## 6. Audio Array Formats & Conventions

MoodScape works entirely in **mono float32 at 24,000 Hz**.

| Convention | Value |
|------------|-------|
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |
| dtype | `np.float32` |
| Value range | [-1.0, 1.0] |
| Array shape (working) | `(N,)` — 1D |
| Array shape (Pedalboard input) | `(1, N)` — reshape before calling |
| Array shape (Pedalboard output) | `(1, M)` — squeeze after calling |

**Standard helper wrapper (use this pattern everywhere):**

```python
def apply_fx(audio: np.ndarray, chain: Pedalboard, sample_rate: int = 24000) -> np.ndarray:
    """Apply FX chain to a 1D mono float32 array. Returns 1D float32."""
    audio = audio.astype(np.float32)
    audio_2d = audio.reshape(1, -1)              # → (1, N)
    processed = chain(audio_2d, sample_rate)      # → (1, M)
    result = processed.squeeze(0)                  # → (M,)
    # Trim to original length (reverb tail may extend output)
    result = result[:len(audio)]
    return result.astype(np.float32)
```

---

## 7. Voice FX Chain — Full Implementation

The voice chain enhances Kokoro's output to sound warm, present, and immersive. The goal is "voice in a peaceful meditation room" — not "voice in a recording booth."

### Effect-by-Effect Rationale

**1. `LowShelfFilter` (+2 dB at 300 Hz) — Warmth EQ**
- Boosts the low-mids to counteract the thin, slightly nasal quality that AI TTS voices sometimes have at default settings.
- 300 Hz is the "body" of a human voice — boosting here adds warmth without adding muddiness.
- Keep gain between +1.5 and +3.0 dB. Above +3 dB starts to sound boomy.

**2. `Compressor` (threshold -20 dB, ratio 3:1) — Consistency**
- AI TTS varies in loudness across phonemes. Compression levels this out so quiet words are as audible as loud ones.
- Attack 10 ms: Fast enough to control peaks but not so fast that it squashes transients.
- Release 100 ms: Allows natural dynamic breathing between words.
- Ratio 3:1 is gentle — this is not a limiter, it's a subtle leveler.

**3. `Reverb` (room_size 0.3, damping 0.7) — Space**
- Gives the voice a sense of physical space — as if recorded in a warm, mid-sized room.
- `room_size=0.3`: A small room. Larger values (>0.5) produce a cathedral effect that blurs words.
- `damping=0.7`: High damping absorbs high-frequency reverb tails, preventing harshness.
- `wet_level`: Controlled by the user's "Voice Reverb" slider (default 0.15 = 15% wet).
- `dry_level`: Should be `1.0 - wet_level` to maintain consistent total output level.

**4. `Limiter` (threshold -1.0 dB) — Protection**
- Prevents the voice from clipping after EQ boost and before mixing.
- Always the last plugin in the voice chain.

### Code

```python
from pedalboard import Pedalboard, LowShelfFilter, Compressor, Reverb, Limiter

def make_voice_chain(reverb_amount: float = 0.15) -> Pedalboard:
    """
    Voice FX chain: warmth EQ → compression → reverb → limiting.
    
    Args:
        reverb_amount: Wet level for reverb (0.0 = dry, 0.5 = very wet).
                       Default 0.15 is subtle and appropriate for meditation.
    """
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        LowShelfFilter(
            cutoff_frequency_hz=300,
            gain_db=2.0,           # Warms voice without muddiness
        ),
        Compressor(
            threshold_db=-20,
            ratio=3.0,
            attack_ms=10,
            release_ms=100,
        ),
        Reverb(
            room_size=0.3,
            damping=0.7,
            wet_level=reverb_amount,
            dry_level=1.0 - reverb_amount,  # Keep total level consistent
            freeze_mode=0.0,               # Ensure reverb decays normally
        ),
        Limiter(threshold_db=-1.0),
    ])
```

### Parameter Tuning Guide

| Parameter | Min | Default | Max | Effect |
|-----------|-----|---------|-----|--------|
| `reverb_amount` (wet_level) | 0.0 | 0.15 | 0.5 | Space around voice |
| LowShelf gain_db | 0.0 | 2.0 | 4.0 | Voice warmth/body |
| Compressor ratio | 1.5 | 3.0 | 6.0 | Level consistency |
| Reverb room_size | 0.1 | 0.3 | 0.6 | Room size |
| Reverb damping | 0.3 | 0.7 | 1.0 | Brightness of reverb tail |

---

## 8. Music FX Chain — Full Implementation

The music chain shapes MusicGen's output to sit comfortably *beneath* the voice. The goal is "warm ambient bed" — present but not distracting, with softened highs that don't compete with vocal frequencies.

### Effect-by-Effect Rationale

**1. `LowShelfFilter` (+1.5 dB at 200 Hz) — Low-End Warmth**
- Adds warmth and body to ambient music, especially for piano and pad textures.
- Lower cutoff than voice EQ (200 Hz vs 300 Hz) because we want to thicken the low end, not the mids.
- Keep gain modest (+1 to +2 dB). Excessive low-end boost will muddy the mix when voice is added.

**2. `HighShelfFilter` (-3.0 dB at 8000 Hz) — High-End Softening**
- The single most important music FX for meditation. AI music often has "digital shimmer" in the highs (MusicGen is particularly prone to this) that becomes distracting when sustained under a human voice.
- Cutting 3 dB at 8 kHz and above makes the music sound warmer, more analog, and less fatiguing to listen to over 10+ minutes.
- This also creates "spectral space" — the highs vacated by the music are where the voice's consonants and air live, so the two sources feel separated rather than cluttered.

**3. `Limiter` (threshold -1.0 dB) — Protection**
- Catches any peaks from MusicGen that exceed safe levels before ducking math is applied.

### Code

```python
from pedalboard import Pedalboard, LowShelfFilter, HighShelfFilter, Limiter

def make_music_chain() -> Pedalboard:
    """
    Music FX chain: low-end warmth → high-end softening → limiting.
    
    The HighShelfFilter is critical — it tames MusicGen's 'digital shimmer'
    in the 8kHz+ range and creates spectral space for the voice.
    """
    return Pedalboard([
        LowShelfFilter(
            cutoff_frequency_hz=200,
            gain_db=1.5,          # Gentle low-end warmth
        ),
        HighShelfFilter(
            cutoff_frequency_hz=8000,
            gain_db=-3.0,         # Soft the highs — makes it less tiring to listen to
        ),
        Limiter(threshold_db=-1.0),
    ])
```

### Optional: Enhance the Music Chain

For a more sophisticated music sound, add a subtle `LowpassFilter` to further tame high frequencies during voice sections. However, since MoodScape handles this dynamically via ducking, the static chain above is sufficient and simpler.

---

## 9. Master Chain — Full Implementation

The master chain is applied after voice and music are mixed together. It is the final protection before export.

```python
from pedalboard import Pedalboard, Limiter

def make_master_chain() -> Pedalboard:
    """
    Final master chain: prevent inter-sample clipping before export.
    
    Note: Do NOT add heavy compression here — that would squash the
    natural dynamic range that makes meditation audio feel spacious.
    A single limiter is all that's needed.
    """
    return Pedalboard([
        Limiter(threshold_db=-0.5),  # -0.5 dB gives a hair of headroom
    ])
```

**Why -0.5 dB and not -1.0 dB?** The LUFS normalization step that follows may boost the overall level slightly. Keeping a tight ceiling at -0.5 dB true peak ensures that after normalization, we don't exceed 0 dBFS. The voice chain and music chain already have their own limiters, so the master limiter is mainly catching sum artifacts (two signals added together can briefly exceed what either alone would produce).

---

## 10. Auto-Ducking — The Core Algorithm

### What Is Ducking and Why It Matters

Ducking is the process of automatically reducing the background music volume when the narrator is speaking, then bringing it back up during pauses. Without ducking, voice and music compete for the same perceptual space — the listener has to work to hear the narrator.

For meditation specifically, proper ducking is essential because:
- Meditation has many deliberate pauses (5–10 seconds). The music should swell gracefully during these.
- The transitions must be smooth and natural — not abrupt.
- The music should still be audible during speech (this is NOT muting — it's a gentle -8 dB reduction).

### The Algorithm (Step by Step)

Pedalboard has no built-in sidechain compressor, so ducking is implemented in NumPy:

**Step 1: Detect voice activity from the voice_activity mask**

The `TTSEngine.synthesize()` method already produces a boolean `voice_activity` array (True where voice is speaking, False during pauses). Use this directly — it is more accurate than computing an amplitude envelope from the audio signal, because silence injection in TTS creates perfect zeros.

```python
# voice_activity is already a bool array from TTSEngine
# shape: (N,) — same length as voice_audio
```

**Step 2: Convert to float and smooth with Butterworth low-pass filter**

The raw boolean mask has instant on/off transitions. A sudden -8 dB drop in music creates a jarring "click." We smooth it with a 2nd-order Butterworth lowpass at ~3 Hz cutoff. This creates natural attack (music ducks down over ~300 ms) and release (music swells back over ~800 ms) behavior — both controlled by the filter's impulse response.

```python
from scipy.signal import butter, filtfilt

def smooth_activity_mask(
    voice_activity: np.ndarray,
    sample_rate: int = 24000,
    cutoff_hz: float = 3.0,
) -> np.ndarray:
    """
    Smooth the boolean voice activity mask with a Butterworth lowpass.
    
    The 3 Hz cutoff translates to roughly:
    - Attack (duck down): ~300 ms
    - Release (swell back): ~800 ms
    These timings feel natural for meditation — not too fast, not too slow.
    """
    envelope = voice_activity.astype(np.float32)
    
    nyquist = sample_rate / 2.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype='low')
    smoothed = filtfilt(b, a, envelope).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    
    return smoothed
```

**Step 3: Convert smoothed envelope to linear gain**

Map the 0→1 smoothed envelope to a gain curve in dB, then convert to linear:

```python
def make_duck_gain_curve(
    smoothed_envelope: np.ndarray,
    duck_amount_db: float = -8.0,
) -> np.ndarray:
    """
    Convert smoothed envelope [0, 1] to a linear gain curve.
    
    Where envelope = 0 (silence): gain = 0 dB → linear 1.0 (no change)
    Where envelope = 1 (speech):  gain = duck_amount_db → linear < 1.0
    
    Args:
        duck_amount_db: Negative dB, e.g. -8.0.
                        -4 dB: subtle dip (music still prominent)
                        -8 dB: standard ducking (music clearly recedes)
                        -12 dB: aggressive ducking (music nearly inaudible)
                        -15 dB: very aggressive (not recommended for meditation)
    """
    gain_db = smoothed_envelope * duck_amount_db     # Scalar map: 0→0 dB, 1→duck_db
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    return gain_linear
```

**Step 4: Apply gain curve to music**

```python
# Element-wise multiply: each sample of music is scaled by its gain value
music_ducked = music_audio * gain_curve
```

### Complete Ducking Function

```python
import numpy as np
from scipy.signal import butter, filtfilt

def compute_duck_curve(
    voice_activity: np.ndarray,
    sample_rate: int = 24000,
    duck_amount_db: float = -8.0,
    cutoff_hz: float = 3.0,
) -> np.ndarray:
    """
    Build a smooth linear gain curve that ducks music when voice is active.
    
    Args:
        voice_activity: Boolean array — True where narrator is speaking.
                        Must be same length as the music array it will be applied to.
        sample_rate:    24000 for MoodScape.
        duck_amount_db: How much to reduce music during speech. Default -8 dB.
                        Range: -4 (subtle) to -15 (aggressive).
        cutoff_hz:      Smoothing filter cutoff. Default 3 Hz.
                        Lower = slower transitions (more gradual).
                        Higher = faster transitions (more abrupt).
    
    Returns:
        Linear gain curve, shape (N,), dtype float32.
        Values are in range [10^(duck_amount_db/20), 1.0].
    """
    envelope = voice_activity.astype(np.float32)
    
    # Smooth transitions with Butterworth lowpass
    nyquist = sample_rate / 2.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype='low')
    smoothed = filtfilt(b, a, envelope).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    
    # Map to dB gain, then to linear
    gain_db = smoothed * duck_amount_db
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    
    return gain_linear
```

### Ducking Parameter Reference

| duck_amount_db | What It Sounds Like | Use Case |
|----------------|---------------------|----------|
| -4 dB | Music barely dips | Suitable when music prompt is naturally quiet |
| -8 dB | **Default — recommended** | Music noticeably recedes, voice is clear |
| -10 dB | Pronounced ducking | Louder music prompts (drums, busy textures) |
| -12 dB | Aggressive | Music nearly inaudible during speech |
| -15 dB | Very aggressive | Not recommended for meditation (jarring) |

### Why NOT Use Amplitude Envelope Detection

Alternative approaches (RMS envelope from voice audio, absolute value smoothing) introduce complexity and can fail when:
- TTS audio has DC offset
- Normalization changes the reference level
- The voice has breaths or noise at silence boundaries

Using the `voice_activity` boolean mask from `TTSEngine` is more reliable because:
- It reflects the actual structural intent (speech vs pause)
- It's sample-accurate (generated from the same TTS timing)
- It has no ambiguity around threshold selection

---

## 11. Track Overlay & Alignment

### Pre-Roll Strategy

The music should begin playing **2 seconds before the voice starts**. This gives the listener time to recognize the ambient setting before the narrator interrupts with instructions. Without pre-roll, the voice and music feel like they start simultaneously, which is jarring.

```python
def overlay_tracks(
    voice: np.ndarray,
    music: np.ndarray,
    music_pre_roll_sec: float = 2.0,
    sample_rate: int = 24000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align voice and music for mixing. Music starts first.
    
    Returns two arrays of equal length, ready to sum.
    The returned voice array has silence prepended for the pre-roll period.
    """
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    
    # Prepend silence to voice
    aligned_voice = np.concatenate([
        np.zeros(pre_roll_samples, dtype=np.float32),
        voice,
    ])
    
    total_length = len(aligned_voice)
    
    # Loop music if shorter (this handles short MusicGen outputs)
    if len(music) < total_length:
        repeats_needed = (total_length // len(music)) + 2
        # Crossfade the loop points to avoid clicking
        music_looped = _loop_with_crossfade(music, total_length, sample_rate)
        aligned_music = music_looped[:total_length]
    else:
        aligned_music = music[:total_length]
    
    return aligned_voice, aligned_music


def _loop_with_crossfade(
    music: np.ndarray,
    target_length: int,
    sample_rate: int = 24000,
    crossfade_sec: float = 1.0,
) -> np.ndarray:
    """Loop music to target_length with crossfade at loop boundaries."""
    crossfade_samples = int(crossfade_sec * sample_rate)
    crossfade_samples = min(crossfade_samples, len(music) // 2)
    
    result = music.copy()
    
    while len(result) < target_length + crossfade_samples:
        fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        
        overlap = result[-crossfade_samples:] * fade_out + music[:crossfade_samples] * fade_in
        result = np.concatenate([result[:-crossfade_samples], overlap, music[crossfade_samples:]])
    
    return result
```

### Voice Activity Alignment

When the voice is prepended with pre-roll silence, the `voice_activity` mask must be extended to match:

```python
pre_roll_samples = int(music_pre_roll_sec * sample_rate)

aligned_activity = np.concatenate([
    np.zeros(pre_roll_samples, dtype=bool),    # False during pre-roll
    voice_activity,
])

# Ensure exact length match
if len(aligned_activity) < len(aligned_voice):
    pad = len(aligned_voice) - len(aligned_activity)
    aligned_activity = np.concatenate([aligned_activity, np.zeros(pad, dtype=bool)])

aligned_activity = aligned_activity[:len(aligned_voice)]
```

---

## 12. Fades & Transitions

All fades in MoodScape are handled with pure NumPy (no Pedalboard needed). Linear ramps are appropriate for meditation — smooth and predictable.

```python
def apply_fades(
    audio: np.ndarray,
    sample_rate: int = 24000,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
) -> np.ndarray:
    """
    Apply linear fade-in and fade-out to the full mixed audio.
    
    Fade-in: First fade_in_sec seconds ramp from 0 → 1
    Fade-out: Last fade_out_sec seconds ramp from 1 → 0
    
    The 3-second default fade-in gives the music room to establish
    before the narrator begins. The 5-second fade-out feels gradual
    and peaceful — longer than music production norms, but appropriate
    for meditation endings.
    """
    result = audio.copy()
    
    fade_in_samples = int(fade_in_sec * sample_rate)
    if 0 < fade_in_samples < len(result):
        ramp = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        result[:fade_in_samples] *= ramp
    
    fade_out_samples = int(fade_out_sec * sample_rate)
    if 0 < fade_out_samples < len(result):
        ramp = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        result[-fade_out_samples:] *= ramp
    
    return result
```

### Timing Recommendation

The fade-in is applied to the **full mix**, not just the music. This means:
- The music fades in from silence at the start.
- If the pre-roll is 2 seconds and the fade-in is 3 seconds, the music reaches full volume about 1 second into the narration.
- This is intentional and sounds natural — the music is still establishing as the narrator begins.

---

## 13. LUFS Normalization

LUFS (Loudness Units Full Scale) is the broadcast standard for audio loudness. Unlike peak normalization, LUFS normalization reflects how humans actually perceive loudness over time.

**Target: -16 LUFS for stereo, -19 LUFS for mono**

Since MoodScape outputs mono audio, the technically correct target is **-19 LUFS**. However, -16 LUFS is the widely understood podcast/streaming standard and what most meditation apps use. The 3 dB difference means the output will sound slightly louder on devices that apply their own loudness normalization. For simplicity and user familiarity, **-16 LUFS is recommended**.

```python
import pyloudnorm as pyln
import numpy as np

def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = 24000,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """
    Normalize audio to target LUFS using ITU-R BS.1770-4 algorithm.
    
    Args:
        audio:       1D float32 mono numpy array.
        sample_rate: Must match what the audio was generated at.
        target_lufs: Target loudness. -16.0 is standard for podcasts/apps.
    
    Returns:
        Normalized float32 array, clipped to [-1, 1].
    
    Edge cases handled:
        - Near-silent audio (loudness = -inf): returned unchanged.
        - Audio shorter than 400ms: pyloudnorm may fail; skip normalization.
    """
    # pyloudnorm requires at least 400ms of audio for BS.1770 gating blocks
    min_samples = int(0.4 * sample_rate)
    if len(audio) < min_samples:
        return audio
    
    meter = pyln.Meter(sample_rate)  # ITU-R BS.1770-4 meter
    
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return audio
    
    # Guard against silent audio (returns -inf)
    if not np.isfinite(loudness):
        return audio
    
    # Guard against extreme gain changes (>40 dB) that indicate a problem
    gain_change = target_lufs - loudness
    if abs(gain_change) > 40.0:
        return audio
    
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)
```

### LUFS Reference Values

| Target | Context |
|--------|---------|
| -14 LUFS | Spotify/Apple Music streaming |
| **-16 LUFS** | **Podcast standard (recommended for MoodScape)** |
| -19 LUFS | Mono podcast standard |
| -23 LUFS | Broadcast TV/radio |
| -24 LUFS | EBU R128 (European broadcast) |

---

## 14. WAV & MP3 Export

### WAV Export (lossless)

```python
import soundfile as sf

def export_wav(audio: np.ndarray, path: str, sample_rate: int = 24000) -> str:
    """Write mono float32 audio to WAV. Returns path."""
    # soundfile expects (samples,) for mono or (samples, channels) for multi
    sf.write(path, audio, sample_rate, subtype='PCM_24')  # 24-bit WAV
    return path
```

Use `PCM_24` (24-bit) rather than `PCM_16` (16-bit) for better dynamic range preservation. The file size difference is negligible for meditation lengths.

### MP3 Export

```python
from pedalboard.io import AudioFile

def export_mp3(audio: np.ndarray, path: str, sample_rate: int = 24000) -> str:
    """Write mono float32 audio to MP3 at V2 quality (~190 kbps VBR)."""
    # AudioFile expects (channels, samples) shape
    audio_2d = audio.reshape(1, -1).astype(np.float32)
    
    with AudioFile(path, 'w', samplerate=sample_rate, num_channels=1, quality=0.2) as f:
        f.write(audio_2d)
    
    return path
```

**Pedalboard's `AudioFile` quality parameter:**
- `quality=0.0`: Highest quality (~245 kbps VBR, largest file)
- `quality=0.2`: High quality (~190 kbps VBR) — **recommended for MoodScape**
- `quality=0.5`: Medium quality (~130 kbps VBR)
- `quality=1.0`: Lowest quality (~65 kbps VBR, smallest file)

For meditation audio with primarily voice and ambient music, V2 (~190 kbps) is transparent — listeners cannot distinguish it from lossless.

### Complete Export Function

```python
import tempfile

def export_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    output_format: str = 'wav',
) -> str:
    """
    Export audio to a temp file and return the path.
    
    Args:
        audio:         1D float32 mono numpy array.
        sample_rate:   24000 for MoodScape.
        output_format: 'wav' or 'mp3'.
    
    Returns:
        Absolute path to the temp file.
    """
    suffix = f'.{output_format}'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    
    if output_format == 'wav':
        sf.write(tmp_path, audio, sample_rate, subtype='PCM_24')
    elif output_format == 'mp3':
        from pedalboard.io import AudioFile
        with AudioFile(tmp_path, 'w', samplerate=sample_rate, num_channels=1, quality=0.2) as f:
            f.write(audio.reshape(1, -1).astype(np.float32))
    else:
        raise ValueError(f'Unsupported format: {output_format!r}. Use "wav" or "mp3".')
    
    return tmp_path
```

---

## 15. Full `audio_processor.py` Reference Implementation

```python
"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import numpy as np
from pedalboard import (
    Compressor,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)


def make_voice_chain(reverb_amount: float = 0.15) -> Pedalboard:
    """
    FX chain for Kokoro narration: warmth → compression → reverb → limit.
    
    Args:
        reverb_amount: Reverb wet level (0.0 = dry, 0.5 = very wet).
                       Exposed as a Gradio slider (default 0.15).
    """
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=300, gain_db=2.0),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        Reverb(
            room_size=0.3,
            damping=0.7,
            wet_level=reverb_amount,
            dry_level=1.0 - reverb_amount,
        ),
        Limiter(threshold_db=-1.0),
    ])


def make_music_chain() -> Pedalboard:
    """FX chain for MusicGen output: warm low end → tamed highs → limit."""
    return Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=1.5),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-3.0),
        Limiter(threshold_db=-1.0),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering limiter applied to the full mix before LUFS normalization."""
    return Pedalboard([
        Limiter(threshold_db=-0.5),
    ])


def apply_fx(
    audio: np.ndarray,
    chain: Pedalboard,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Apply a Pedalboard FX chain to a mono audio array.
    
    Handles all the shape manipulation internally.
    Input and output are both 1D float32 arrays.
    """
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    audio_2d = audio.reshape(1, -1)              # (1, N)
    processed = chain(audio_2d, sample_rate)      # (1, M)
    result = processed.squeeze(0)                  # (M,)
    result = result[:len(audio)]                   # Trim reverb tail
    return result.astype(np.float32)
```

---

## 16. Full `mixer.py` Reference Implementation

```python
"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, filtfilt


SAMPLE_RATE = 24000


# ─────────────────────────────────────────────────────────────────────────────
# Ducking
# ─────────────────────────────────────────────────────────────────────────────

def compute_duck_curve(
    voice_activity: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -8.0,
    cutoff_hz: float = 3.0,
) -> np.ndarray:
    """
    Create a smooth linear gain curve for music ducking.
    
    Uses Butterworth lowpass at ~3 Hz to create natural attack/release.
    At 3 Hz cutoff, the duck-down takes ~300 ms and the swell-back takes ~800 ms.
    
    Returns float32 array of linear gain values in range [10^(duck_db/20), 1.0].
    """
    envelope = voice_activity.astype(np.float32)
    
    nyquist = sample_rate / 2.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype='low')
    smoothed = filtfilt(b, a, envelope).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    
    gain_db = smoothed * duck_amount_db
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    
    return gain_linear


# ─────────────────────────────────────────────────────────────────────────────
# Overlay / alignment
# ─────────────────────────────────────────────────────────────────────────────

def _loop_with_crossfade(
    music: np.ndarray,
    target_length: int,
    sample_rate: int = SAMPLE_RATE,
    crossfade_sec: float = 1.0,
) -> np.ndarray:
    """Loop music array to target_length with crossfade at boundaries."""
    crossfade_samples = min(int(crossfade_sec * sample_rate), len(music) // 2)
    result = music.copy()
    
    while len(result) < target_length + crossfade_samples:
        fo = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fi = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        overlap = result[-crossfade_samples:] * fo + music[:crossfade_samples] * fi
        result = np.concatenate([result[:-crossfade_samples], overlap, music[crossfade_samples:]])
    
    return result


def overlay_tracks(
    voice: np.ndarray,
    music: np.ndarray,
    music_pre_roll_sec: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align voice and music. Music starts first by pre_roll_sec seconds.
    Returns (aligned_voice, aligned_music) — both same length, ready to sum.
    """
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    aligned_voice = np.concatenate([
        np.zeros(pre_roll_samples, dtype=np.float32),
        voice,
    ])
    total_length = len(aligned_voice)
    
    if len(music) < total_length:
        music = _loop_with_crossfade(music, total_length, sample_rate)
    
    aligned_music = music[:total_length].astype(np.float32)
    return aligned_voice, aligned_music


# ─────────────────────────────────────────────────────────────────────────────
# Fades
# ─────────────────────────────────────────────────────────────────────────────

def apply_fades(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
) -> np.ndarray:
    """Apply linear fade-in and fade-out to audio."""
    result = audio.copy()
    
    fi_samples = int(fade_in_sec * sample_rate)
    if 0 < fi_samples < len(result):
        result[:fi_samples] *= np.linspace(0.0, 1.0, fi_samples, dtype=np.float32)
    
    fo_samples = int(fade_out_sec * sample_rate)
    if 0 < fo_samples < len(result):
        result[-fo_samples:] *= np.linspace(1.0, 0.0, fo_samples, dtype=np.float32)
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LUFS normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Normalize audio to target LUFS. Returns float32, clipped to [-1, 1]."""
    min_samples = int(0.4 * sample_rate)
    if len(audio) < min_samples:
        return audio
    
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return audio
    
    if not np.isfinite(loudness):
        return audio
    
    if abs(target_lufs - loudness) > 40.0:
        return audio
    
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main mix function
# ─────────────────────────────────────────────────────────────────────────────

def mix(
    voice_audio: np.ndarray,
    voice_activity: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -8.0,
    music_pre_roll_sec: float = 2.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """
    Full mix pipeline: align → duck → overlay → fades → normalize.
    
    Called after voice FX and music FX have already been applied.
    Returns mixed mono float32 array ready for master chain + export.
    """
    # 1. Align voice and music with pre-roll
    aligned_voice, aligned_music = overlay_tracks(
        voice_audio, music_audio, music_pre_roll_sec, sample_rate
    )
    
    # 2. Extend voice_activity to match pre-roll offset
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    aligned_activity = np.concatenate([
        np.zeros(pre_roll_samples, dtype=bool),
        voice_activity,
    ])
    # Match exact length
    target_len = len(aligned_voice)
    if len(aligned_activity) < target_len:
        aligned_activity = np.concatenate([
            aligned_activity,
            np.zeros(target_len - len(aligned_activity), dtype=bool),
        ])
    aligned_activity = aligned_activity[:target_len]
    
    # 3. Compute and apply ducking
    duck_curve = compute_duck_curve(aligned_activity, sample_rate, duck_amount_db)
    ducked_music = aligned_music * duck_curve
    
    # 4. Sum voice + ducked music
    mixed = aligned_voice + ducked_music
    
    # 5. Apply fades
    mixed = apply_fades(mixed, sample_rate, fade_in_sec, fade_out_sec)
    
    # 6. Normalize loudness
    mixed = normalize_loudness(mixed, sample_rate, target_lufs)
    
    return mixed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    output_format: str = 'wav',
) -> str:
    """Export audio to a temp file. Returns absolute path."""
    suffix = f'.{output_format}'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    
    if output_format == 'wav':
        sf.write(tmp_path, audio, sample_rate, subtype='PCM_24')
    elif output_format == 'mp3':
        from pedalboard.io import AudioFile
        with AudioFile(tmp_path, 'w', samplerate=sample_rate, num_channels=1, quality=0.2) as f:
            f.write(audio.reshape(1, -1).astype(np.float32))
    else:
        raise ValueError(f'Unsupported format: {output_format!r}')
    
    return tmp_path
```

---

## 17. Performance & Memory Best Practices

### Model Sequencing (Avoid OOM)

Kokoro TTS and MusicGen must be loaded and unloaded sequentially — never simultaneously. On 8 GB VRAM systems, both models cannot fit at once.

The `pipeline.py` enforces this order:
1. Load Kokoro → generate voice → **unload Kokoro + `torch.cuda.empty_cache()`**
2. Load MusicGen → generate music → **unload MusicGen + `torch.cuda.empty_cache()`**
3. All Pedalboard processing happens on CPU — no GPU needed

### Pedalboard Processing is CPU-Based

All Pedalboard FX chains run on CPU. This is actually an advantage for MoodScape because:
- No GPU memory contention with the AI models
- JUCE's C++ implementation is extremely fast on CPU
- A 10-minute meditation processes in under 1 second on modern hardware

### Array Size Expectations

At 24,000 Hz, float32:

| Duration | Array Size |
|----------|-----------|
| 1 minute | ~5.5 MB |
| 5 minutes | ~27.6 MB |
| 10 minutes | ~55.2 MB |
| 30 minutes | ~165.6 MB |

These fit comfortably in RAM. No streaming or chunking is needed for Pedalboard processing in MoodScape.

### numpy.float32 Throughout

Always use `np.float32`, never `np.float64`. Pedalboard internally uses 32-bit floats. Passing float64 arrays triggers unnecessary conversion. All AI model outputs (Kokoro, MusicGen) should be explicitly cast with `.astype(np.float32)` immediately after generation.

---

## 18. Troubleshooting & Edge Cases

### "Clicking" or "Popping" in the Mix

**Cause:** Abrupt gain changes in the ducking curve, or loop boundaries in the music.

**Fix:**
- Ensure `compute_duck_curve` uses `filtfilt` with the Butterworth filter — not a raw boolean mask.
- Ensure `_loop_with_crossfade` uses a 1-second crossfade (not shorter) when looping music.
- Ensure `apply_fades` is applied to the full mixed signal, not to individual tracks.

### Reverb Tail Exceeds Array Length

**Cause:** Pedalboard's `Reverb` may return an array slightly longer than the input due to the reverb tail.

**Fix:** Always trim processed output to original length:
```python
result = processed.squeeze(0)[:len(audio)]
```

### pyloudnorm Fails on Short Audio

**Cause:** BS.1770 loudness measurement requires at least 400 ms of audio.

**Fix:** Guard in `normalize_loudness`:
```python
if len(audio) < int(0.4 * sample_rate):
    return audio
```

### pyloudnorm Returns -inf

**Cause:** Audio is completely silent or near-zero.

**Fix:** Guard with `np.isfinite(loudness)` check — already in the reference implementation.

### Music Is Too Loud After Normalization

**Cause:** The ducking creates large silent regions that make the integrated loudness appear lower than it sounds during the voiced sections. LUFS normalization then boosts the whole track.

**Fix:** This is expected behavior. -16 LUFS is measured over the entire track, so sections with both voice and music will be at an appropriate listening level. If output consistently sounds too loud, reduce target to -18 LUFS. Do not change the ducking depth.

### MusicGen Output Is Stereo

**Cause:** MusicGen stereo variants produce `(2, N)` tensors.

**Fix:** In `music_engine.py`, immediately after generation:
```python
audio = wav[0].cpu().numpy()  # shape: (channels, samples) or (samples,)
if audio.ndim == 2 and audio.shape[0] > 1:
    audio = audio.mean(axis=0)  # Average to mono
elif audio.ndim == 2:
    audio = audio[0]             # Already mono, just unwrap
```

### Pedalboard AudioFile MP3 Write Fails

**Cause:** Some systems lack LAME MP3 encoder support.

**Fix:** Wrap in try/except and fall back to WAV:
```python
try:
    with AudioFile(path, 'w', samplerate=sample_rate, num_channels=1, quality=0.2) as f:
        f.write(audio.reshape(1, -1))
except Exception as e:
    # Fallback: write WAV instead
    wav_path = path.replace('.mp3', '.wav')
    sf.write(wav_path, audio, sample_rate)
    return wav_path
```

### Voice Activity Mask Length Mismatch

**Cause:** TTS may produce audio that is slightly shorter or longer than expected due to synthesis timing. The `voice_activity` mask is generated sample-by-sample with the audio, so they always match — but after Pedalboard reverb processing, the voice_audio may be trimmed differently.

**Fix:** Always align voice_activity to the *post-FX* voice_audio length:
```python
# After apply_fx(voice_audio, voice_chain):
voice_activity = voice_activity[:len(voice_audio)]  # Trim to match
if len(voice_activity) < len(voice_audio):
    pad = len(voice_audio) - len(voice_activity)
    voice_activity = np.concatenate([voice_activity, np.zeros(pad, dtype=bool)])
```

---

## Summary: Parameter Quick Reference

### All Configurable Parameters for MoodScape UI

| Parameter | Default | Range | Where Used |
|-----------|---------|-------|-----------|
| Voice reverb wet level | 0.15 | 0.0–0.5 | `make_voice_chain(reverb_amount=...)` |
| Music ducking amount | -8.0 dB | -4 to -15 | `compute_duck_curve(duck_amount_db=...)` |
| Fade in duration | 3.0 sec | 0–10 | `apply_fades(fade_in_sec=...)` |
| Fade out duration | 5.0 sec | 0–10 | `apply_fades(fade_out_sec=...)` |
| Music pre-roll | 2.0 sec | 0–5 | `overlay_tracks(music_pre_roll_sec=...)` |
| LUFS target | -16.0 | -12 to -23 | `normalize_loudness(target_lufs=...)` |
| Output format | "wav" | wav, mp3 | `export_audio(output_format=...)` |

### FX Chain Parameter Stability (Do Not Expose to User)

These are tuned values — exposing them as sliders would overwhelm users without benefit:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Voice LowShelf gain | +2.0 dB | Calibrated for Kokoro-82M output |
| Voice Compressor ratio | 3:1 | Gentle leveling appropriate for TTS |
| Voice Reverb room_size | 0.3 | Small-medium room, not cathedral |
| Voice Reverb damping | 0.7 | Absorbs harshness from reverb tail |
| Music LowShelf gain | +1.5 dB | Subtle warmth enhancement |
| Music HighShelf gain | -3.0 dB | Critical: removes MusicGen digital shimmer |
| Duck smoothing cutoff | 3 Hz | 300 ms attack, 800 ms release |
| Master limiter threshold | -0.5 dB | Tight ceiling before LUFS normalization |