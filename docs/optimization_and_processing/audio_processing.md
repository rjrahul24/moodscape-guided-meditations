# Audio Processing & Memory Optimization Guide

This document outlines the core architectural logic that powers the generation, standardizations, and export memory management of MoodScape's audio.

## 1. Sample Rate Pipeline Standardization
Generative AI audio models operate natively across a multitude of disparate sample frequencies. 
- **Voice TTS (Kokoro/Parler)**: ~24kHz natively
- **MusicGen**: ~32kHz natively

If these arrays are fed blindly into identical FX Pedals or summed together, extreme timing synchronization and pitch-shift ("chipmunking") degradation occurs. 

**MoodScape enforces an aggressive upsampling strategy:**
All generative outputs are intercepted instantly to upsample to **44.1kHz (44100Hz)**. This is processed securely via TorchAudio's `functional.resample` to maintain precision format boundaries across PyTorch CUDA/MPS configurations before Pedalboard handles FX processing on CPU paths. 

## 2. Dynamic Audio Mixing Processing

### Mask-Based Ducking (Primary Method)
MoodScape uses **voice-activity mask–driven ducking** (`apply_mask_ducking` in `mixer.py`) as the primary ducking method. Instead of re-deriving vocal presence via RMS energy, this method uses the pre-computed `voice_activity` boolean mask — a sample-accurate ground-truth map of when the voice is speaking, constructed in `kokoro_engine.synthesize()`.

- **Lookahead**: The ducking envelope is shifted **100ms** earlier via `np.roll`, so the music starts fading *before* the first syllable — matching broadcast/DAW behaviour.
- **Attack**: **150ms** — a slower, more natural fade-in for music suppression.
- **Release**: **2000ms** (2 seconds) — a deliberately slow recovery that matches meditation pacing, allowing music to gradually swell back during pauses.
- **Duck depth**: **-9 dB** — firmly ducks music during narration while keeping it subtly audible.

This approach is more precise and computationally cheaper than the RMS-based method, which is retained as `apply_rms_ducking` for fallback use (e.g. Parler TTS edge cases where a mask may be unavailable).

### Spectral Masking
To clarify vocal enunciation, MoodScape applies a **HighShelfFilter** on ambient tracks that gently rolls off frequencies above **10kHz by -4dB**. This softens MusicGen's synthetic high-end "digital shimmer" while preserving the full ambient timbre between 3–10kHz. Previously a 3kHz LowpassFilter was used, but this destroyed ambient pad harmonics.

## 3. Music FX Chain

The current music FX chain in `audio_processor.py`:

```python
def make_music_chain() -> Pedalboard:
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.0, q=0.7),      # Low-end warmth
        PeakFilter(cutoff_frequency_hz=1800, gain_db=-3.0, q=0.6),   # Vocal pocket notch
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=-4.0),   # Gentle HF rolloff
        Limiter(threshold_db=-1.0),
    ])
```

## 4. Master Chain

The master chain now includes a **35Hz subsonic highpass filter** as the first plugin. This removes inaudible MusicGen rumble that otherwise consumes digital headroom, causing the limiter to trigger earlier and produce a quieter or more compressed final mix.

```python
def make_master_chain() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=35.0),   # Remove inaudible MusicGen rumble
        Gain(gain_db=-3.0),
        Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=300.0),
        Limiter(threshold_db=-0.1),
    ])
```

## 5. Equal-Power Crossfades

All crossfade operations in MoodScape use **equal-power cosine crossfades** (`cos²/sin²`) rather than linear fades. This prevents the ~3dB loudness dip at the midpoint that linear crossfades produce. This applies to:

- **MusicGen segment stitching** (`music_engine.py`): 2-second crossfade at each 30-second segment boundary.
- **Music looping** (`mixer.py`): 2-second crossfade when music is looped to cover the full meditation duration.

## 6. Streaming Exports (Memory Safeties)
Exporting long audio combinations (e.g. up to 12 hour stretches of 44.1kHz `np.float32`) quickly bottlenecks local compute memory footprint and invokes out-of-memory kernel panics.

**Chunked AudioFile Protocol:**
1. A **unified LUFS target of −14 LUFS** (streaming distribution standard — Spotify, Apple Music, YouTube) calculates a constant **Gain Scalar** via a lightweight `pyloudnorm` meter implementation.
2. MoodScape initiates Pedalboard's underlying `AudioFile` protocol, binding directly to disk buffers.
3. The generation sweeps through the multi-hour array iteratively in **20-second chunk boundaries**.
4. The memory blocks sequentially get normalized, wrapped inside the `HighpassFilter(35Hz)` → `Gain(-3.0)` → `Compressor(-18dB, 2:1, 30ms/300ms)` → `Limiter(-0.1)` Master Effect protections, and written to disk without needing simultaneous loading.

This paradigm scales indefinitely for background batch generations.

## 7. Text Normalization (TTS Pre-Processing)

Before any text reaches the TTS engine, `text_preprocessor.py` runs an `expand_for_tts()` pass that:

- **Expands abbreviations**: `sec` → `seconds`, `min` → `minutes`, `Hz` → `hertz`, `e.g` → `for example`, etc.
- **Converts digits to words**: integers 0–999 are converted to their English word equivalents (e.g., `4` → `four`, `120` → `one hundred and twenty`).

This prevents Kokoro's G2P engine from mispronouncing or rushing through numeric and abbreviated tokens in meditation scripts.
