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
MoodScape uses **voice-activity mask-driven ducking** (`apply_mask_ducking` in `mixer.py`) as the primary ducking method. Instead of re-deriving vocal presence via RMS energy, this method uses the pre-computed `voice_activity` boolean mask — a sample-accurate ground-truth map of when the voice is speaking, constructed natively in `core/kokoro_tts/engine.py` or `core/parler_tts/engine.py`.

The `mix()` function calls `apply_mask_ducking` with meditation-optimized parameters:

- **Lookahead**: **400ms** — music starts fading well before the first syllable for a smooth, breath-like transition.
- **Attack**: **900ms** — very slow fade-in for music suppression; combined with 400ms lookahead, the total pre-duck window is ~1.3s.
- **Release**: **2500ms** (2.5 seconds) — deliberately slow recovery that matches meditation pacing, allowing music to gradually swell back during pauses.
- **Duck depth**: Configurable via UI (default **-21 dB**), applied on top of the base music volume (-17 dB), yielding -38 dB total during speech — nearly inaudible behind the voice.

The raw `apply_mask_ducking` defaults (-9 dB, 150ms attack, 2000ms release, 100ms lookahead) are retained for non-meditation use cases. The RMS-based method (`apply_rms_ducking`) is available as a fallback.

### Spectral Masking
To clarify vocal enunciation, MoodScape applies a **HighShelfFilter** on ambient tracks that gently rolls off frequencies above **10kHz by -4dB**. This softens MusicGen's synthetic high-end "digital shimmer" while preserving the full ambient timbre between 3–10kHz. Previously a 3kHz LowpassFilter was used, but this destroyed ambient pad harmonics.

## 3. Music FX Chain

The current music FX chain in `audio_processor.py`:

```python
def make_music_chain() -> Pedalboard:
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.0, q=0.7),       # Low-end warmth
        PeakFilter(cutoff_frequency_hz=1500, gain_db=-5.0, q=0.3),    # Wide vocal pocket notch
        PeakFilter(cutoff_frequency_hz=5500, gain_db=0.8, q=0.6),     # Clarity/air presence
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-3.0),     # HF shimmer rolloff
        Limiter(threshold_db=-1.0),
    ])
```

Engine-specific variants exist for ACE-Step 1.5 (`make_acestep_music_chain`) and Lyria RealTime (`make_lyria_music_chain`) in `core/audio_processor.py`.

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

## 8. Spectral Gating Noise Reduction

After chunk assembly and before the voice FX chain, Kokoro audio passes through a **stationary spectral gating** step (`reduce_synthesis_noise()` in `core/kokoro_tts/postprocessor.py`). This uses the `noisereduce` library with conservative parameters (`prop_decrease=0.6`, `n_std_thresh=1.5`) to reduce low-level ISTFTNet vocoder hiss by ~6 dB without damaging soft consonants.

This replaces the neural denoiser (resemble-enhance), which is disabled on Apple Silicon due to instability. The spectral gating approach is lighter-weight, deterministic, and runs entirely on CPU.
