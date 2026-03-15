# Audio Processing & Memory Optimization Guide

This document outlines the core architectural logic that powers the generation, standardizations, and export memory management of MoodScape's audio.

## 1. Sample Rate Pipeline Standardization

Generative AI audio models operate at different native rates:

| Source | Native rate | Notes |
|--------|-------------|-------|
| Kokoro TTS | 24 kHz | CPU-only on Apple Silicon |
| F5-TTS | 24 kHz | MPS (Apple Silicon GPU) |
| MusicGen | 32 kHz | CPU / MPS via AudioCraft |
| ACE-Step 1.5 | **48 kHz** stereo | MLX backend |
| Lyria RealTime | 48 kHz stereo | Cloud WebSocket API |

**MoodScape pipeline resample strategy:**

All audio is upsampled to a *mix sample rate* before Pedalboard FX and mixing. The mix rate depends on the music engine selected:

- **ACE-Step / Lyria path**: mix at **48 kHz** (preserves native fidelity of the music engine).
- **MusicGen path**: mix at **44.1 kHz** (MusicGen's 32 kHz is upsampled via Kaiser-windowed sinc).
- The mix sample rate is also the export rate (configurable in the UI via "48 kHz Output" checkbox).
- TTS audio (24 kHz) is upsampled to the mix rate:
  - Kokoro: `torchaudio.functional.resample` (rolloff=0.9475, lowpass_filter_width=64)
  - F5-TTS: `librosa.resample(res_type="soxr_vhq")` — highest accuracy mode, minimises zero-crossing errors
- **Rule**: never downsample then re-upsample. Always upsample from the lower-rate source.

## 2. Dynamic Audio Mixing Processing

### Lookahead Sidechain Ducking (Primary Method)

MoodScape uses **vectorized offline lookahead sidechain ducking** (`apply_rms_ducking` in `mixer.py`) as the primary ducking method. The method pre-computes the full voice RMS envelope in advance, then shifts it backward by a lookahead offset so the music ducks *before* the voice onset — matching professional broadcast DAW behaviour.

The `mix()` function calls `apply_rms_ducking` with these meditation-optimized parameters:

- **Lookahead**: **75ms** — music starts fading before the first syllable.
- **Attack**: **50ms** — music drops promptly once the voice begins.
- **Release**: **500ms** — music recovers gently during pauses.
- **Duck depth**: Configurable via UI (default **-21 dB**), applied on top of the base music volume (-17 dB), yielding ~-38 dB total during speech — nearly inaudible behind the voice.

A secondary `apply_envelope_ducking()` (10ms lookahead, 800ms release) is available in `mixer.py` for alternative ducking curves.

### Spectral Masking (Vocal Pocket Carving)

Before mixing, `make_vocal_pocket_chain()` in `core/audio_processor.py` carves spectral room for the voice in the music track:
- HPF 30 Hz (subsonic)
- -3 dB @ 300 Hz (low-mid mud)
- -2 dB @ 1 kHz (vowel space)
- -4 dB @ 3 kHz (sibilance / consonant zone)
- LPF 12 kHz

## 3. Music FX Chains

Three engine-specific chains exist in `core/audio_processor.py`:

### MusicGen (`make_music_chain`)
```python
PeakFilter(300 Hz, +2.0 dB, Q=0.7)       # Low-end warmth
PeakFilter(5500 Hz, +0.8 dB, Q=0.6)      # Clarity/air presence (psychoacoustic super-resolution)
HighShelfFilter(8000 Hz, -3.0 dB)         # Targets 8-12 kHz autoregressive shimmer
Limiter(-1.0 dB)
```

### ACE-Step 1.5 (`make_acestep_music_chain`)
```python
HighpassFilter(60 Hz)                      # Sub-bass removal
LowShelfFilter(200 Hz, +2.0 dB)           # Warmth
HighShelfFilter(10000 Hz, -1.0 dB)        # Gentle HF smoothing
Compressor(-20 dB, 2.5:1, 80ms/800ms)    # Glue compression
Limiter(-0.5 dB)
```

### Lyria RealTime (`make_lyria_music_chain`)
```python
HighpassFilter(60 Hz)
PeakFilter(250 Hz, -1.5 dB, Q=0.8)       # Mud notch
HighShelfFilter(9000 Hz, -2.5 dB)         # More aggressive HF rolloff (brighter source)
Compressor(-18 dB, 2:1, 80ms/500ms)
Limiter(-0.5 dB)
```

## 4. Master Chain

```python
def make_master_chain() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(30 Hz),              # Subsonic removal
        Compressor(-24 dB, 1.5:1, 30ms/300ms),  # Gentle glue compression
        Gain(+1.0 dB),                      # Makeup gain
        Limiter(-1.5 dB),                   # True-peak safe for lossy codecs
    ])
```

The master chain is applied per-chunk in `export_audio()` via streaming to avoid memory spikes on long sessions.

## 5. Equal-Power Crossfades

All crossfade operations use **equal-power cosine crossfades** (cos²/sin²) rather than linear fades. This prevents the ~3 dB loudness dip at the midpoint. This applies to:

- **MusicGen segment stitching** (`music_engine.py`): 2-second macro crossfade at each segment seam, plus a **micro-crossfade** (64-sample triangular window at zero-crossing) to eliminate residual HF clicks.
- **ACE-Step continuation** (`acestep_engine.py`): 2-second equal-power cosine crossfade at each cover segment seam.
- **Music looping** (`mixer.py`): 2-second crossfade when music is looped to cover the full meditation duration.
- **TTS chunk assembly**: 300ms cosine-squared crossfade for both Kokoro and F5-TTS engines.

## 6. Paralinguistic Breath Sound Injection

Both TTS engines support audible breath cues at tagged positions in the meditation script. Instead of inserting silence, actual breath audio samples are spliced in.

**Supported tags in scripts:**

| Tag | Injected sound | Duration |
|-----|---------------|----------|
| `[breath]` | Neutral breath | 1.2s |
| `[inhale]` | Rising inhale | 1.5s |
| `[exhale]` | Falling exhale | 1.8s |

**How it works:**
- `core/kokoro_tts/preprocessor.py` and `core/f5_tts/preprocessor.py` parse these tags and emit `{"type": "breath", "subtype": "inhale"|"exhale"|"breath"}` segments (not pause segments).
- The TTS engines load the appropriate WAV from `assets/breath_sounds/` via `core/breath_sounds.py` (cached module-level, 75ms cosine fade-in/fade-out applied).
- F5-TTS blends the breath audio with low-amplitude room tone to avoid digital silence.
- If the WAV files are missing, both engines fall back to 1.2s of silence (graceful degradation).

**Regenerating the synthetic breath samples:**
```bash
python scripts/generate_breath_samples.py
```
This creates `assets/breath_sounds/inhale.wav`, `exhale.wav`, and `breath.wav` at 24 kHz mono.

## 7. Streaming Exports (Memory Safety)

Long audio arrays are exported chunk-by-chunk using Pedalboard's `AudioFile` protocol:

1. A per-session LUFS gain scalar is pre-computed via `pyloudnorm` (ITU-R BS.1770-4 integrated loudness, target **-19 LUFS**).
2. The master chain runs over the full in-memory array, then export streams in **20-second chunks**.
3. Resampling (to 44.1 kHz or 48 kHz) and normalization apply per-chunk.
4. True-peak safety clip at -1.5 dBFS (-0.84 linear) before writing.

LUFS targets by mode:
- Full mix (voice + music): **-19 LUFS**
- Stereo: **-19 LUFS** / Mono: **-21 LUFS** (EBU R128 offset)

## 8. Text Normalization (TTS Pre-Processing)

Before any text reaches the Kokoro TTS engine, `core/kokoro_tts/preprocessor.py` runs:

- **Abbreviation expansion**: `sec` → `seconds`, `min` → `minutes`, `Hz` → `hertz`, `e.g.` → `for example`, etc.
- **Digit-to-words**: integers 0–999 converted to English words (`120` → `one hundred and twenty`).
- **IPA phoneme injection**: 30+ Sanskrit/yoga terms injected with explicit IPA (`chakra` → `[chakra](/tʃɑːkɹə/)`).
- **Prosody punctuation**: commas inserted at breath-group boundaries (max 12 words) before conjunctions.
- **Token-aware chunking**: sentences merged into 100–150 token chunks (400-token ceiling).

F5-TTS preprocessing does **not** include these steps — it feeds raw prose directly to the model's G2P layer.

## 9. Spectral Gating Noise Reduction

After Kokoro chunk assembly and before the voice FX chain, audio passes through **stationary spectral gating** (`reduce_synthesis_noise()` in `core/kokoro_tts/postprocessor.py`). This uses the `noisereduce` library with conservative parameters (`prop_decrease=0.6`, `n_std_thresh=1.5`) to reduce ISTFTNet vocoder hiss by ~6 dB without damaging soft consonants.

F5-TTS (Vocos vocoder) does not require spectral gating — the vocoder output is already clean.

## 10. Quality Assurance Monitoring

`core/qa_monitor.py` runs automated checks after the master chain and before export:

| Check | Method | Pass Condition |
|-------|--------|---------------|
| **LUFS verification** | pyloudnorm integrated loudness | Within ±2 dB of -16 LUFS target |
| **Clipping detection** | Sample threshold at 0.99 | < 0.1% clipped samples |
| **Spectral balance** | Welch PSD (scipy) warmth vs presence | Warmth (100–300 Hz) ≥ presence (2–5 kHz) |
| **Silence gaps** | RMS windowing (100ms) | No gap > 15s |
| **Silence ratio** | RMS windowing (50ms) | 15–60% (meditation pacing) |
| **Spectral rolloff** | librosa 85th-percentile rolloff | Median rolloff ≤ 8000 Hz |
| **Onset strength** | librosa onset envelope | Peak/median ratio < 5.0 |

For per-segment regeneration decisions (music engines), `compute_composite_score()` combines all checks into a single float ∈ [0, 1] for A/B selection across retry candidates.
