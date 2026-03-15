# Advanced Audio Post-Processing Pipeline

## 1. Objective

A high-fidelity vocal and music post-processing "Mastering Chain" for MoodScape. The pipeline handles raw output from both **Kokoro TTS** and **F5-TTS**, removing digital artifacts while enhancing warmth, intimacy, and musical cohesion. Optimized for Apple Silicon M1 Max with 36 GB unified memory.

---

## 2. Technical Stack

| Library | Purpose |
|---|---|
| `pedalboard` ≥ 0.9 | Spotify's audio plugin host — EQ, compression, de-essing, limiting, convolution reverb |
| `torchaudio` + `torch` | Tensor-based resampling on MPS |
| `pyloudnorm` | ITU-R BS.1770-4 loudness normalization (LUFS) |
| `scipy` | Butterworth filtering, Welch PSD for QA checks |
| `noisereduce` | Stationary spectral gating (Kokoro ISTFTNet hiss reduction) |
| `librosa` | Spectral rolloff, onset strength, high-accuracy resampling (soxr_vhq) |

---

## 3. Two-Phase Signal Chain Architecture

### Kokoro TTS Path

```
Kokoro TTS (24 kHz, CPU)
  │
  ▼
┌──────────────────────────────────────┐
│ Per-chunk cleanup (postprocessor)    │
│   • Hard-clip guard [-1, 1]          │
│   • Trailing silence trim (-45 dBFS) │
│   • Spectral flatness check (loops)  │
│   • RMS normalize to -23 dBFS        │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│ Segment assembly                     │
│   • 300ms cosine² crossfade          │
│   • 25ms pre-roll + 100ms fade-in    │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│ Spectral gating noise reduction      │
│   • noisereduce (stationary mode)    │
│   • prop_decrease=0.6 (conservative) │
│   • Reduces ISTFTNet hiss ~6 dB      │
└──────────────────────────────────────┘
  │
  ▼
Upsample → mix sample rate (44.1 kHz or 48 kHz)
  │
  ▼
┌──────────────────────────────────────┐
│ Phase B: KokoroMasteringEngine       │
│   1. Highpass 80 Hz                  │
│   2. LowShelf +2 dB @ 200 Hz        │
│   3. PeakFilter -1.5 dB @ 400 Hz    │
│   4. PeakFilter +1.5 dB @ 3500 Hz   │
│   5. PeakFilter +1.2 dB @ 2500 Hz   │
│   6. De-Esser (7 kHz: +4→compress   │
│      →-4 dB, boost-compress-cut)    │
│   7. Lowpass 9.5 kHz (Nyquist mask) │
│   8. Limiter -0.5 dB                │
└──────────────────────────────────────┘
```

### F5-TTS Path

```
F5-TTS (24 kHz, Vocos vocoder, MPS)
  │
  ▼
┌──────────────────────────────────────┐
│ Per-chunk synthesis cleanup          │
│   • Trailing silence trim (-45 dBFS) │
│   • WPM normalization via time-      │
│     stretch (±15% of session median) │
│   • Silero VAD noise suppression     │
│   • Chained reference continuity     │
│     (last 50% of prev chunk as ref)  │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│ Segment assembly                     │
│   • 300ms cosine crossfade           │
│   • Room tone in pauses (~-60 dBFS)  │
│   • Breath sounds at [inhale]/       │
│     [exhale]/[breath] tags           │
└──────────────────────────────────────┘
  │
  ▼
Upsample → mix sample rate (soxr_vhq)
  │
  ▼
┌──────────────────────────────────────┐
│ Phase B: F5MasteringEngine           │
│   1. Split-band de-esser (4–8 kHz)  │
│   2. NoiseGate -50 dB               │
│   3. Highpass 80 Hz                  │
│   4. PeakFilter -2 dB @ 300 Hz      │
│   5. LowShelf +2 dB @ 200 Hz        │
│   6. PeakFilter +1.5 dB @ 2800 Hz   │
│   7. HighShelf +1.5 dB @ 10 kHz     │
│   8. Lowpass 13 kHz (preserves air) │
│   9. Compressor (-24 dB, 2.5:1)     │
│  10. Reverb (10% wet)               │
│  11. Limiter -1.5 dB                │
└──────────────────────────────────────┘
```

### Shared Path (Both Engines)

```
Voice FX Chain (at mix sample rate)
  • Convolution reverb (IR: warm_studio / wooden_hall / stone_chapel, 6-12% wet)
  • Limiter -1.0 dB

Music FX Chain (engine-specific EQ — see audio_processing.md §3)

Vocal Pocket Carving (applied to music before mixing)
  • HPF 30 Hz | -3 dB @ 300 Hz | -2 dB @ 1 kHz | -4 dB @ 3 kHz | LPF 12 kHz

Mixer
  • 4s music pre-roll, 8s music post-roll
  • Lookahead sidechain ducking
    - Base music: -17 dB | Duck: -21 dB → ~-38 dB total during speech
    - Lookahead: 75ms | Attack: 50ms | Release: 500ms
  • Linear fades (3s in / 6s out)

QA checks (7 metrics — see §9)

Master Chain
  • HPF 30 Hz → Compressor (-24 dB, 1.5:1, 30ms/300ms) → Gain (+1 dB) → Limiter (-1.5 dB)
  Applied per-chunk in export_audio() to avoid memory spikes

Export: 44.1 kHz or 48 kHz / 24-bit WAV | Target: -19 LUFS integrated
```

---

## 4. Why This Order?

- **Spectral gating** (Kokoro only) runs on dry 24 kHz audio — noise profile is most stable before any FX.
- **Phase B mastering** runs at the mix rate (44.1/48 kHz) so EQ filters operate well below Nyquist.
- **Voice FX chain** (convolution reverb) follows Phase B — reverb applied to the already-mastered signal.
- **Vocal pocket carving** happens on the music track, not the voice, to preserve the voice chain integrity.
- **Sidechain ducking** runs offline (full envelope pre-computed with lookahead shift) for precise timing.
- **Master chain** runs per-chunk in the streaming export — prevents memory spikes on long sessions.

---

## 5. Breath Sound Injection

`[breath]`, `[inhale]`, and `[exhale]` tags in meditation scripts inject real breath audio samples instead of silence:

| Tag | File | Duration |
|-----|------|----------|
| `[breath]` | `assets/breath_sounds/breath.wav` | 1.2s |
| `[inhale]` | `assets/breath_sounds/inhale.wav` | 1.5s |
| `[exhale]` | `assets/breath_sounds/exhale.wav` | 1.8s |

**How it works:** Both preprocessors emit `{"type": "breath", "subtype": "..."}` segments. TTS engines load and splice the audio via `core/breath_sounds.py` (cached per session, 75ms cosine fade applied). F5-TTS blends room tone behind the breath audio. Falls back to 1.2s silence if files are missing.

**Regenerate samples:** `python scripts/generate_breath_samples.py`

---

## 6. De-Essing Details

### Kokoro: Boost-Compress-Cut (inside Phase B)
```
PeakFilter(7 kHz, +4 dB, Q=2.0)
Compressor(-20 dB, 3:1, 1ms/50ms)
PeakFilter(7 kHz, -4 dB, Q=2.0)
```
Only triggers when sibilance exceeds the threshold. ±4 dB (reduced from ±6 dB) avoids over-attenuation of natural consonants.

### F5-TTS: Split-Band De-Esser (before Phase B)
Isolates 4–8 kHz sibilant band via 4th-order Butterworth bandpass, compresses aggressively (-18 dB threshold, 4:1 ratio, 0.5ms attack, 10ms release), recombines with non-sibilant signal.

---

## 7. Convolution Reverb IR Files

Three impulse responses are bundled in `assets/impulse_responses/`:

| IR file | Character | Use case |
|---------|-----------|----------|
| `warm_studio.wav` | Short decay, intimate | Default — body scans, breathwork |
| `wooden_hall.wav` | Medium space, natural warmth | Visualizations, journey meditations |
| `stone_chapel.wav` | Long decay, ethereal | Deep relaxation, expansive sessions |

Selectable in the UI under "Reverb Space". Applied via Pedalboard `Convolution` at 6–12% wet.

---

## 8. Implementation Files

| File | Role |
|---|---|
| `core/kokoro_tts/postprocessor.py` | `KokoroMasteringEngine` — per-chunk cleanup, crossfade, spectral gating, Kokoro Phase B EQ, voice FX chain |
| `core/f5_tts/postprocessor.py` | `F5MasteringEngine` — split-band de-esser, crossfade assembly, F5 Phase B EQ |
| `core/breath_sounds.py` | Shared breath sample loader (cached, 75ms fade-in/out) |
| `core/audio_processor.py` | Music FX chains (MusicGen/ACE-Step/Lyria), vocal pocket chain, master chain |
| `core/mixer.py` | Lookahead sidechain ducking, overlay, cosine crossfade looping, fades, LUFS, streaming export |
| `core/kokoro_tts/preprocessor.py` | Script parsing (breath/pause/voice), text expansion, IPA injection, prosody punctuation, token chunking |
| `core/f5_tts/preprocessor.py` | Script parsing (breath/pause/voice), character-count chunking |
| `core/qa_monitor.py` | QA: LUFS, clipping, spectral balance, silence gaps, silence ratio, spectral rolloff, onset strength, composite score |
| `core/pipeline.py` | Orchestrates the full signal chain end-to-end |

---

## 9. Quality Assurance and A/B Selection

After the master chain, `qa_monitor.run_qa_checks()` runs 7 automated checks:

| Check | Pass Condition |
|-------|---------------|
| LUFS verification | Within ±2 dB of -16 LUFS |
| Clipping detection | < 0.1% samples at ±0.99 |
| Spectral balance | Warmth (100–300 Hz) ≥ presence (2–5 kHz) |
| Silence gaps | No gap > 15s |
| Silence ratio | 15–60% (meditation pacing) |
| Spectral rolloff | 85th-percentile rolloff ≤ 8000 Hz |
| Onset strength | Peak/median onset ratio < 5.0 |

**A/B selection in music engines:** On retry, all candidates are scored via `compute_composite_score()` (weighted composite of all 7 checks, range 0–1). The highest-scoring candidate is selected. Early exit if score > 0.8.

---

## 10. Verification Checklist

- [x] `[breath]` / `[inhale]` / `[exhale]` tags inject actual breath audio samples
- [x] Kokoro: spectral gating (noisereduce, stationary, prop_decrease=0.6) reduces ISTFTNet hiss ~6 dB
- [x] F5-TTS (Vocos): no spectral gating needed — clean vocoder output
- [x] Phase B EQ and limiting at mix sample rate (44.1 or 48 kHz), well below Nyquist
- [x] Kokoro: 9.5 kHz lowpass masks 12 kHz Nyquist brick-wall
- [x] F5-TTS: 13 kHz lowpass preserves Vocos native air bandwidth
- [x] Sub-bass below 80 Hz removed from voice (HighpassFilter)
- [x] Subsonic below 30 Hz removed from master (HighpassFilter)
- [x] Sidechain ducking: 75ms lookahead, 50ms attack, 500ms release
- [x] Equal-power cosine² crossfades: 300ms TTS assembly, 2s music, 2s music loop
- [x] MusicGen seams: micro-crossfade (64-sample triangular at zero-crossing) after _stitch()
- [x] Streaming export in 20s chunks — no full-array load at export time
- [x] LUFS target: -19 LUFS integrated (per session)
- [x] 7 QA checks including spectral rolloff and onset strength (new)
- [x] A/B selection: best composite-scored candidate kept across retry attempts
- [x] Convolution reverb IR selectable: warm_studio / wooden_hall / stone_chapel
