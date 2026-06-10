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

## 3. Signal Chain Architecture

### Kokoro TTS Path

```
Kokoro TTS (24 kHz, CPU)
  │
  ▼
┌──────────────────────────────────────┐
│ Per-chunk cleanup (postprocessor)    │
│   • DC offset removal                │
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
│   • Room-tone pauses (bandpass noise │
│     100–800 Hz, -55 dBFS)           │
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
┌──────────────────────────────────────┐
│ Pitch humanization (per speech chunk)│
│   • pyworld: drift ±6¢ @ 0.5 Hz     │
│   • vibrato ±3¢ @ 5 Hz              │
│   • jitter ±2¢ (random)             │
│   • formant shift 0.97 (3% lower)   │
│   • applied to speech only; pauses  │
│     and room-tone chunks are skipped │
└──────────────────────────────────────┘
  │
  ▼
Upsample → mix sample rate (soxr_vhq for all TTS engines)
  │
  ▼
┌──────────────────────────────────────┐
│ Unified Voice FX: build_voice_chain()│
│   1. NoiseGate (-40 dB)             │
│   2. Highpass 80 Hz                  │
│   3. PeakFilter -2.5 dB @ 400 Hz    │
│      (Q=1.0, mud cut)               │
│   4. LowShelf +1.5 dB @ 200 Hz      │
│      (warmth / proximity effect)    │
│   5. Compressor 2.5:1 @ -28 dB      │
│      (15ms/150ms; catches whisper-  │
│       level meditation delivery)    │
│   6. HighShelf +1.0 dB @ 10 kHz    │
│      (air / intimacy)               │
│   7. Convolution reverb (18% wet)   │
│   8. Limiter -1.0 dBTP              │
│                                      │
│  Soft tanh saturation (drive=1.5,   │
│  12% wet) is applied before chain.  │
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
│   2. Tape saturation (drive 1.08)   │
│   3. NoiseGate -45 dB               │
│   4. Highpass 80 Hz                  │
│   5. PeakFilter -2 dB @ 300 Hz      │
│   6. LowShelf +2 dB @ 200 Hz        │
│   7. PeakFilter -2.0 dB @ 3000 Hz   │
│   8. HighShelf -3.0 dB @ 7.5 kHz    │
│   9. HighShelf +1.0 dB @ 10 kHz     │
│  10. Lowpass 12 kHz (preserves air) │
│  11. Compressor (-20 dB, 2.5:1)     │
│  12. Limiter -1.5 dB                │
└──────────────────────────────────────┘
```

### Shared Path (Both Engines)

```
 Pre-EQ ( only)
  • Spectral repair: noisereduce (stationary, prop_decrease=0.45)
  • Tape saturation: asymmetric soft clipping (drive=0.2, bias=0.10)

ACE-Step Pre-EQ (ACE-Step only — see audio_processing.md §3)
  • Spectral repair: noisereduce (stationary, prop_decrease=0.65)
  • Tape saturation: asymmetric soft clipping (drive=0.3, bias=0.15)

Uploaded Instrumental (music_model="upload")
  • Decode (pedalboard AudioFile) → resample 48 kHz → mono → fit_to_length() loop/trim
  • Enters this shared path like a generated bed: NO pre-EQ / noise reduction / neural
    enhancement / stem separation (the file is already a finished instrumental)
  • Pre-mix normalize to -16 LUFS, then make_upload_music_chain() (light: HPF 30,
    -2 dB pocket @ 2 kHz, LPF 14 kHz, limiter)

Music FX Chain (engine-specific EQ — see audio_processing.md §3)

Neural Enhancement ( only — optional)
  • Apollo GAN (ICASSP 2025) — codec artifact removal
  • Loads after music engine unloads (~7 GB)
  • Graceful fallback if Apollo not installed

ACE-Step /  Post-EQ (ACE-Step and )
  • Organic noise floor: pink noise at -58 dB, LPF 8 kHz

Vocal Pocket Carving (applied to music before mixing)
  • HPF 30 Hz | -3 dB @ 300 Hz | -2 dB @ 1 kHz | -4 dB @ 3 kHz | LPF 12 kHz

Stereo Upmixing (opt-in)
  • Haas effect: 16ms delay on right channel, 0.92 gain
  • Music only — voice stays center-panned
  • Applied after ducking in mix()

Voice Pre-Normalization
  • Voice normalized to -18 LUFS before mixing for consistent voice-to-music ratio

Mixer
  • 8s music pre-roll, 15s music post-roll
  • Breathing sidechain duck (apply_breathing_duck) — fullband, script/VAD-aware
    - Phrases detected from voice RMS envelope (detect_phrases, -40 dB threshold)
    - Predictive S-curve descent ~600ms before each phrase (700ms ramp)
    - Holds at duck_amount_db (-16 dB default) during speech
    - S-curve release over 1.5s; +1.5 dB lift during pauses >= 1.5s
    - Reactive envelope-follower safety net for off-script breaths
    - Base music: -16 dB | Duck: -16 dB → bed ~28 LU under voice during speech
  • Exponential fades (3s in / 8s out) — natural DAW-style curves

QA checks (11 metrics + ducking QA — see §9)

Master Chain
  • HPF 30 Hz → Compressor (1.5:1 @ -12 dB, 50ms/200ms) → High-shelf +1 dB @ 12 kHz
  • No limiter in the chain — LUFS-normalize then mixer.true_peak_limit() at export
  Applied in export_audio(); audio streamed to file in 20s chunks

Export: 44.1 kHz or 48 kHz / 24-bit WAV | Target: -16 LUFS | TP ceiling: -1.0 dBTP
```

---

## 4. Why This Order?

- **Spectral gating** (Kokoro only) runs on dry 24 kHz audio — noise profile is most stable before any FX.
- **Unified voice FX chain** (`build_voice_chain()`) runs at the mix rate (44.1/48 kHz) so EQ filters operate well below Nyquist. Convolution reverb is integrated into this chain, with the LPF placed after reverb to catch reverb HF content.
- **Vocal pocket carving** happens on the music track, not the voice, to preserve the voice chain integrity.
- **Voice pre-normalization** to -18 LUFS ensures consistent voice-to-music ratio regardless of TTS engine.
- **Breathing sidechain duck** runs offline (full gain envelope pre-computed from detected phrases) for precise, predictive timing — the bed starts falling before the first syllable and rises gradually in pauses.
- **Master chain** (HPF + gentle bus compressor + air shelf; no limiter) is applied in `export_audio()`, with LUFS normalization and oversampled true-peak limiting after it; audio is streamed to file in 20 s chunks to prevent memory spikes.

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

### Kokoro: Broad Presence Boost + High-Shelf (inside `build_voice_chain()`)
```
PeakFilter(3 kHz, +1.0 dB, Q=0.6)   — gentle broad lift across 2–5 kHz for intelligibility and
                                       warm, forward vocal character; not a sharp resonance boost
HighShelfFilter(7.5 kHz, -3.0 dB)   — single de-harsh shelf removing ISTFTNet HF artifacts
```
The +1.0 dB broad presence boost (Q=0.6) preserves Kokoro's natural expressiveness. A wider Q (0.6 vs typically 1.0+) means this is a gentle lift across the full upper midrange, adding intelligibility and warmth without amplifying any single resonant frequency. The -3.0 dB shelf at 7.5 kHz tames vocoder HF artifacts without over-dulling the voice's 'air'. Tape saturation (drive=1.05) adds harmonic warmth before the chain.

### F5-TTS: Split-Band De-Esser (before voice FX chain)
Isolates 4–8 kHz sibilant band via 4th-order Butterworth bandpass, compresses aggressively (-20 dB threshold, 4:1 ratio, 0.5ms attack, 10ms release), recombines with non-sibilant signal. Followed by subtle tape saturation (`tanh(x*1.08)/1.08`) for harmonic warmth.

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
| `core/kokoro_tts/postprocessor.py` | Per-chunk cleanup, crossfade assembly, spectral gating, unified voice FX chain (`build_voice_chain()`) |
| `core/f5_tts/postprocessor.py` | `F5MasteringEngine` — split-band de-esser, crossfade assembly, F5 Phase B EQ |
| `core/breath_sounds.py` | Shared breath sample loader (cached, 75ms fade-in/out) |
| `core/mixer.py` | Breathing sidechain duck (script + reactive), overlay, cosine crossfade looping, exponential fades, LUFS, true-peak limiting, streaming export |
| `core/kokoro_tts/preprocessor.py` | Script parsing (breath/pause/voice), text expansion, IPA injection, prosody punctuation, token chunking |
| `core/f5_tts/preprocessor.py` | Script parsing (breath/pause/voice), character-count chunking |
| `core/qa_monitor.py` | QA: LUFS, clipping, spectral balance, silence gaps, silence ratio, spectral rolloff, onset strength, spectral smoothness, harmonic stability, onset density, dynamic range, voice-music ratio, ducking smoothness, composite score |
| `core/stereo_upmix.py` | Haas-effect stereo upmixing (opt-in, music only) |
| `core/pipeline.py` | Orchestrates the full signal chain end-to-end |

---

## 9. Quality Assurance and A/B Selection

After the master chain, `qa_monitor.run_qa_checks()` runs 11 automated checks:

| Check | Pass Condition |
|-------|---------------|
| LUFS verification | Within ±2 dB of -16 LUFS |
| Clipping detection | < 0.1% samples at ±0.99 |
| Spectral balance | Warmth (100–300 Hz) ≥ presence (2–5 kHz) |
| Silence gaps | No gap > 15s |
| Silence ratio | 15–60% (meditation pacing) |
| Spectral rolloff | 85th-percentile rolloff ≤ 8000 Hz |
| Onset strength | Peak/median onset ratio < 5.0 |
| Spectral smoothness | Centroid variance < 50 |
| Harmonic stability | Chroma autocorrelation > 0.85 |
| Onset density | < 0.5 onsets/sec |
| Dynamic range | RMS std-dev < 0.01 |

**A/B selection in music engines:** On retry, all candidates are scored via `compute_composite_score()` (weighted composite of all 11 checks, range 0–1). The highest-scoring candidate is selected. Early exit if score > 0.8.

---

## 10. Verification Checklist

- [x] `[breath]` / `[inhale]` / `[exhale]` tags inject actual breath audio samples
- [x] Kokoro: spectral gating (noisereduce, stationary, prop_decrease=0.6, n_std_thresh=2.0, freq_mask_smooth_hz=500) reduces ISTFTNet hiss ~6 dB
- [x] F5-TTS (Vocos): no spectral gating needed — clean vocoder output
- [x] Voice FX chain EQ and limiting at mix sample rate (44.1 or 48 kHz), well below Nyquist
- [x] Kokoro: 9.5 kHz lowpass masks 12 kHz Nyquist brick-wall
- [x] F5-TTS: 12 kHz lowpass preserves Vocos native air bandwidth
- [x] Sub-bass below 80 Hz removed from voice (HighpassFilter)
- [x] Subsonic below 30 Hz removed from master (HighpassFilter)
- [x] Breathing sidechain duck: 600ms pre-descent, 700ms S-curve attack, 1500ms release, +1.5 dB pause lift
- [x] Equal-power cosine² crossfades: 300ms TTS assembly, 2s music, 2s music loop
- [x]  seams: micro-crossfade (64-sample triangular at zero-crossing) after _stitch()
- [x] Exponential fade curves (default) for natural meditation transitions
- [x] Streaming export in 20s chunks — no full-array load at export time
- [x] LUFS target: -16 LUFS integrated (per session)
- [x] 11 QA checks including spectral smoothness, harmonic stability, onset density, and dynamic range
- [x] A/B selection: best composite-scored candidate kept across retry attempts
- [x] Convolution reverb IR selectable: warm_studio / wooden_hall / stone_chapel
