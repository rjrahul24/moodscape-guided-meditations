# Advanced Audio Post-Processing Pipeline

## 1. Objective

A high-fidelity vocal and music post-processing "Mastering Chain" for MoodScape. The pipeline handles raw output from both **Kokoro** and **Parler-TTS 2.2B**, removing digital artifacts while enhancing warmth, intimacy, and musical cohesion. Optimized for Apple Silicon with 36 GB unified memory.

---

## 2. Technical Stack

| Library | Purpose |
|---|---|
| `resemble-enhance` | AI-driven neural vocal restoration & denoising |
| `pedalboard` ≥ 0.9 | Spotify's audio plugin host — EQ, compression, de-essing, limiting |
| `torchaudio` + `torch` | Tensor-based resampling (24 kHz → 44.1 kHz) on MPS |
| `pyloudnorm` | ITU-R BS.1770 loudness normalization (LUFS) |
| `scipy` | Butterworth filtering for sidechain ducking curves |

```bash
pip install resemble-enhance pedalboard scipy torchaudio pyloudnorm
```

---

## 3. Two-Phase Signal Chain Architecture

The mastering engine is split into **two phases** placed at different points in the pipeline for optimal signal quality:

```
TTS (24 kHz)
  │
  ▼
┌──────────────────────────────────────┐
│ Per-chunk cleanup (postprocessor)    │
│   • Hard-clip guard, trailing trim   │
│   • Spectral flatness detection      │
│   • RMS normalization (-23 dBFS)     │
│   • 22ms cos² crossfade assembly     │
│   • 100ms fade-in, 50ms fade-out     │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│ Spectral gating noise reduction      │  ← Replaces disabled neural denoiser
│   • noisereduce (stationary mode)    │
│   • prop_decrease=0.6, conservative  │
└──────────────────────────────────────┘
  │
  ▼
Upsample → 44.1 kHz (torchaudio)
  │
  ▼
┌──────────────────────────────────────┐
│ Phase B: master_vocals() (44.1 kHz)  │  ← Kokoro-specific mastering EQ
│   1. Highpass 80 Hz                  │
│   2. LowShelf +2 dB @ 200 Hz        │
│   3. Surgical EQ (mud, presence)     │
│   4. De-Esser (7 kHz ±4 dB boost→   │
│      compress→cut)                   │
│   5. Lowpass 9.5 kHz (Nyquist mask) │
│   6. Limiter -0.5 dB                │
└──────────────────────────────────────┘
  │
  ▼
Voice FX Chain (44.1 kHz)
  • NoiseGate (-42 dB) → Highpass 90 Hz
  • Compressor (2.5:1, -18 dB)
  • HighShelf -5.5 dB @ 6.5 kHz
  • Lowpass 8 kHz → Reverb → Limiter
  │
  ▼
Music FX Chain (44.1 kHz)
  • PeakFilter +2 dB @ 300 Hz (warmth)
  • PeakFilter -5 dB @ 1500 Hz (vocal pocket)
  • PeakFilter +0.8 dB @ 5500 Hz (air)
  • HighShelfFilter -3 dB @ 8 kHz
  • Limiter
  │
  ▼
Mixer (44.1 kHz)
  • Align voice + music (2s pre-roll)
  • Mask-based ducking (-21 dB, 400ms lookahead, 900ms attack, 2500ms release)
  • Linear fades (3s in / 5s out)
  • LUFS normalization (-14 LUFS unified target)
  • Master: HPF 35Hz → Gain(-3dB) → Bus Compressor (2:1) → Limiter (-1.0 dB)
  │
  ▼
Export: 44.1 kHz / 16-bit PCM WAV
```

### Why This Order?

- **Spectral gating** runs on dry audio at 24 kHz before any FX — the noise profile is most stable here
- **Phase B mastering** runs at *44.1 kHz* so EQ filters (especially the 7 kHz de-esser) operate well below Nyquist for stable, predictable curves
- The voice FX chain (reverb, compression) follows Phase B — reverb is applied to the mastered signal
- Neural denoising (resemble-enhance) is **disabled** on Apple Silicon due to instability; spectral gating is the active replacement

---

## 4. Phase B Detail: The Mastering Chain

### 4.1 Highpass Filter (80 Hz)
Removes DC offset and sub-bass rumble from TTS output. Prevents low-frequency energy from muddying the mix.

### 4.2 Warmth Boost (+2 dB @ 200 Hz Low-Shelf)
Simulates the "Proximity Effect" of a close-mic studio recording. Applied **only once** in the pipeline (consolidated from previous duplicate boosts).

### 4.3 De-Esser (6–8 kHz)
Uses the professional "boost → compress → cut" technique:
```
PeakFilter(7 kHz, +4 dB, Q=2.0)   →  Boost sibilance band
Compressor(-20 dB, 3:1, 1ms/50ms) →  Compress when sibilance is loud
PeakFilter(7 kHz, -4 dB, Q=2.0)   →  Cut the boost back to neutral
```
Reduced from ±6 dB to ±4 dB to avoid triggering on every sibilant. Only affects the sibilance band when it exceeds the threshold.

### 4.4 Lowpass Filter (9.5 kHz)
Psychoacoustic "super-resolution" smoothing: Kokoro TTS runs at 24 kHz native (Nyquist: 12 kHz), upsampled to 44.1 kHz. The 9.5 kHz lowpass creates a smooth, analog-sounding rolloff in the 10.5-12 kHz region, masking the abrupt digital brick-wall at the 12 kHz Nyquist boundary. Sibilance (7 kHz) is already handled by the de-esser above.

### 4.5 Limiter (-0.5 dB)
Final brick-wall limiter prevents any overs in the exported file.

---

## 5. Implementation Files

| File | Role |
|---|---|
| `core/kokoro_tts/postprocessor.py` | `KokoroMasteringEngine` — per-chunk cleanup, crossfade assembly, spectral gating noise reduction, voice FX chain, master EQ chain |
| `core/parler_tts/postprocessor.py` | `MasteringEngine` — Phase A (`restore_vocals`) and Phase B (`master_vocals`) for Parler TTS |
| `core/audio_processor.py` | Voice FX chain (compression + reverb + limiter), Music FX chains (MusicGen, ACE-Step, Lyria), Master chain (HPF 35Hz → Gain → Compressor → Limiter) |
| `core/mixer.py` | Mask-based ducking, overlay, equal-power crossfade looping, fades, LUFS normalization, resampling, export |
| `core/kokoro_tts/preprocessor.py` | Script parsing, text expansion, IPA phoneme injection, prosody punctuation, token-aware chunking |
| `core/kokoro_tts/engine.py` | TTS synthesis with per-chunk artifact trimming, inter-sentence pausing, spectral gating |
| `core/qa_monitor.py` | Quality assurance: clipping, LUFS, silence gaps, spectral balance, silence ratio |
| `core/pipeline.py` | Orchestrates the full signal chain end-to-end |

---

## 6. Key Implementation: `MasteringEngine`

```python
class MasteringEngine:
    def restore_vocals(self, audio, sr=24000):
        """Phase A: Neural denoising via resemble-enhance.
        Run on dry audio BEFORE reverb."""

    def master_vocals(self, audio, sr=44100):
        """Phase B: Highpass → Warmth → De-Ess → Lowpass → Limiter.
        Run AFTER mix and resample to 44.1 kHz."""
```

---

## 7. Pipeline Signal Chain Order

```python
# In core/pipeline.py — MeditationPipeline.generate()
voice_audio = tts.synthesize(...)                           # Step 3:  TTS @ 24 kHz (includes spectral gating)
# restore_vocals() bypassed — Kokoro output is pre-clean    # Step 3.5: Neural denoise (disabled)
voice_audio = resample_to_44100(voice_audio, 24000)         # Step 3.9: Upsample to 44.1 kHz
voice_audio = mastering_engine.master_vocals(voice_audio)   # Phase B:  EQ / de-ess / limit @ 44.1 kHz
voice_audio = apply_fx(voice_audio, voice_chain)            # Step 7:   Reverb + dynamics
music_audio = apply_fx(music_audio, music_chain)            # Step 8:   Music EQ
mixed = mix(voice_audio, activity, music_audio)             # Step 9:   Duck + fade + LUFS
# Master chain applied per-chunk in export_audio()          # Step 10:  HPF + Compressor + Limiter
export_audio(mixed, mix_sr, "wav", master_chain=chain)      # Step 11:  Chunked export @ 44.1kHz
```

---

## 8. Deployment Notes

1. **Memory:** Use `torch.bfloat16` for Parler-TTS 2.2B to leave VRAM for resemble-enhance
2. **Sentence Chunking:** For scripts > 2 minutes, generate in sentence chunks to prevent TTS hallucination
3. **Music Choice:** For sleep journeys, prioritize low-transient music (pads/drones) for smoother ducking

---

## 9. Verification Checklist

- [x] Spectral gating noise reduction active (noisereduce, stationary mode)
- [x] Output audio has no audible hiss above 9.5 kHz (LowpassFilter masks Nyquist)
- [x] Sub-bass rumble below 80 Hz is removed from voice (HighpassFilter 80Hz)
- [x] Subsonic energy below 35 Hz removed from master (HighpassFilter 35Hz)
- [x] Sibilance (6-8 kHz) is attenuated via de-esser (±4 dB boost→compress→cut @ 7 kHz)
- [x] Only one warmth boost (+2 dB @ 200 Hz) — no duplicate EQ
- [x] EQ and filtering operate at 44.1 kHz (not 24 kHz near-Nyquist)
- [x] Background music ducks smoothly via voice-activity mask (400ms lookahead, 900ms attack, 2500ms release)
- [x] TTS chunks cleaned by artifact trimmer (trailing silence + spectral flatness)
- [x] Master bus compressor (2:1, 30ms/300ms) smooths peaks before final limiter
- [x] Unified LUFS target: **-14 LUFS** (streaming distribution standard)
- [x] Numbers and abbreviations expanded to words before TTS inference
- [x] IPA phoneme injection for 30+ Sanskrit/yoga terms
- [x] Prosody punctuation enhancement for natural meditation phrasing
- [x] Token-aware chunking (100-150 target, 400 ceiling) prevents rushed speech
- [x] QA checks: clipping, LUFS, silence gaps, spectral balance, silence ratio
- [x] Equal-power cosine crossfades used for all segment stitching and music looping
- [x] Final file exported at **44.1 kHz / 16-bit PCM**