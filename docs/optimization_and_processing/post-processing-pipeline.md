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
│ Phase A: restore_vocals()            │  ← Dry audio, before reverb
│   • resemble-enhance neural denoise  │
│   • run_denoise=True (MPS)           │
└──────────────────────────────────────┘
  │
  ▼
Voice FX Chain (24 kHz)
  • Compressor (dynamics)
  • Reverb (spatial)
  • Limiter
  │
  ▼
Music FX Chain (24 kHz)
  • PeakFilter +2 dB @ 300 Hz (warmth)
  • PeakFilter -3 dB @ 1800 Hz (vocal pocket)
  • HighShelfFilter -4 dB @ 10 kHz (gentle HF rolloff)
  • Limiter
  │
  ▼
Mixer (24 kHz)
  • Align voice + music (2s pre-roll)
  • Mask-based ducking (-9 dB, 100ms lookahead, 150ms attack, 2000ms release)
  • Linear fades (3s in / 5s out)
  • LUFS normalization (-14 LUFS unified target)
  • Master: HPF 35Hz → Gain → Bus Compressor (2:1) → Limiter (-0.1 dB)
  │
  ▼
Resample → 44.1 kHz (torchaudio)
  │
  ▼
┌──────────────────────────────────────┐
│ Phase B: master_vocals()  (44.1 kHz) │  ← After mix, at final SR
│   1. Highpass 80 Hz                  │
│   2. LowShelf +2 dB @ 200 Hz        │
│   3. De-Esser (7 kHz boost→comp→cut)│
│   4. Lowpass 15 kHz                  │
│   5. Limiter -0.5 dB                │
└──────────────────────────────────────┘
  │
  ▼
Export: 44.1 kHz / 16-bit PCM WAV
```

### Why Two Phases?

- **Phase A** runs on *dry* audio so the neural denoiser sees clean signal without reverb tails
- **Phase B** runs at *44.1 kHz* so EQ filters (especially the 7 kHz de-esser) operate well below Nyquist for stable, predictable curves
- The voice FX chain (reverb) sits between the two phases — reverb is applied to denoised audio, then the mastered mix is EQ'd at the final sample rate

---

## 4. Phase B Detail: The Mastering Chain

### 4.1 Highpass Filter (80 Hz)
Removes DC offset and sub-bass rumble from TTS output. Prevents low-frequency energy from muddying the mix.

### 4.2 Warmth Boost (+2 dB @ 200 Hz Low-Shelf)
Simulates the "Proximity Effect" of a close-mic studio recording. Applied **only once** in the pipeline (consolidated from previous duplicate boosts).

### 4.3 De-Esser (6–8 kHz)
Uses the professional "boost → compress → cut" technique:
```
PeakFilter(7 kHz, +6 dB, Q=2.0)   →  Boost sibilance band
Compressor(-20 dB, 4:1, 1ms/30ms) →  Compress when sibilance is loud
PeakFilter(7 kHz, -6 dB, Q=2.0)   →  Cut the boost back to neutral
```
This only affects the sibilance band when it exceeds the threshold, leaving the rest of the spectrum untouched.

### 4.4 Lowpass Filter (15 kHz)
Removes digital hiss and aliasing artifacts above 15 kHz. Set to 15 kHz (not 12 kHz) to preserve vocal "air" and clarity.

### 4.5 Limiter (-0.5 dB)
Final brick-wall limiter prevents any overs in the exported file.

---

## 5. Implementation Files

| File | Role |
|---|---|
| `core/post_processor.py` | `MasteringEngine` — Phase A (`restore_vocals`) and Phase B (`master_vocals`) |
| `core/audio_processor.py` | Voice FX chain (compression + reverb + limiter), Music FX chain (HighShelfFilter), Master chain (HPF 35Hz → Gain → Compressor → Limiter) |
| `core/mixer.py` | Mask-based ducking, overlay, equal-power crossfade looping, fades, LUFS normalization, resampling, export |
| `core/text_preprocessor.py` | Text normalization (number/abbreviation expansion for TTS) |
| `core/kokoro_engine.py` | TTS synthesis with per-chunk artifact trimmer (silence + spectral flatness detection) |
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
voice_audio = tts.synthesize(...)                          # Step 3:  TTS @ 24 kHz
voice_audio = mastering_engine.restore_vocals(voice_audio)  # Step 3.5: Phase A denoise
voice_audio = apply_fx(voice_audio, voice_chain)            # Step 7:  Reverb + dynamics
music_audio = apply_fx(music_audio, music_chain)            # Step 8:  Music EQ
mixed = mix(voice_audio, activity, music_audio)             # Step 9:  Duck + fade + LUFS
mixed = apply_fx(mixed, master_chain)                       # Step 10: Master limiter
mixed_44k = resample_for_export(mixed, 24000, 44100)       # Step 11: Upsample
mixed_44k = mastering_engine.master_vocals(mixed_44k)       # Step 11: Phase B EQ
export_audio(mixed_44k, 44100, "wav")                       # Step 12: 44.1kHz/16-bit
```

---

## 8. Deployment Notes

1. **Memory:** Use `torch.bfloat16` for Parler-TTS 2.2B to leave VRAM for resemble-enhance
2. **Sentence Chunking:** For scripts > 2 minutes, generate in sentence chunks to prevent TTS hallucination
3. **Music Choice:** For sleep journeys, prioritize low-transient music (pads/drones) for smoother ducking

---

## 9. Verification Checklist

- [x] Output audio has no audible hiss above 15 kHz
- [x] Sub-bass rumble below 80 Hz is removed from voice (HighpassFilter 80Hz)
- [x] Subsonic energy below 35 Hz removed from master (HighpassFilter 35Hz)
- [x] Sibilance (6–8 kHz) is attenuated without affecting other frequencies
- [x] Only one warmth boost (+2 dB @ 200 Hz) — no duplicate EQ
- [x] De-esser uses frequency-targeted boost→compress→cut (not full-band compression)
- [x] EQ and filtering operate at 44.1 kHz (not 24 kHz near-Nyquist)
- [x] Background music ducks smoothly via voice-activity mask (100ms lookahead, 150ms attack, 2000ms release)
- [x] TTS chunks are cleaned by artifact trimmer (trailing silence + spectral flatness)
- [x] Master bus compressor (2:1, 30ms/300ms) smooths peaks before final limiter
- [x] Unified LUFS target: **−14 LUFS** (streaming distribution standard)
- [x] Numbers and abbreviations expanded to words before TTS inference
- [x] MusicGen spectral flux guard rejects segments with percussive transients
- [x] Equal-power cosine crossfades used for all segment stitching and music looping
- [x] Final file exported at **44.1 kHz / 16-bit PCM**