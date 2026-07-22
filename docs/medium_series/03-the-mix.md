# Article 3 — The Mix: Making a Voice and Music Breathe Together

**Thesis (one line):** The hardest problem in guided-meditation audio isn't generating
the voice or the music — it's sitting the narration *on* the music so the bed breathes
around the words instead of fighting them.

**Target reader takeaway:** A production-grade approach to voice-over-music: predictive
sidechain ducking computed offline from phrase timestamps, per-session loudness
calibration from measured LUFS, correct loudness staging, and why a well-known "plugin
limiter" was the villain of the whole mix.

**Suggested length:** 2,800–3,500 words. This is the flagship piece.

---

## Narrative beats (in order)

1. **Hook — three bug reports, one root cause.** Users reported the mix sounded
   *static/harsh*, the ducking felt *flat/weak*, and the bed was *too loud*. Commit
   `8a50df0` fixed all three at once. The static wasn't in the generators — it was the
   **limiter**.

2. **The villain: pedalboard 0.9.23's `Limiter`.** The forensics: it inflates
   sub-threshold signal by ~+4.75 dB and adds broadband distortion ("static"). It was
   ripped out of *every* chain — both music chains and the master chain — and replaced
   with a custom true-peak limiter that only acts at export, after loudness
   normalization. Great "measure, don't trust the plugin" story.

3. **The custom true-peak limiter.** Why true-peak matters (inter-sample peaks exceed
   sample peaks after D/A reconstruction), how it's built: 4× oversample → per-sample
   target gain that keeps the ceiling → look-ahead via a `minimum_filter1d` → zero-phase
   smoothed release *clamped to the target so the ceiling never breaks* → downsample.
   Fully vectorized, no per-sample Python loop. Transparent below threshold — the whole
   point.

4. **The centerpiece: the "breathing" duck.** Contrast two philosophies:
   - **Reactive ducker (the old way):** an envelope follower listens to the voice and
     pulls the music down *after* it hears sound. Always late, always pumps.
   - **Predictive/"breathing" duck (the new way):** detect the voice's phrases *offline*
     from the VAD mask, then *pre-compute* the entire gain envelope: the bed starts
     descending ~600 ms *before* each phrase (professional broadcast behavior), holds
     low during speech, releases over ~1.5 s, and *lifts* slightly during pauses ≥1.5 s
     so the bed audibly breathes. A tiny reactive follower is kept only as a safety net
     for off-script breaths, combined so the script wins wherever it lifts.

5. **Getting the *level* right automatically: adaptive bed calibration.** The insight
   that fixed the "too loud/too quiet depending on the upload" problem: don't trust
   fixed dB constants — *measure* the post-FX stems' short-term LUFS and solve for the
   bed gain so the music sits a fixed loudness-units distance under the voice. Calibrated
   so a nominal session reproduces the old (−16, −16), and only off-nominal material gets
   corrected. Ships ON by default (`MOODSCAPE_ADAPTIVE_BED`).

6. **Loudness staging — the numbers that matter.** The staged targets and *why order
   matters*: voice normalized to −18 LUFS before mixing; the final export does master
   EQ/glue → LUFS-normalize to −16 → true-peak limit to −1 dBTP. Normalize *first*, then
   limit — limiting first then normalizing would re-exceed the ceiling. −16 LUFS / −1
   dBTP matches Apple Music and avoids platform re-limiting.

7. **The master chain philosophy: glue, not loudness.** No limiter here on purpose. Just
   HPF 30 Hz → a gentle 1.5:1 "glue" comp at −12 dB (higher threshold than the old −22
   dB so it engages less and never pumps) → +1 dB air shelf at 12 kHz.

8. **Two beds, two chains — and the music-source pivot.** Lyria output is a raw
   generative stream (needs mud/presence taming); an uploaded instrumental is already a
   finished production (touch it as little as possible — just protect it and carve a
   speech pocket). Tie in stem separation being *skipped* for uploads. Optional: the
   fade curve (exponential, gentle steepness 2.0 rising so it doesn't read as "silence
   then music") and the two research experiments — spectral/mid-band ducking and a
   shared convolution reverb send so voice and bed share one room.

9. **Close — mixing is where meditation audio is won.** The generators get you 60%;
   the last 40% is entirely in the mix.

---

## Code snippets to include

### 1. The forensics: the limiter that was the bug (`core/audio_processor.py:137`)

```python
def make_master_chain() -> Pedalboard:
    """Final mastering EQ + glue (NO limiter — see below)."""
    # Peak control is deliberately NOT done here. pedalboard 0.9.23's Limiter
    # inflates sub-threshold signals by ~+4.75 dB and generates broadband
    # distortion. True-peak limiting to -1 dBTP is applied cleanly, after LUFS
    # normalization, by mixer.true_peak_limit inside export_audio.
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        Compressor(threshold_db=-12.0, ratio=1.5, attack_ms=50.0, release_ms=200.0),
        HighShelfFilter(cutoff_frequency_hz=12000.0, gain_db=1.0),
    ])
```

### 2. The predictive gain envelope — the "breathing" (`core/mixer.py:513`, `:418`)

```python
def compute_breathing_gain_db(voice_audio, sample_rate, duck_depth_db=-15.0,
                              pre_descent_ms=600.0, attack_ramp_ms=700.0,
                              release_ms=1500.0, lift_db=1.5, lift_pause_s=1.5, ...):
    """Combined breathing-duck gain envelope (dB). Phrases detected from the voice
    via RMS-envelope VAD; the bed falls with a predictive S-curve BEFORE each
    phrase, sits at duck_depth_db during speech, and lifts in long pauses."""
    phrases  = detect_phrases(voice_audio, sample_rate, ...)
    g_script = _script_gain_db(n, sample_rate, phrases, pre_descent_ms=..., duck_db=duck_depth_db, ...)
    g_react  = _reactive_gain_db(voice_audio, sample_rate, range_db=duck_depth_db)  # safety net
    return combine_script_with_reactive(g_script, g_react)  # script wins where it lifts
```

Inside `_script_gain_db`, the pre-emptive descent — this is the "starts before you
speak" behavior, using a smoothstep S-curve:

```python
# Per-phrase predictive descent → hold → release.
desc_start = int(round((t_on - pre_descent_ms / 1000.0) * sample_rate))   # ~600 ms EARLY
g_db[seg_start:seg_end] = g0 + (duck_db - g0) * _smoothstep(t)            # S-curve down
# ... hold at duck_db during [t_on, t_off], then release over ~1.5 s toward a
#     pause "lift" (+1.5 dB) if the following gap is long enough to breathe.
```

### 3. Calibrating the bed level from *measured* loudness (`core/mixer.py:350`)

```python
def calibrate_music_bed(voice_audio, music_audio, sample_rate, ...,
                        speech_offset_lu=30.5, pause_offset_lu=14.5, ...):
    """Measure post-FX stems, return (music_volume_db, duck_amount_db). The bed
    gain is set so the music's short-term LUFS sits pause_offset_lu below the
    voice's speech-region LUFS; the duck covers the remaining separation.
    Nominal material calibrates back to the legacy (-16, -16); only a hot or
    whisper-quiet upload gets corrected."""
    voice_lufs = float(np.median(l_v[speech_mask & finite_v]))   # voice loudness while speaking
    music_lufs = float(np.median(l_m[finite_m]))                 # bed loudness
    music_volume_db = float(np.clip((voice_lufs - pause_offset_lu) - music_lufs, -24.0, -8.0))
    duck_amount_db  = float(np.clip(-(speech_offset_lu - pause_offset_lu), -20.0, -10.0))
    return music_volume_db, duck_amount_db
```

### 4. Loudness staging — normalize, *then* true-peak limit (`core/mixer.py:938`, `:689`)

```python
# export_audio: order is critical.
mix_lufs_gain = calculate_loudness_gain(mastered_audio, sample_rate, target_lufs)  # → -16 LUFS
mastered_audio = (mastered_audio * mix_lufs_gain).astype(np.float32)
mastered_audio = true_peak_limit(mastered_audio, sample_rate, threshold_db=-1.0)   # → -1 dBTP
# Limiting first then normalizing would re-exceed the ceiling.
```

```python
def true_peak_limit(audio, sample_rate, threshold_db=-1.0, oversample=4, ...):
    """Oversampled true-peak brickwall limiter (replaces pedalboard Limiter).
    Transparent below threshold, unlike pedalboard 0.9.23's Limiter (~+4.75 dB)."""
    up = resample_poly(x, oversample, 1, axis=-1)                 # 4x oversample
    detector = np.max(np.abs(up), axis=0)
    target = np.minimum(1.0, thr / np.maximum(detector, 1e-12))   # per-sample ceiling gain
    target = minimum_filter1d(target, size=la, origin=-(la // 2)) # look-ahead
    gain = np.minimum(_zero_phase_smooth(target, up_sr, smooth_hz), target)  # release, clamped
    up *= gain[np.newaxis, :]
    return resample_poly(up, 1, oversample, axis=-1)              # back down
```

### 5. Two beds, two philosophies (`core/audio_processor.py:46` vs `:86`)

```python
# Lyria: a raw generative stream — tame mud + presence, roll off extended highs.
def make_lyria_music_chain():
    return Pedalboard([HighpassFilter(60.0), PeakFilter(250, -1.5, 0.8),
                       PeakFilter(4500, -2.0, 0.7), HighShelfFilter(9000.0, -2.5),
                       Compressor(-18.0, 2.0, 80.0, 500.0)])          # no Limiter

# Upload: already a finished production — barely touch it, just carve a speech pocket.
def make_upload_music_chain():
    return Pedalboard([HighpassFilter(30.0), PeakFilter(2000, -2.0, 0.7),
                       LowpassFilter(14000.0)])                       # no Limiter
```

---

## Learnings box

> - **Measure your plugins.** The "static" was a stock `Limiter` inflating quiet signal
>   ~+4.75 dB. A null test against dry signal would have caught it immediately.
> - **Duck predictively, offline.** Compute the whole gain envelope from phrase
>   timestamps and start the descent ~600 ms *before* speech. Reactive envelope
>   followers are always late and pump.
> - **Let the bed breathe.** Lift the music slightly (+1.5 dB) during pauses ≥1.5 s.
>   The absence of movement is what makes cheap voice-overs sound flat.
> - **Calibrate level from measured LUFS, not fixed dB.** One constant can't serve a hot
>   upload and a whisper-quiet one; solve for the gain from the actual loudness.
> - **Order of operations in mastering is not optional.** Normalize → then true-peak
>   limit. Ship at −16 LUFS / −1 dBTP to match streaming platforms.
> - **Match the chain to the source.** A generative stream and a finished instrumental
>   need opposite amounts of processing.

## Source appendix
Commits: `8a50df0` (mixing rewrite), `9bc4a5d` + `4efd6ed` (adaptive calibration),
`cd612ad` (remove reactive ducker/Stitch), `611ae59` + `a0d5ad3` (upload sources).
Code: `core/mixer.py` (`compute_breathing_gain_db`, `_script_gain_db`, `apply_breathing_duck`,
`calibrate_music_bed`, `true_peak_limit`, `mix`, `export_audio`, `apply_fades`),
`core/audio_processor.py` (`make_master_chain`, `make_lyria_music_chain`, `make_upload_music_chain`, `make_vocal_pocket_chain`).
Docs: `docs/optimization_and_processing/audio_processing.md`, `docs/GOTCHAS.md` (Mixing & Mastering), `CLAUDE.md` (Pipeline Flow steps 8–12).
