# Article 2 — Making a Synthetic Voice Sound Human

**Thesis (one line):** Modern neural TTS is *too* clean — perfectly denoised, perfectly
steady — and that perfection reads as robotic. Realism is deliberately engineered back
in.

**Target reader takeaway:** Concrete DSP techniques (micro-pitch modeling, formant
warmth, phrase-final declination, close-mic EQ, careful de-essing) to move a synthetic
voice from "AI narrator" to "someone breathing in the room with you" — plus how to run
these as reversible experiments.

**Suggested length:** 2,200–2,800 words.

---

## Narrative beats (in order)

1. **Hook — the over-denoising walk-back.** The counter-intuitive opening: I made the
   voice *worse* by cleaning it too well. KokoroV2 (`400f65c`) explicitly *reduced*
   noise reduction because aggressive spectral gating scrubbed out breath and
   naturalness, leaving a sterile "AI voice". The first lesson of realism is knowing
   when to stop cleaning.

2. **Three layers of pitch variation humans have and TTS lacks.** The centerpiece
   technique: real speech carries slow drift (vocal-fold tension), subtle vibrato on
   sustained vowels, and random micro-jitter (neural noise). Model all three, keep the
   combined modulation under ±15 cents so it never "trembles", and add formant warmth
   by lowering formants ~3% (a slightly larger simulated vocal tract). All in one
   pyworld analysis/resynthesis pass.

3. **Close-mic intimacy is an EQ decision.** Walk through the Kokoro voice chain as a
   signal flow: gate → HPF → mud cut → proximity warmth → whisper-catching compression
   → air shelf → plate reverb. Emphasize the reversal in beat: an earlier version
   *cut* 7.5 kHz and low-passed at 9.5 kHz and sounded "dull and muffled"; the fix was
   a small +1 dB air shelf at 10 kHz instead.

4. **F5 is a different animal — cloning artifacts, not synthesis hiss.** F5 clones a
   reference clip and has *no duration predictor*, which creates two very specific
   problems worth a section each:
   - **The stray-syllable leak.** On a short line ("Breathe in…"), F5 pads the required
     mel length with leftover *reference audio* — leaking a random syllable. The fix is
     delightfully physical: end the reference with ~1 s of −55 dBFS noise so it leaks
     *silence* instead. (`MOODSCAPE_F5_REF_PAD`, ships ON.)
   - **The mastering retune** (`297f62a`): two-stage de-esser, "Abbey Road" parallel
     reverb (reverb on a HPF'd parallel path, not in series), tape saturation ~15% wet.

5. **Research as reversible experiments.** How the risky realism ideas (microprosody,
   `cfg` "expressiveness", preserve-reference-dynamics) shipped **OFF** behind env
   flags with Gradio A/B toggles, while the two safe fixes shipped **ON** as
   kill-switches. This is the discipline that let aggressive DSP live in the codebase
   without destabilizing the golden path.

6. **Close — realism is a subtraction *and* an addition.** You remove the over-cleaning,
   then add back the imperfections that signal "human". End on microprosody: gliding
   the phrase-final pitch down by ~120 cents is literally the relaxation cue human
   meditation guides use.

---

## Code snippets to include

### 1. Three-layer pitch humanization + formant warmth (`core/kokoro_tts/postprocessor.py:455`)

```python
def humanize_voice(audio, sr=24000, drift_hz=0.5, drift_cents=6.0,
                   vibrato_hz=5.0, vibrato_cents=3.0, jitter_cents=2.0,
                   formant_shift=0.97):
    """Natural speech has three pitch layers TTS lacks: slow drift (~0.5 Hz),
    subtle vibrato (~5 Hz), random micro-jitter. Combined stays < ±15 cents so
    it never trembles. Formants lowered 3% for warmth. One pyworld pass."""
    f0, t = pw.harvest(audio_f64, sr)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)
    ap = pw.d4c(audio_f64, f0, t, sr)

    drift   = drift_cents  * np.sin(2*np.pi*drift_hz  * t_frames)     # vocal-fold tension
    vibrato = vibrato_cents * np.sin(vib_phase)                       # sustained-vowel vibrato
    jitter  = gaussian_filter1d(np.random.randn(n) * jitter_cents, 3) # neural micro-noise

    total_cents = (drift + vibrato + jitter) * voiced
    f0_mod = np.where(voiced, f0 * 2 ** (total_cents / 1200.0), f0)
    # ... warp spectral envelope by formant_shift, then pw.synthesize(...)
```

### 2. The stray-syllable fix — pad the reference with quiet noise (`core/f5_tts/engine.py:153`)

```python
# F5-TTS has no duration predictor, so on short generations ("Breathe in…") it
# pads the required mel length with leftover reference audio — leaking a stray
# syllable. Ending the reference with ~1 s of quiet noise makes it leak *silence*.
# -55 dBFS: above F5's internal -42 dBFS edge-trimmer so it survives preprocessing,
# yet well below the speech.
if os.environ.get("MOODSCAPE_F5_REF_PAD", "1") == "1":
    n_pad = int(pad_sec * file_sr)                 # ~1.0 s
    pad_rms = 10 ** (pad_dbfs / 20.0)              # -55 dBFS
    tail = np.random.randn(n_pad).astype(np.float32) * pad_rms
    audio = np.concatenate([audio.astype(np.float32), tail])
```

### 3. The close-mic voice chain as documented signal flow (`core/kokoro_tts/postprocessor.py:554`)

Quote the docstring's numbered chain (it reads like a mastering engineer's notes):

```
1. NoiseGate -40 dB   — gates synthesis artifacts, spares breathy /h/, /f/
2. HPF 80 Hz          — sub-bass rumble + plosive energy
3. PeakFilter -2.5 dB @ 400 Hz  — ISTFTNet boxy low-mid resonance
4. LowShelf +2.5 dB @ 180 Hz    — close-mic proximity warmth (chest/body)
5. Compressor 2.5:1 @ -28 dB    — catches whisper-level meditation delivery
6. HighShelf +1.0 dB @ 10 kHz   — "air"/intimacy (replaced a dull -3.5 dB cut)
7. Convolution reverb 18% wet   — plate IR, the studio standard for intimate voice
```

### 4. (Optional) phrase-final declination — the relaxation cue (`core/f5_tts/postprocessor.py:161`)

```python
def apply_microprosody(audio, sr=24000, decline_cents=120.0, tail_ms=600.0,
                       ap_scale=1.05, pitch_scale=1.15, formant_shift=0.98, ...):
    """Widen pitch about the mean (expressive contour), glide the final 600 ms
    of f0 down by 120 cents (the relaxation cue human guides use), lift
    aperiodicity for breathiness. OFF by default — WORLD resynthesis can colour
    Vocos's already-clean output, so it's an opt-in A/B experiment."""
```

---

## Learnings box

> - **Stop denoising sooner than you think.** Full-strength spectral gating / DeepFilter
>   removed the breath that signals "human". KokoroV2 dialed it back; IndexTTS blended
>   DeepFilter wet rather than applying it 100%.
> - **Model imperfection explicitly.** Drift + vibrato + jitter, capped < ±15 cents.
>   Beyond that it trembles; below it, it's dead.
> - **Air, not scoop.** A +1 dB shelf at 10 kHz beat cutting the highs — the earlier
>   dull/muffled voice came from over-attenuating presence.
> - **Know your engine's failure mode.** F5's missing duration predictor is a *physical*
>   bug you fix in the *reference*, not the model.
> - **Ship risky DSP as reversible flags.** Two safe fixes ON as kill-switches; the rest
>   OFF behind Gradio A/B toggles. See `CLAUDE.md` → "Research Experiment Flags".

## Source appendix
Commits: `400f65c` (KokoroV2), `297f62a` (F5 mastering retune), `769469d` (DeepFilter blend), research pass (flags).
Code: `core/kokoro_tts/postprocessor.py` (`humanize_voice`, `build_voice_chain`),
`core/f5_tts/engine.py` (`_condition_reference_audio`), `core/f5_tts/postprocessor.py` (`apply_microprosody`).
Docs: `docs/optimization_and_processing/post-processing-pipeline.md`, `docs/prompting_guides/vocal_*_instructions.md`, `CLAUDE.md`.
