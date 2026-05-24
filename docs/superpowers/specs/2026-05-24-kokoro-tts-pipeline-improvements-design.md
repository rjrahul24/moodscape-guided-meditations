# Kokoro TTS Pipeline Improvements — Design Spec
**Date:** 2026-05-24  
**Approach:** A — Precision Refinement  
**Status:** Approved

---

## 1. Context & Motivation

This spec captures the findings from a full audit of the Kokoro TTS pipeline against the *Kokoro TTS Meditation Audio Blueprint* research document. The goal is studio-grade, emotion-rich, expressive audio suited for guided meditations.

The existing implementation is strong: SLERP voice blending, per-sentence speed variation, IPA injection for 30+ Sanskrit terms, spectral gating, cosine² crossfade assembly, and a full Pedalboard voice FX chain are all in place. This spec targets six specific gaps identified in the audit.

---

## 2. Scope

**In scope:**
- Fix pyworld `humanize_voice` application scope (engine.py)
- Add global G2P lexicon overrides for core English meditation words (engine.py)
- Add stress reduction/boost markers to preprocessor (preprocessor.py)
- Add `pure_calm` voice preset with negative-weight extrapolation (voice_manager.py, app.py)
- Conservative FX chain tweaks: compressor -28 dB, reverb 18% (postprocessor.py)
- Documentation reconciliation (kokoro_tts.md, vocal_kokoro_instructions.md, CLAUDE.md, post-processing-pipeline.md)

**Out of scope:**
- Multi-variant generation (low priority, pipeline complexity)
- Target duration matching algorithm
- Room tone WAV replacement (synthetic noise adequate)
- FX chain structural changes (LPF vs air boost debate settled by existing tuning)
- Any changes to F5-TTS, Chatterbox, ACE-Step, HeartMuLa, or Lyria pipelines

---

## 3. Design

### 3.1 pyworld Humanization — Scope Fix

**Current behaviour:** `humanize_voice()` is called on the full concatenated audio array at the end of `synthesize()`, which includes room-tone pause segments. This wastes compute and risks artifacts at speech→silence boundaries where WORLD's pitch tracker encounters unvoiced signal snaps.

**Target behaviour:** `humanize_voice()` is called per speech chunk, immediately after `apply_segment_fades()`, before appending to `audio_chunks`. Pause and room-tone chunks are never passed through pyworld.

**File:** `core/kokoro_tts/engine.py`

```
# Inside the per-sentence synthesis loop:
speech_audio = crossfade_chunks(speech_parts)
speech_audio = apply_segment_fades(speech_audio)
speech_audio = humanize_voice(speech_audio, sr=SAMPLE_RATE)  # ← moved here
audio_chunks.append(speech_audio)

# End of synthesize() — remove the old call:
# voice_audio = humanize_voice(voice_audio, sr=SAMPLE_RATE)  ← DELETE
```

**pyworld parameters (unchanged):**
- `drift_hz=0.5`, `drift_cents=6.0` — slow pitch drift (vocal fold tension simulation)
- `vibrato_hz=5.0`, `vibrato_cents=3.0` — subtle vibrato
- `jitter_cents=2.0` — random micro-jitter
- `formant_shift=0.97` — 3% lower formants for perceived warmth

**Research basis:** Small pyworld perturbations (±6–15 cents total) on ISTFTNet output are validated by academic work on prosodic parameter manipulation in TTS (arXiv 2409.12176). The key constraint is keeping total modulation under ±15 cents to avoid a trembling quality — the current parameters satisfy this.

---

### 3.2 Global G2P Lexicon Overrides

**Current behaviour:** Core English meditation verbs (`breathe`, `relax`, `release`, etc.) are synthesized with their standard conversational dictionary pronunciations — brisk, naturally-stressed delivery that lacks the elongated vowels and reduced stress of meditation speech.

**Target behaviour:** At `load_model()` time, write elongated IPA variants directly into `pipeline.g2p.lexicon.golds`. These take effect for every subsequent synthesis call without any text manipulation.

**File:** `core/kokoro_tts/engine.py`

New private method `_apply_meditation_lexicon(pipe)` called only after the **American English** `KPipeline` is created (`self.pipeline`, `lang_code='a'`). The British pipeline (`self.pipeline_en_gb`) is intentionally excluded — the specified IPA uses rhotic American phonemes (e.g., `bɹiːːð` with `ɹ`) that would produce incorrect non-rhotic pronunciation in British English.

**Lexicon overrides:**

| Word | Standard delivery issue | IPA override |
|------|------------------------|-------------|
| `breathe` | Short 'ee', brisk | `bɹiːːð` — hyper-elongated 'ee' |
| `breath` | Short vowel | `bɹɛːθ` — elongated open vowel |
| `exhale` | Standard stress | `ɛksˈheɪːl` — elongated 'ay' on release |
| `inhale` | Standard stress | `ɪnˈheɪːl` — elongated 'ay' on intake |
| `relax` | Short 'a', energetic | `ɹɪˈlæːks` — elongated 'a' |
| `release` | Short 'ee' | `ɹɪˈliːːs` — hyper-elongated 'ee' |
| `soften` | Standard 'o' | `ˈsɔːfən` — elongated 'o' for warmth |
| `surrender` | Primary stress on second syllable | `səˈɹɛndə` — reduced stress, trailing schwa |
| `dissolve` | Standard 'o' | `dɪˈzɒːlv` — elongated 'o' |
| `melt` | Short vowel | `mɛːlt` — elongated vowel |
| `sink` | Short vowel | `sɪːŋk` — elongated vowel |
| `drift` | Short vowel | `dɹɪːft` — elongated vowel |

**Failure mode:** If a future misaki version changes the `g2p.lexicon.golds` attribute path, the method catches `AttributeError` and logs a warning — synthesis continues normally with default pronunciations.

**Interaction with existing IPA injection:** These overrides operate at the G2P level (lower than text). The existing regex-based Sanskrit IPA injection in `preprocessor.py` operates at the text level. Both systems compose cleanly — Sanskrit terms are handled by text injection; English meditation words are handled by lexicon override.

---

### 3.3 Stress Reduction / Boost Markers

**Current behaviour:** No stress annotations are applied to any English words. All words carry their full dictionary stress, which can make tension-related words (`tension`, `stress`, `worry`) sound energetically charged and make affirmation words (`peace`, `calm`, `still`) sound flat.

**Target behaviour:** A new `_apply_stress_markers()` function in `preprocessor.py` wraps targeted words with misaki stress notation before the prosody punctuation step.

**File:** `core/kokoro_tts/preprocessor.py`

**Stress reduction targets** — `[word](-1)` flattens pitch contour, makes word sound "already released":
- `tension`, `tense`, `worry`, `worries`, `stress`, `fear`, `pain`, `hurry`, `thoughts`, `distraction`

**Stress boost targets** — `[word](+1)` adds gentle emphasis as affirmation:
- `peace`, `calm`, `still`, `stillness`, `safe`, `whole`, `free`, `light`, `gently`, `softly`

**Collision guard:** The regex uses `(?<!\[)` lookbehind to skip words already inside an IPA injection block `[word](/IPA/)`, preventing double-wrapping.

**Pipeline position:** Called as step 4b in `preprocess_for_meditation()`, between `inject_phonemes()` (step 3) and `enhance_prosody_punctuation()` (step 4). This ordering ensures IPA blocks are already in place before stress markers are applied, so the collision guard works correctly.

---

### 3.4 `pure_calm` Voice Preset

**Current behaviour:** All 6 presets use positive-weight SLERP blending. No preset removes tension characteristics — they blend existing voices but cannot subtract a specific tonal quality.

**Target behaviour:** A new `pure_calm` preset uses negative-weight extrapolation to subtract 5% of `af_bella`'s energetic characteristics from the blend, producing a voice quality that does not exist as any native model voice.

**File:** `core/kokoro_tts/voice_manager.py`

New function `blend_with_extrapolation(voice_weights)`:
- Supports negative weight values
- After blending, L2-renormalizes the result to match the primary voice's embedding norm, preventing amplitude drift from the subtraction operation
- If `current_norm < 1e-6` (degenerate case), returns result unchanged

New preset entry:
```python
"pure_calm": {
    "description": "Ultra-low tension — warmth with conversational energy subtracted",
    "blend": {
        "af_heart":  0.60,   # primary warmth and stability
        "af_sarah":  0.30,   # soft, natural breathiness
        "af_aoede":  0.10,   # musical prosody
        "af_bella": -0.05,   # subtract conversational tension/energy
    },
    "method": "extrapolation",
},
```

`get_voice()` updated to route presets with `"method": "extrapolation"` to `blend_with_extrapolation()` instead of `slerp_blend()`.

**File:** `app.py` — add `("Pure Calm — tension-free ultra-soft", "pure_calm")` to `KOKORO_VOICE_CHOICES`.

---

### 3.5 Conservative FX Chain Tweaks

**File:** `core/kokoro_tts/postprocessor.py`, `build_voice_chain()`

#### Compressor threshold: -22 dB → -28 dB

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `threshold_db` | -22 | -28 | Catches whisper-level meditation delivery that sits below -22 dB. 6 dB lower threshold at same 2.5:1 ratio ≈ 2–3 dB additional GR on soft phrases — transparent but consistent. |
| `ratio` | 2.5 | 2.5 | Unchanged — gentle enough to avoid pumping |
| `attack_ms` | 15 | 15 | Unchanged |
| `release_ms` | 150 | 150 | Unchanged |

#### Reverb wet level: 15% → 18%

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `reverb_amount` default | 0.15 | 0.18 | More enveloping acoustic space for deep relaxation and visualization scripts. 3% increase is perceptible but not "wet" — stays well below the 25% the research recommends (which risks muddiness on headphones). |

**What is NOT changed:** `HighShelfFilter(+1.0 dB @ 10 kHz)` stays. It was specifically chosen to replace the previous `-3.5 dB cut + 9.5 kHz LPF` that made voices dull. The research's LPF recommendation conflicts with this existing design decision; the existing decision is correct for this pipeline.

---

### 3.6 Documentation Reconciliation

All changes are documentation-only — no logic.

| File | Inconsistency | Fix |
|------|--------------|-----|
| `docs/model_implementation_guides/kokoro_tts.md §13` | Says `humanize_voice` is "DISABLED" | Replace with accurate description: active, per-speech-chunk scope, parameters listed |
| `docs/model_implementation_guides/kokoro_tts.md §10` | Stage 3 FX chain shows `-22 dB` compressor, `15%` reverb | Update to `-28 dB`, `18%` |
| `docs/prompting_guides/vocal_kokoro_instructions.md` | Line 89: `\n\n` → `[pause:2.5s]` | Fix to `[pause:6.5s]`; add note explaining 6.5s for spacious meditation pacing |
| `CLAUDE.md` pipeline flow | No Stage 2c in pipeline table | Add `2c: humanize_voice() per-speech-chunk` |
| `CLAUDE.md` FX Chain Summary table | Stale compressor/reverb values | Update `build_voice_chain()` row |
| `CLAUDE.md` Key Constants | No humanize_voice constants listed | Add row with drift/vibrato/jitter/formant values |
| `docs/optimization_and_processing/post-processing-pipeline.md` | Kokoro TTS Path diagram missing humanize step; stale FX values | Add humanize stage, update compressor/reverb |

---

## 4. File Change Summary

| File | Type | Change |
|------|------|--------|
| `core/kokoro_tts/engine.py` | Code | Move humanize_voice to per-chunk; add `_apply_meditation_lexicon()` |
| `core/kokoro_tts/preprocessor.py` | Code | Add `_apply_stress_markers()`, call in `preprocess_for_meditation()` |
| `core/kokoro_tts/voice_manager.py` | Code | Add `blend_with_extrapolation()`, `pure_calm` preset, update `get_voice()` |
| `core/kokoro_tts/postprocessor.py` | Code | Compressor -28 dB, reverb default 18% |
| `app.py` | Code | Add `pure_calm` to voice choices |
| `docs/model_implementation_guides/kokoro_tts.md` | Docs | Fix humanize_voice status, update FX values |
| `docs/prompting_guides/vocal_kokoro_instructions.md` | Docs | Fix paragraph pause duration |
| `CLAUDE.md` | Docs | Add Stage 2c, update FX table, add constants |
| `docs/optimization_and_processing/post-processing-pipeline.md` | Docs | Add humanize stage, update FX values |

---

## 5. Testing

| Test | Method |
|------|--------|
| pyworld scope fix | Run existing `test_kokoro_postprocessor.py`; verify no regression; add test asserting `humanize_voice` called 0 times on full assembly |
| G2P lexicon overrides | Add unit test: mock `KPipeline`, assert `g2p.lexicon.golds` contains override keys after `load_model()` |
| Stress markers | Add unit tests in `test_text_preprocessor.py`: verify `[tension](-1)` and `[peace](+1)` appear in output; verify no double-wrapping on Sanskrit IPA blocks |
| `pure_calm` preset | Add test in `test_voice_manager.py`: verify `get_voice("pure_calm")` returns a tensor with same shape as single-voice tensor; verify no NaN values |
| FX chain tweaks | Run existing `test_meditation_mastering.py`; verify audio passes QA checks |
| Documentation | Visual review only — no automated test |

---

## 6. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| misaki API change breaks G2P lexicon path | Low | `try/except AttributeError` with warning fallback |
| Negative-weight extrapolation produces NaN tensor | Very low | L2-norm check before returning; clamp to primary norm |
| pyworld per-chunk adds latency on long scripts | Low | humanize_voice already called on full audio — per-chunk is same total compute; short clips (<500ms) are skipped |
| Stress markers double-wrap Sanskrit IPA blocks | Low | `(?<!\[)` lookbehind guard in regex |
| Compressor change causes pumping on very quiet passages | Low | 2.5:1 ratio is gentle; -28 dB still conservative relative to broadcast standard (-32 to -40) |

---

## 7. Success Criteria

- All unit tests pass with no regressions
- `humanize_voice` is verifiably called per speech chunk, not on the full assembly
- G2P overrides are loaded once at model initialization and apply globally
- `pure_calm` voice preset is available in the UI and produces a non-NaN tensor
- All three documentation inconsistencies are resolved
- Audio output passes existing QA checks (LUFS, clipping, spectral balance)
