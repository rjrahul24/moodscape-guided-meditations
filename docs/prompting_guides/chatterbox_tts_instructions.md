<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine:** `core/chatterbox_tts/engine.py :: ChatterboxEngine`
**Preprocessor:** `core/kokoro_tts/preprocessor.py :: prepare_segments()` (shared with Kokoro)
**Script tags:** `[pause:Xs]` · `[breath]` / `[inhale]` / `[exhale]` · `\n\n` paragraph break (6.5s)
**Key controls:** `exaggeration` (0.0-1.0, default 0.25) · `cfg_weight` (hardcoded 0.3)
**Voice cloning:** Zero-shot from 5-15s `.wav` reference — no transcript needed
**Output:** 24 kHz mono float32 + voice activity mask
**See also:** `docs/model_implementation_guides/chatterbox_tts.md` · `CLAUDE.md#task-routing-guide`
<!-- ────────────────────────────────────────────────────────────────── -->

# MoodScape — Vocal Script Instructions for Chatterbox TTS

This document is the authoritative reference for writing meditation vocal scripts when using Chatterbox TTS (Resemble AI, 0.5B parameters) in MoodScape. It is intended to be given to an LLM to generate complete, production-ready vocal scripts that the app can process without errors or degraded audio quality.

Chatterbox uses the **same meditation script format and preprocessor as Kokoro** (shared `core/kokoro_tts/preprocessor.py`). The key differences are: emotion intensity control via the `exaggeration` parameter, zero-shot voice cloning from a short reference clip (no transcript required), and MPS-accelerated inference on Apple Silicon.

---

## How the App Processes Your Script

The pipeline from raw script text to audio follows this exact sequence:

```
Raw script text
      |
      v core/kokoro_tts/preprocessor.py -- parse_script()
List of segments: {"type": "speech", "text": "..."},
                  {"type": "pause", "duration_sec": N.N},
                  {"type": "breath", "subtype": "breath|inhale|exhale"}
      |
      v core/kokoro_tts/preprocessor.py -- preprocess_for_meditation()
For each speech segment:
  - Expand digits/abbreviations to spoken words
  - Convert formal phrasing to contractions (warmer prosody)
  - Inject IPA phonemes for Sanskrit/yoga terms
  - Enhance prosody punctuation (comma insertion at breath boundaries)
  - Inject contemplative ellipses before sensory words
  - Vary sentence lengths (promote long-clause commas to periods)
      |
      v core/chatterbox_tts/engine.py -- ChatterboxEngine.synthesize()
For each "speech" segment:
  - _split_text_into_sentences() at .!? boundaries (ellipses preserved)
  - Short sentences (<4 words) merged with next sentence
  - Per sentence: model.generate(text, exaggeration, cfg_weight, audio_prompt_path)
  - Trim trailing silence (-45 dB threshold)
  - Insert room-tone pauses between sentences (0.8s normal, 1.2s after "...")
For each "pause" segment:
  - Insert room-tone noise (from Kokoro postprocessor)
For each "breath" segment:
  - Insert real breath audio sample (from breath_sounds library)
      |
      v Concatenate all chunks
      v Spectral gating (reduce_synthesis_noise)
      |
      v voice_audio (float32, 24 kHz, mono) + voice_activity mask
```

**Critical rule:** The app never passes `[pause:Xs]` markers to Chatterbox. The shared Kokoro preprocessor strips all pause markers and converts them to room-tone audio arrays before any TTS call happens. If you accidentally pass a pause marker string to Chatterbox, it will try to pronounce the words "pause" and "3s" — which is never what you want.

---

## Gradio UI Controls Reference

When "Chatterbox" is selected as the Voice Engine, the following controls become available:

| Control | Type | Range / Options | Default | Description |
|---|---|---|---|---|
| Voice Engine | Radio | Kokoro, F5-TTS, **Chatterbox** | Kokoro | Select Chatterbox for emotion-controlled TTS with voice cloning |
| Emotion Intensity | Slider | 0.0 - 1.0, step 0.05 | **0.45** | Controls `exaggeration` parameter: 0 = monotone, 0.25 = flat calm, 0.45 = meditation ideal (warmth + care), 1.0 = dramatic |
| Voice Reference | Audio upload | `.wav` file, 5-15s | (empty = default voice) | Zero-shot voice cloning from reference clip. No transcript needed. |
| Speaking Speed | Slider | 0.5 - 1.0 | **0.90** | Label reads "pacing controlled by emotion intensity". Speed is passed to `synthesize()` but Chatterbox pacing is primarily driven by `exaggeration` and `cfg_weight`. |
| Voice Reverb | Slider | 0.0 - 1.0 | 0.15 | Convolution reverb wet amount (shared with all TTS engines) |
| Reverb Space | Dropdown | Warm Studio, Wooden Hall, Stone Chapel | Warm Studio | Impulse response selection |
| Music Ducking | Slider | dB | -20 dB | Music attenuation during speech |

**Visibility:** The Chatterbox controls group (`chatterbox-group`) is hidden when Chatterbox is not selected or when the generation mode is "Instrumental Only".

**Hardcoded parameter:** `cfg_weight` is fixed at **0.2** in the pipeline (`pipeline.py`). This value produces slower, more deliberate pacing ideal for meditation — per the official Resemble AI recommendation that lower `cfg_weight` compensates for exaggeration-induced speedup.

---

## Script Format Specification

Chatterbox uses the identical script format as Kokoro (shared preprocessor). All script tags, pause syntax, and paragraph break behavior are the same.

### Pause Syntax

```
[pause:Xs]
```

Where `X` is an integer or decimal number:

```
[pause:3s]       <- 3 seconds of silence
[pause:2.5s]     <- 2.5 seconds of silence
[pause:10s]      <- 10 seconds of silence
[pause:0.5s]     <- half-second (minimum useful value)
```

**Alternate format (also accepted):**
```
[3 second pause]     <- normalised to [pause:3s]
[5 sec pause]        <- normalised to [pause:5s]
```

**Invalid forms that will not be parsed:**
```
[pause: 3s]      <- space before number -- NOT parsed
[pause:3]        <- missing "s" -- NOT parsed (will be spoken as text)
[Pause:3s]       <- capital P -- NOT parsed (case-sensitive)
```

### Breath Markers

```
[breath]       <- natural breath sound (~1.2s)
[inhale]       <- inhale sound (~1.2s)
[exhale]       <- exhale sound (~1.2s)
```

These insert real breath audio samples from the `core/breath_sounds.py` library.

### Paragraph Breaks

A double newline (`\n\n`) is automatically converted to a **6.5-second pause** (the `_PARAGRAPH_PAUSE_SEC` constant in the shared Kokoro preprocessor). Single newlines are treated as a space within the same speech segment.

### Consecutive Pause Merging

The parser merges consecutive pauses. An explicit `[pause:Xs]` overrides a preceding paragraph-break pause; two explicit pauses back-to-back have their durations summed:

```
[pause:3s] [pause:2s]    ->  becomes a single 5-second pause
```

### Automatic Text Preprocessing

The shared Kokoro preprocessor applies all of the following automatically:

- **Digit expansion:** `4` becomes `four`, `120` becomes `one hundred and twenty`
- **Abbreviation expansion:** `sec` becomes `seconds`, `min` becomes `minutes`
- **Contraction conversion:** `you are` becomes `you're` (warmer prosody)
- **Sanskrit/yoga IPA injection:** `chakra`, `pranayama`, `namaste`, etc. receive correct phoneme annotations
- **Prosody punctuation:** Commas inserted at natural breath boundaries
- **Sensory ellipses:** Contemplative `...` inserted before words like `warmth`, `peace`, `stillness`
- **Sentence length variation:** Long clauses may be split into separate sentences

---

## Emotion Intensity Guide (Exaggeration Parameter)

The `exaggeration` parameter is the primary control that distinguishes Chatterbox from Kokoro and F5-TTS. It controls the intensity of emotional expression and vocal dynamics.

| Range | Effect | Use Case | Notes |
|---|---|---|---|
| 0.0 | Monotone, flat delivery | **Avoid** | Robotic quality, no natural variation |
| 0.05 - 0.20 | Very subdued, minimal expression | Deep relaxation, yoga nidra, sleep | Near-whisper quality, extremely calm |
| 0.25 - 0.35 | Calm with subtle warmth | Deep sleep meditations | Gentle but understated |
| **0.40 - 0.50** | **Warm expressiveness with care** | **Standard guided meditation** | **0.45 is the app default and meditation sweet spot** |
| 0.50 - 0.65 | Noticeable expression, engaging delivery | Guided visualization, body scan with active cues | Still meditative with more engagement |
| 0.65 - 0.80 | Expressive, dynamic delivery | Energetic morning meditation | Higher exaggeration speeds up speech |
| 0.80 - 1.0 | Dramatic, highly expressive | **Avoid for meditation** | Extreme values cause instability |

**Interaction with cfg_weight:** Higher exaggeration tends to speed up speech. The hardcoded `cfg_weight=0.2` compensates by producing slower, more deliberate pacing. This combination (moderate exaggeration + low cfg_weight) is the recommended meditation configuration.

**Interaction with reference audio:** The exaggeration parameter modulates the emotional intensity of whatever voice characteristics are captured from the reference clip. A calm reference clip combined with `exaggeration=0.45` produces warm, expressive meditation output.

---

## Voice Cloning Guide

Chatterbox performs zero-shot voice cloning from a short reference audio clip. Unlike F5-TTS, **no transcript of the reference audio is needed**.

### Reference Audio Requirements

| Property | Requirement |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24 kHz or higher (model resamples internally) |
| Channels | Mono preferred (stereo is accepted) |
| Duration | **5-15 seconds** (Resemble AI recommends at least 10s for best quality) |
| Content | Single speaker, no background noise |

### Recording Tips for Meditation Voice Cloning

- **Speak at meditation pace:** Calm, unhurried, approximately 100-120 WPM
- **Use a warm, compassionate tone:** Not whispered, not monotone — gentle and present
- **Record in a quiet room:** No HVAC, fans, or room echo. Soft furnishings absorb reflections.
- **Microphone distance:** 20-30 cm, slightly off-axis to reduce plosives
- **Content:** Speak 2-3 natural meditation sentences. Example: "Gently close your eyes. Allow your breath to find its natural rhythm. Feel the weight of your body being fully supported."
- **Consistency:** Maintain the same distance, volume, and tone throughout the clip

### What Happens Without Reference Audio

If no reference audio is provided (the Voice Reference field is empty), Chatterbox uses its built-in default voice. This voice is suitable for general-purpose narration and works acceptably for meditation.

### Custom Voice Library

Place `.wav` files in `core/chatterbox_tts/assets/reference_audio/` to make them available as named voices. The engine scans this directory on `get_available_voices()` and returns each file as a selectable voice option (filename stem becomes the voice ID).

### Reference Audio Conditioning Internals

Chatterbox processes reference audio through two internal pathways:
- **T3 encoder:** 6 seconds at 16 kHz (96,000 samples) for speaker embedding
- **S3Gen decoder:** 10 seconds at 24 kHz (240,000 samples) for voice characteristics

This means clips shorter than 3 seconds provide insufficient data for reliable cloning, and clips longer than ~15 seconds offer diminishing returns.

---

## Script Writing Best Practices

### Sentence Length and Structure

Chatterbox handles **short to moderate sentences** better than long paragraphs. The engine splits each speech segment into individual sentences at `.!?` boundaries and synthesizes each sentence separately.

- **Ideal sentence length:** 8-20 words
- **Maximum recommended:** 30-40 words (longer sentences risk quality degradation)
- **Minimum:** 4+ words per sentence (shorter sentences are merged with the next by the preprocessor)
- **One idea per sentence** — avoid complex nested subordinate clauses

### Ellipsis for Contemplative Pauses

Use `...` (three dots) or the Unicode ellipsis character intentionally. The engine recognizes them:
- Ellipses are preserved during sentence splitting (not treated as sentence boundaries)
- Sentences ending with `...` receive a **1.8-second** inter-sentence pause (vs. 1.0s for regular endings)
- The preprocessor also automatically injects `...` before sensory words like `warmth`, `peace`, `stillness`

### Inter-Sentence Pauses (Automatic)

The engine automatically inserts room-tone silence between sentences within the same speech segment:

- **1.0 seconds** after sentences ending in `.` `!` `?`
- **1.8 seconds** after sentences ending with `...` or `...`

These are in addition to any explicit `[pause:Xs]` markers.

### Pause Duration Reference

| Context | Recommended Duration |
|---|---|
| Between short phrases (same breath) | `[pause:2s]` - `[pause:3s]` |
| After a breathing instruction (inhale cue) | `[pause:4s]` |
| After a breathing instruction (exhale cue) | `[pause:4s]` - `[pause:6s]` |
| After a body awareness instruction | `[pause:3s]` - `[pause:5s]` |
| After a visualization sentence | `[pause:5s]` |
| Deep stillness / spacious silence | `[pause:8s]` - `[pause:10s]` |
| Between major sections | `[pause:5s]` - `[pause:8s]` |
| Opening / setting the space | `[pause:3s]` - `[pause:5s]` |
| Closing / return to awareness | `[pause:3s]` - `[pause:5s]` |

### Language Style

Chatterbox performs best with:

- **Short, clear sentences** — one idea per sentence
- **Active voice** — "Notice your breath" not "Your breath should be noticed"
- **Simple vocabulary** — avoid medical or technical jargon
- **Present tense and imperatives** — "Feel", "Notice", "Allow", "Let", "Breathe"
- **Natural contractions** — "you're", "let's", "it's" (the preprocessor auto-converts formal phrasing)
- **Avoid parenthetical asides** — they disrupt the synthesis flow

---

## Anti-Patterns

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| `exaggeration > 0.5` for meditation | Dramatic delivery breaks calm atmosphere; higher values speed up speech | Keep at 0.20-0.30 for meditation |
| Using `[laugh]`, `[chuckle]`, `[cough]` tags | Paralinguistic tags are only supported in Chatterbox **Turbo** (not the standard 0.5B model). The standard model speaks these as literal text. | Remove all paralinguistic tags from meditation scripts |
| Noisy reference audio | Cloned voice inherits noise artifacts and room characteristics | Record in a quiet room with no background noise |
| Reference audio < 3 seconds | Insufficient voice characteristics captured by T3/S3Gen encoders | Use 5-15 second clips (10s is ideal) |
| Reference audio > 30 seconds | Diminishing returns, slower processing, no quality improvement | Trim to 10-15 seconds of the best segment |
| Very long sentences (> 50 words) | Chatterbox quality degrades on long inputs; synthesis becomes less coherent | Break into multiple 8-20 word sentences with pauses |
| Missing punctuation | Sentence splitter cannot find boundaries; entire paragraph synthesized as one unit | End every sentence with `.` `!` or `?` |
| `[pause: 3s]` with a space | Not parsed by shared preprocessor; spoken as text | Use `[pause:3s]` exactly |
| Stage directions in script | Chatterbox speaks all text literally | Remove all non-spoken text |
| Reference audio with fast speech | Output inherits fast pacing even at low exaggeration | Re-record reference at meditation pace (~100-120 WPM), or lower `cfg_weight` |
| Stacking consecutive pauses | They merge and total may not match intent | Use a single pause with the total duration |
| Going below speed 0.65 in the UI | Speed slider is passed but Chatterbox pacing is primarily driven by exaggeration/cfg_weight | Use Emotion Intensity slider (0.25 default) for pacing control |

---

## Key Differences from Kokoro and F5-TTS

| Feature | Kokoro | F5-TTS | Chatterbox |
|---|---|---|---|
| Voice selection | Pre-trained voice embeddings (6 presets + blending) | Zero-shot cloned from reference + transcript | Zero-shot cloned from reference only (no transcript) |
| Emotion control | None (prosody via punctuation only) | None (mimics reference delivery) | `exaggeration` dial (0.0-1.0) |
| Pacing control | `speed` parameter (0.65-1.0) | `speed` + optional WPM | `cfg_weight` (hardcoded 0.3) + `exaggeration` interaction |
| G2P | External (misaki + espeak-ng) | Internal (F5's own) | Internal (Chatterbox's own) |
| IPA injection | Supported (`[word](/IPA/)`) | Not supported | Not directly used (preprocessor injects for Kokoro G2P; Chatterbox has its own G2P) |
| Paralinguistic tags | Not supported | Not supported | Standard model: not supported. Turbo only. |
| Script preprocessor | `core/kokoro_tts/preprocessor.py` | `core/f5_tts/preprocessor.py` | `core/kokoro_tts/preprocessor.py` (shared) |
| Paragraph pause | 6.5 seconds | 3.0 seconds | 6.5 seconds (shared preprocessor) |
| Inter-sentence gap | 0.8s (1.2s after `...`) | 0.4s + 300ms crossfade | 0.8s (1.2s after `...`) |
| Device | CPU only (MPS bus errors) | MPS | MPS (with patched torch.load) |
| Model size | 82M parameters | ~1.2B parameters | 0.5B parameters (~4-6 GB memory) |
| Voice FX chain | `build_voice_chain()` | `build_f5_voice_chain()` | `build_voice_chain()` (same as Kokoro) |
| Output sample rate | 24 kHz | 24 kHz | 24 kHz |

---

## Concrete Example — 5-Minute Body Scan Meditation

### Recommended UI Settings

| Setting | Value | Rationale |
|---|---|---|
| Voice Engine | Chatterbox | Emotion-controlled TTS |
| Emotion Intensity | 0.25 | Calm delivery with natural micro-expressiveness |
| Voice Reference | Clean 10s `.wav` of calm speaker, or empty for default | Zero-shot cloning captures target voice |
| Speaking Speed | 0.90 | Meditation pace (pacing primarily driven by exaggeration/cfg_weight) |
| Voice Reverb | 0.15 | Subtle room presence |
| Reverb Space | Warm Studio | Intimate, short decay |
| Music Ducking | -20 dB | Music drops during speech |

### Script

```
Welcome to this body scan meditation. [pause:3s]

Find a comfortable position, either lying down or seated. [pause:4s]

Gently close your eyes. [pause:5s]

[breath]

Take a slow breath in through your nose... [pause:4s] and release through your mouth. [pause:6s]

[breath]

Again, breathe in... [pause:4s] and let go. [pause:6s]

Begin by bringing your awareness to the top of your head. [pause:3s] Notice any sensations there. [pause:5s]

Slowly move your attention down to your forehead. [pause:3s] Feel any tension, and allow it to soften. [pause:5s]

Let your awareness drift to your eyes and cheeks. [pause:3s] Release any tightness you find there. [pause:5s]

[breath]

Now bring your attention to your jaw and mouth. [pause:3s] Let your jaw hang open slightly. [pause:4s]

Feel the warmth spreading down through your neck. [pause:3s] Your shoulders begin to drop. [pause:5s]

Move your awareness into your arms. [pause:3s] From shoulders to elbows, elbows to wrists, wrists to fingertips. [pause:6s]

[breath]

Now bring your attention to your chest. [pause:3s] Feel the gentle rise and fall of each breath. [pause:6s]

Let your awareness sink into your belly. [pause:3s] Notice how it softens with each exhale. [pause:5s]

[breath]

Move your attention to your lower back. [pause:3s] Send warmth and kindness to this area. [pause:5s]

Feel your awareness flowing down through your hips. [pause:3s] Through your thighs. [pause:3s] Past your knees. [pause:5s]

Continue down through your calves and shins. [pause:3s] Into your ankles and feet. [pause:4s] All the way to the tips of your toes. [pause:6s]

[breath]

Now feel your entire body as one. [pause:3s] A single field of awareness and sensation. [pause:8s]

There is nothing you need to do right now. [pause:5s] Simply rest in this awareness. [pause:10s]

[breath]

When you are ready, begin to deepen your breath. [pause:4s]

Gently wiggle your fingers and toes. [pause:3s]

Bring your awareness back to the room around you. [pause:5s]

Slowly open your eyes. [pause:3s]

Thank you for taking this time for yourself. [pause:3s]
```

### What the Preprocessor Produces

The shared Kokoro preprocessor converts the script above into a sequence of typed segments:

```
[speech] "Welcome to this body scan meditation."
[pause]  3.0s
[pause]  6.5s  (paragraph break)
[speech] "Find a comfortable position, either lying down or seated."
[pause]  4.0s
[pause]  6.5s  (paragraph break)
[speech] "Gently close your eyes."
[pause]  5.0s
[pause]  6.5s  (paragraph break)
[breath] breath
[pause]  6.5s  (paragraph break)
[speech] "Take a slow breath in through your nose..."
[pause]  4.0s
[speech] "and release through your mouth."
[pause]  6.0s
...
```

Each `[speech]` segment is then processed by `preprocess_for_meditation()` (contraction conversion, prosody punctuation, sensory ellipses) before being passed to `ChatterboxEngine.synthesize()`, which further splits into individual sentences for per-sentence generation.

---

## Recommended UI Settings Summary

| Setting | Value | Rationale |
|---|---|---|
| **Voice Engine** | Chatterbox | Emotion-controlled TTS with voice cloning |
| **Emotion Intensity** | **0.25** | Research-validated calm delivery; natural micro-expressiveness without drama |
| **Voice Reference** | Clean 5-15s `.wav` or empty | Default voice is suitable; cloned voice matches reference characteristics |
| **Speaking Speed** | **0.90** | Meditation pace (actual pacing primarily driven by exaggeration + cfg_weight) |
| **Voice Reverb** | 0.15 | Subtle room presence. Increase to 0.25 for spacious feel. |
| **Reverb Space** | Warm Studio | Intimate, short decay. Use Wooden Hall for longer sessions. |
| **Music Ducking** | -20 dB | Music drops 20 dB during speech. Use -25 dB if music overpowers voice. |
| **Fade In** | 3 seconds | Gentle entry |
| **Fade Out** | 6 seconds | Gradual ending |
| **Output Format** | WAV | Lossless. Use MP3 only for distribution. |

### Music Engine Pairing

Chatterbox TTS works with any music engine. Recommended pairings:

| Music Engine | Best For |
|---|---|
| **ACE-Step 1.5** | Rich, evolving ambient textures. Best quality but slower generation. |
| **HeartMuLa** | Simple ambient pads. Faster generation, more predictable output. |
| **Lyria RealTime** | Cloud-generated, high-quality ambient music. Requires API key. |

---

## Audio Output Specification

The Chatterbox TTS engine outputs:
- **Sample rate:** 24,000 Hz (24 kHz) — resampled to match if the model's native rate differs
- **Channels:** Mono (1 channel)
- **Data type:** float32, values in range -1.0 to +1.0
- **Voice activity mask:** Boolean array (True = speaking, False = silence) used for music ducking
- **Mixing sample rate:** 48,000 Hz (upsampled from 24 kHz with high-accuracy `librosa soxr_vhq` resampling)
- **Voice FX chain:** Same as Kokoro (`build_voice_chain()`) — NoiseGate, HPF, EQ, Compressor, presence boost, ConvReverb, Limiter

---

## Duration Estimation

At default settings (exaggeration=0.25, cfg_weight=0.3, speed=0.90):

- **~120-150 words of speech** approximates 1 minute of audio (before adding pauses)
- Add all explicit pause durations (in seconds) to the speech duration
- Paragraph breaks (`\n\n`) add 6.5 seconds each
- `[breath]` markers add approximately 0.6 seconds each (room-tone pause; Chatterbox generates naturalistic breathing via exaggeration)
- Automatic inter-sentence pauses add 1.0-1.8 seconds between each pair of sentences

**Target length examples:**
- 5-minute meditation: ~400-500 words of speech + 90-120s of total explicit pauses
- 10-minute meditation: ~900-1100 words of speech + 150-200s of total explicit pauses
- 20-minute meditation: ~1800-2200 words of speech + 300-400s of total explicit pauses
