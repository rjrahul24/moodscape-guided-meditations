<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/index_tts/preprocessor.py` · `core/index_tts/engine.py` · `core/index_tts/voice_registry.py`
**Script tags:** `[pause:Xs]` · `[breath]` · `\n\n` paragraph break (3.5s) · `[voice:phase_name]` for multi-phase
**Chunk limit:** 250 chars (auto-split at sentence boundaries)
**Text rules:** colons → commas · ellipses → periods · em-dashes → commas · hyphens in compounds removed
**Voice assets:** `assets/speakers/*.wav` (24kHz mono, 5-10s) — no transcript required
**Emotion assets:** `assets/emotions/*.wav` (24kHz mono, 3-10s) or user-uploaded via Gradio UI
**See also:** `docs/model_implementation_guides/index_tts.md` · `CLAUDE.md#task-routing-guide`
<!-- ────────────────────────────────────────────────────────────────── -->

# MoodScape — Vocal Script Instructions for IndexTTS-2

This document is the authoritative reference for writing meditation vocal scripts when using IndexTTS-2 in MoodScape. It is intended to be given to an LLM to generate complete, production-ready vocal scripts that the app can process without errors or degraded audio quality.

IndexTTS-2 is a zero-shot voice cloning engine with independent emotion control — it reproduces the voice timbre from a reference audio clip while applying emotional tone from a separate emotion reference or preset. The quality of your output depends on both the reference audio and the script you write.

---

## How the App Processes Your Script

The pipeline from raw script text to audio follows this exact sequence:

```
Raw script text
      │
      ▼ core/index_tts/preprocessor.py — parse_script()
List of segments: {"type": "speech", "text": "..."},
                  {"type": "pause", "duration_sec": N.N},
                  {"type": "breath", "subtype": "breath|inhale|exhale"},
                  {"type": "voice", "voice": "phase_name"}
      │
      ▼ core/index_tts/preprocessor.py — normalize_for_indextts() + split_into_chunks()
Meditation lexicon phoneticizes sanskrit/pali terms (Om → ohm, pranayama → prah-nah-yama, ...)
Each speech block split into ≤220-character chunks at sentence boundaries
(220 keeps BPE-token count safely under IndexTTS-2's 120-token attention window)
      │
      ▼ core/index_tts/engine.py — IndexTTSEngine.synthesize()
For each "speech" chunk:
  - Normalise text (collapse whitespace, lowercase ALL_CAPS)
  - Call IndexTTS2.infer() with the full meditation-tuned param set:
      emo_vector=[0,0,0,0,0,0,0,1.0] (calm)  OR  emo_audio_prompt=<path>
      emo_alpha=0.70, top_p=0.85, temperature=0.70,
      interval_silence=200ms, max_text_tokens_per_segment=120
  - Read output WAV → float32 numpy array
  - Trim trailing silence, apply Silero VAD
For each "pause" segment:
  - Insert room-tone noise
For each "breath" segment:
  - Insert real breath sample
      │
      ▼ Assembly — 0.6s room-tone gap + 300ms crossfade between speech chunks
      │
      ▼ Concatenate all chunks → voice_audio (float32, 24 kHz, mono)
         + voice_activity mask (True where voice is speaking)
```

**Critical rule:** The app never passes `[pause:Xs]` markers to IndexTTS-2. The script parser strips all pause markers and converts them to room tone arrays before any TTS call happens.

---

## Script Format Specification

### Pause Syntax

```
[pause:Xs]
```

Where `X` is an integer or decimal number. Examples:

```
[pause:3s]       ← 3 seconds of silence
[pause:2.5s]     ← 2.5 seconds of silence
[pause:10s]      ← 10 seconds of silence
[pause:0.5s]     ← half-second (minimum useful value)
```

Alternate format (also accepted):

```
[3 second pause]     ← normalised to [pause:3s]
[5 sec pause]        ← normalised to [pause:5s]
```

### Breath Markers

```
[breath]       ← 1.2s natural breath sound
[inhale]       ← 1.2s inhale sound
[exhale]       ← 1.2s exhale sound
```

These insert real breath audio samples.

### Paragraph Breaks

A double newline (`\n\n`) is automatically converted to a `[pause:3.5s]` marker — a paragraph-level pause. Single newlines are treated as a space within the same speech segment.

**Exception:** If one side of a `\n\n` boundary contains only a pause marker (e.g. `[pause:5s]` on its own line), the paragraph break is treated as formatting and no extra pause is added.

### Voice Phase Switching

```
[voice:phase_name]
```

Switches to a different reference audio phase mid-script (when multi-phase voices are configured).

### Consecutive Pause Merging

Back-to-back pauses are merged, keeping only the **longest** duration:

```
[pause:3s] [pause:5s]    →  becomes a single 5-second pause (not 8s)
```

---

## Key Differences from Kokoro and F5-TTS Scripts

| Feature | Kokoro | F5-TTS | IndexTTS-2 |
|---|---|---|---|
| Voice | Pre-trained embeddings | Zero-shot (ref audio + transcript) | Zero-shot (ref audio only) |
| Emotion control | Not supported | Not supported | ✓ (separate emotion ref audio) |
| G2P | External (misaki + espeak-ng) | Internal (F5's own G2P) | Internal (autoregressive G2P) |
| IPA injection | Supported | Not supported | Not supported |
| Text expansion | Auto | Auto | Auto |
| Chunk limit | 150 tokens | 300 characters | **250 characters** |
| Paragraph pause | 2.5 seconds | 3.0 seconds | **3.5 seconds** |
| Inter-sentence gap | 0.8s | 0.4s + 300ms crossfade | **0.6s + 300ms crossfade** |
| Breath sounds | Not supported | Supported | Supported |
| Speed default | 0.70 | 0.88 | **1.0** (natural rhythm) |
| Transcript required | N/A | Yes (verbatim .txt) | **No** |

**Important:** Do not use IPA phoneme injection syntax with IndexTTS-2. It will not be parsed and will be spoken as literal text.

---

## Writing Style for IndexTTS-2

IndexTTS-2 performs its own grapheme-to-phoneme conversion from raw text via its autoregressive decoder. Write natural, conversational meditation prose.

### Do

- Write in natural prose — spell words as they should be read
- Keep sentences short (under 180 characters ideal, under 250 maximum)
- Use explicit `[pause:Xs]` markers for precise timing control
- Use `[breath]` markers between breathing instructions for natural rhythm
- Use commas for subtle micro-pauses within sentences
- Write one clear idea per sentence
- Use shorter sentences than F5-TTS (250 vs 300 char limit)
- Take advantage of emotion control — the model can produce calm/warm/soothing delivery

### Do Not

- Do not use IPA injection syntax — IndexTTS-2 ignores it and speaks it literally
- Do not use ALL CAPS for emphasis — the engine auto-lowercases them
- Do not write sentences longer than 250 characters — they will be split mid-sentence
- Do not include stage directions or notes — IndexTTS-2 will attempt to speak them
- Do not rely on single newlines for pauses — they are treated as spaces
- Do not use parenthetical asides — they disrupt autoregressive prosody
- Avoid very short standalone sentences (1–2 words) — autoregressive models may produce unreliable audio

---

## Emotion Control

IndexTTS-2 supports emotion control independently of speaker identity through a Gradient Reversal Layer that disentangles timbre from arousal. The engine ships with **two paths**, and the audio reference takes precedence when both are available.

### Default: calm vector preset

When no emotion audio is selected, the engine passes the deterministic 8D vector:

```
emo_vector = [happy=0, angry=0, sad=0, afraid=0, disgusted=0,
              melancholic=0, surprised=0, calm=1.0]
emo_alpha  = 0.70   # 70% emotion override + 30% speaker timbre
```

This is the recommended path for meditation. It's reproducible across runs, avoids the Qwen3 text-emotion path's known "calm → sad" misclassification, and lands at the API's perceptually-tuned maximum calm intensity (~0.5625 post-bias) without triggering the 0.8-sum penalty.

### Override: audio reference

Selecting an emotion in the Gradio dropdown or uploading a custom emotion clip switches to `emo_audio_prompt=<path>`. Use this for emotions outside the 8D vector space, or to match a specific recorded mood.

### Available emotion reference WAVs

Emotion is controlled via reference audio clips placed in `assets/emotions/`:

| Emotion | File | Best For |
|---|---|---|
| Calm | `calm.wav` | Body scans, relaxation, general meditation |
| Warm | `warm.wav` | Loving-kindness, gratitude, compassion |
| Soothing | `soothing.wav` | Sleep meditations, deep relaxation |
| Neutral | (none) | Instructional meditations, technique teaching |

### Custom Emotion Audio

Users can also upload custom emotion reference audio directly in the Gradio UI. This is useful for:
- Creating unique emotional tones not covered by presets
- Matching the emotion to a specific meditation theme
- Fine-tuning the emotional delivery for a particular voice

### Emotion Tips for Meditation Scripts

- **Body scan meditations**: Use "calm" — steady, grounded emotional delivery
- **Loving-kindness meditations**: Use "warm" — nurturing, compassionate tone
- **Sleep meditations**: Use "soothing" — gentle, lullaby-like quality
- **Breathing exercises**: Use "calm" or "neutral" — clear, instructional delivery

---

## Speaking Speed — Not a Knob

**The speed slider is disabled for IndexTTS-2.** The v2 API does not expose reliable time-stretching (Issue [#422](https://github.com/index-tts/index-tts/issues/422)), and post-hoc time-stretching introduces phase artifacts unacceptable for headphone-grade meditation audio.

Pacing is shaped instead by:

1. **The calm emotion vector** — pure calm prosody is naturally slower and breathier than the reference audio.
2. **The reference audio's natural pace** — speak slowly when recording your speaker reference.
3. **Explicit `[pause:Xs]` markers** — the strongest pacing tool available to the script writer.
4. **Paragraph breaks** — auto-inserted 3.5s pauses between paragraphs.
5. **`interval_silence=200ms`** — API-internal silence between micro-segments.
6. **The 600ms room-tone gap + 300ms crossfade** the engine adds between speech chunks.

---

## Meditation Lexicon — Auto-Phoneticized Terms

The preprocessor automatically rewrites common sanskrit/pali terms into phonetic spellings before they reach the BPE tokenizer (which would otherwise fracture them and produce letter-by-letter reads or fricative artifacts). You can write these terms naturally in your script:

| Source | Rewritten to |
|---|---|
| Om, Aum | ohm |
| pranayama | prah-nah-yama |
| vipassana | vi-pah-sana |
| metta | meh-tah |
| samadhi | sah-mah-dee |
| namaste | nah-mas-tay |
| chakra | chah-kra |
| kundalini | koon-da-lee-nee |
| mudra | moo-drah |
| savasana / shavasana | shah-vah-sana |
| koan | koh-an |
| zazen | zah-zen |
| dharma | dar-ma |
| sangha | sang-ha |
| tantra | tahn-tra |
| sutra | soo-tra |

Add more terms by extending `INDEX_MEDITATION_LEXICON` in [core/index_tts/preprocessor.py](../../core/index_tts/preprocessor.py). Patterns are case-insensitive regexes; values use hyphens for syllabification and must survive the compound-hyphen stripper that runs before the lexicon pass.

---

## Reference Audio Best Practices

### Speaker Reference (Voice Cloning)

| Property | Value |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24,000 Hz |
| Channels | Mono |
| Duration | 5–10 seconds |
| Bit depth | 16-bit or 32-bit float |

**Key difference from F5-TTS:** No verbatim transcript file is needed. IndexTTS-2 performs its own speech analysis directly from the audio.

### Recording Guidelines

**Pace and delivery:**
- Speak at the exact pace you want in the output
- Target approximately 100–120 words per minute
- Leave natural micro-pauses between sentences
- Use a calm, warm, compassionate tone

**Content:**
- Speak 2–3 complete, natural meditation sentences
- Content should represent the meditation style
- Avoid filler words unless they are part of your delivery

**Environment:**
- Quiet room with soft furnishings
- No HVAC, fans, or appliance noise
- Pop filter recommended
- Microphone 20–30 cm from mouth

Convert any recording:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 voice_name.wav
```

---

## Recommended Gradio UI Settings for IndexTTS-2

| Setting | Recommended Value | Notes |
|---|---|---|
| **TTS Voice Engine** | IndexTTS-2 | Autoregressive voice cloning with emotion control |
| **Voice** | (select your voice) | Choose from voices in `assets/speakers/` |
| **Emotion** | Calm | Recommended for most meditations |
| **Custom Emotion Audio** | (optional) | Upload WAV for custom emotion |
| **Speaking Speed** | 1.0 | Natural rhythm (default) |
| **Voice Reverb** | 0.15 | Subtle room presence |
| **Reverb Space** | Warm Studio | Intimate, short decay |
| **Music Ducking** | -20 dB | Music drops during speech |
| **Fade In** | 3 seconds | Gentle entry |
| **Fade Out** | 6 seconds | Gradual ending |
| **Output Format** | WAV | Lossless quality |

---

## Sentence Length & Structure Guidelines

IndexTTS-2 chunks text at sentence boundaries (`.`, `!`, `?`, `…`) up to 250 characters per chunk. Each chunk is synthesized in a single autoregressive pass.

### Optimal Sentence Structure

- **Ideal length:** 8–18 words per sentence
- **Maximum:** Keep sentences under 180 characters for best quality (250 is the hard limit)
- **Minimum:** Avoid 1–2 word standalone sentences — they produce unreliable audio
- **One idea per sentence** — do not nest complex subordinate clauses

**Too complex (avoid):**
```
As you breathe in slowly and deeply through your nostrils while feeling the cool air enter your body and simultaneously becoming aware of the expansion of your ribcage, allow your mind to quiet itself.
```

**Better:**
```
Breathe in slowly through your nose. [pause:4s] Feel the cool air entering your body. [pause:3s] Notice your ribcage gently expanding. [pause:3s] Allow your mind to become quiet. [pause:5s]
```

### Pause Duration Reference

| Context | Recommended Duration |
|---|---|
| Between short phrases (same breath) | `[pause:2s]` – `[pause:3s]` |
| After a breathing instruction | `[pause:4s]` – `[pause:6s]` |
| After a body awareness cue | `[pause:3s]` – `[pause:5s]` |
| After a visualisation sentence | `[pause:5s]` – `[pause:7s]` |
| Deep stillness / spacious silence | `[pause:8s]` – `[pause:10s]` |
| Between major sections | Use paragraph break (auto 3.5s) or `[pause:6s]` – `[pause:8s]` |
| Opening / setting the space | `[pause:3s]` – `[pause:5s]` |
| Closing / return to awareness | `[pause:3s]` – `[pause:5s]` |

---

## Anti-Patterns — What NOT to Do

| Anti-Pattern | What Goes Wrong | Fix |
|---|---|---|
| IPA injection `[word](/ˈaɪpə/)` | Spoken as literal text | Remove — IndexTTS-2 does its own G2P |
| ALL CAPS words like "RELAX" | Auto-lowercased | Write "relax" in normal case |
| Digits ("breathe for 4 counts") | Auto-expanded, but words are clearer | Write "breathe for four counts" |
| Sentences > 250 chars | Split mid-sentence, may cause repetition | Break into shorter sentences |
| Stage directions in script | Spoken aloud | Remove all non-spoken text |
| Single newline for pause | Treated as a space | Use `[pause:Xs]` or double newline |
| Very short sentences (1–2 words) | Unreliable audio quality | Attach to adjacent sentence |
| Long unbroken paragraphs | Many chunks without pauses | Break with explicit pauses every 2–3 sentences |
| Using F5's speed (0.88) | IndexTTS-2 uses different scale | Use 1.0 for natural rhythm |

---

## Complete Example Script

This is a well-formed script demonstrating all features for IndexTTS-2:

```
Welcome to this moment of stillness. [pause:3s]

Allow yourself to arrive, just as you are. [pause:5s]

Find a comfortable position. [pause:2s] Let your body settle into the surface beneath you. [pause:4s]

Gently close your eyes. [pause:5s]

[breath]

Take a slow breath in through your nose. [pause:4s] And release through your mouth. [pause:6s]

[breath]

Again, breathe in. [pause:4s] And let go. [pause:6s]

Notice the weight of your body. [pause:3s] Feel how you are fully supported. [pause:5s]

There is nothing you need to do right now. [pause:4s] Nothing to fix, nothing to change. [pause:6s]

Let any thoughts that arise drift past, like clouds moving across a wide sky. [pause:8s]

[breath]

With each exhale, release a little more. [pause:5s]

You are here. [pause:3s] You are safe. [pause:3s] You are completely at peace. [pause:10s]

When you are ready, begin to deepen your breath. [pause:4s]

Gently bring your awareness back to the room. [pause:5s]

Slowly open your eyes. [pause:3s]

Thank you for taking this time for yourself. [pause:3s]
```

**What this script produces (approximately):**
- At speed 1.0 with a calm reference and "calm" emotion, approximately 3–4 minutes of audio
- Each `[pause:Xs]` creates room-tone silence of that exact duration
- `[breath]` markers insert real breath audio samples (1.2s each)
- Automatic 0.6s room-tone gaps + 300ms crossfades between speech chunks
- Paragraph breaks (blank lines) add 3.5s pauses

---

## Audio Output Specification

The IndexTTS-2 engine outputs:
- **Sample rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono (1 channel)
- **Data type:** float32, values in range -1.0 to +1.0
- **Voice activity mask:** Boolean array (True = speaking, False = silence) used for music ducking
- **Mixing sample rate:** 48,000 Hz (upsampled from 24 kHz with high-accuracy sinc resampling)

---

## Duration Estimation

At the default speed of 1.0 (natural rhythm mode), IndexTTS-2 generates approximately 8–12 seconds of audio per 250 characters of text (roughly 40–55 words).

Rough rules of thumb:
- **~100–130 words of speech** ≈ 1 minute of audio (before adding pauses)
- Add all explicit pause durations to the speech duration
- `[breath]` markers add ~1.2 seconds each

**Target length examples:**
- 5-minute meditation: ~400–500 words of speech + 90–120s of total explicit pauses
- 10-minute meditation: ~800–1000 words of speech + 150–200s of total explicit pauses
- 20-minute meditation: ~1600–2000 words of speech + 300–400s of total explicit pauses
