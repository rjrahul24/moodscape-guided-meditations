# MoodScape — Vocal Script Instructions for F5-TTS

This document is the authoritative reference for writing meditation vocal scripts when using F5-TTS in MoodScape. It is intended to be given to an LLM to generate complete, production-ready vocal scripts that the app can process without errors or degraded audio quality.

F5-TTS is a zero-shot voice cloning engine — it reproduces the voice, tone, and pace of a short reference audio clip. The quality of your output depends on both the reference audio and the script you write.

---

## How the App Processes Your Script

The pipeline from raw script text to audio follows this exact sequence:

```
Raw script text
      │
      ▼ core/f5_tts/preprocessor.py — parse_script()
List of segments: {"type": "speech", "text": "..."},
                  {"type": "pause", "duration_sec": N.N},
                  {"type": "breath", "subtype": "breath|inhale|exhale"},
                  {"type": "voice", "voice": "phase_name"}
      │
      ▼ core/f5_tts/preprocessor.py — split_into_chunks()
Each speech block split into ≤300-character chunks at sentence boundaries
      │
      ▼ core/f5_tts/engine.py — F5Engine.synthesize()
For each "speech" chunk:
  - Normalise text (collapse whitespace, lowercase ALL_CAPS, add trailing "...")
  - Select reference audio (always the static reference for consistent voice)
  - Call F5TTS infer() with speed, nfe_step=32, cfg_strength=1.0, seed
  - Trim trailing silence, build activity mask
For each "pause" segment:
  - Insert room tone (low-level ambient noise, ~-60 dBFS)
For each "breath" segment:
  - Insert real breath sample blended with room tone
      │
      ▼ Silero VAD — crop trailing non-speech, attenuate interior gaps to 15%
      ▼ Assembly — 0.8s room tone gap + 300ms crossfade between speech chunks
      │
      ▼ Concatenate all chunks → voice_audio (float32, 24 kHz, mono)
         + voice_activity mask (True where voice is speaking)
```

**Critical rule:** The app never passes `[pause:Xs]` markers to F5-TTS. The script parser strips all pause markers and converts them to room tone arrays before any TTS call happens.

---

## Script Format Specification

### Pause Syntax

```
[pause:Xs]
```

Where `X` is an integer or decimal number. Examples:

```
[pause:3s]       ← 3 seconds of room tone
[pause:2.5s]     ← 2.5 seconds of room tone
[pause:10s]      ← 10 seconds of room tone
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

These insert real breath audio samples, blended with room tone.

### Paragraph Breaks

A double newline (`\n\n`) is automatically converted to a `[pause:6.5s]` marker — a long paragraph-level pause. Single newlines are treated as a space within the same speech segment.

**Exception:** If one side of a `\n\n` boundary contains only a pause marker (e.g. `[pause:5s]` on its own line), the paragraph break is treated as formatting and no extra pause is added.

### Voice Phase Switching

```
[voice:phase_name]
```

Switches to a different reference audio phase mid-script. This requires multi-phase voice definitions in `voices.toml`. See the Multi-Phase Voices section below.

### Consecutive Pause Merging

Back-to-back pauses are merged, keeping only the **longest** duration:

```
[pause:3s] [pause:5s]    →  becomes a single 5-second pause (not 8s)
```

---

## Key Differences from Kokoro Scripts

| Feature | Kokoro | F5-TTS |
|---|---|---|
| Voice | Pre-trained voice embeddings | Zero-shot cloned from reference audio |
| G2P | External (misaki + espeak-ng) | Internal (F5's own G2P from raw text) |
| IPA injection | Supported (`[word](/ˈaɪpə/)`) | Not supported — do not use |
| Text expansion | Auto (digits, abbreviations) | Not applied — write natural prose |
| Chunk limit | 150 tokens | 300 characters |
| Paragraph pause | 2.5 seconds | 6.5 seconds |
| Inter-sentence gap | 0.8s (1.2s after `...`) | 0.8s room tone + 300ms crossfade |
| Breath sounds | Not supported | `[breath]`, `[inhale]`, `[exhale]` |
| Speed default | 0.70 | 0.80 |
| Auto trailing `...` | No | Yes (appended if sentence doesn't end with `?`, `!`, or `...`) |

**Important:** Do not use IPA phoneme injection syntax with F5-TTS. It will not be parsed and will be spoken as literal text.

---

## Writing Style for F5-TTS

F5-TTS performs its own grapheme-to-phoneme conversion from raw text. Write natural, conversational meditation prose.

### Do

- Write in natural prose — spell words as they should be read
- Keep sentences short (under 200 characters ideal, under 300 maximum)
- Use explicit `[pause:Xs]` markers for precise timing control
- Use `[breath]` markers between breathing instructions for natural rhythm
- Use `...` (ellipsis) for trailing, drifting quality — the engine preserves these
- Use commas for subtle micro-pauses within sentences
- Write one clear idea per sentence

### Do Not

- Do not use IPA injection syntax — F5 ignores it and speaks it literally
- Do not use ALL CAPS for emphasis — the engine auto-lowercases them to prevent letter-by-letter G2P spelling
- Do not write numbers as digits (write "four" not "4") — F5's G2P may stumble on digits
- Do not include stage directions or notes in the script — F5 will attempt to speak them
- Do not write sentences longer than 300 characters — they will be split mid-sentence
- Do not rely on single newlines for pauses — they are treated as spaces
- Do not use parenthetical asides — they disrupt F5's prosody model

---

## Available Voices

F5-TTS voices are zero-shot cloned from reference audio clips. The following voices are pre-registered:

| Voice (UI Label) | Slug | Character |
|---|---|---|
| Female Meditative Warm | `female_meditative_warm` | Warm, introspective tone — good for body scans and visualisations |
| Female Relaxing Calm | `female_relaxing_calm` | Calm, supportive delivery — good for sleep and relaxation meditations |
| Female Warm Composed | `female_warm_composed` | Composed, settled presence — good for grounding meditations |

### Adding Custom Voices

To register a new voice:

1. Place a 24 kHz mono WAV file in `core/f5_tts/assets/reference_audio/your_voice_name.wav`
2. Place a verbatim transcript in `core/f5_tts/assets/reference_transcript/your_voice_name.txt`
3. Restart the app — the voice will appear in the dropdown automatically

The slug is derived from the filename (e.g. `your_voice_name.wav` → slug `your_voice_name`).

---

## Speaking Speed

The `speed` slider in the UI ranges from **0.65 to 1.0**. When F5-TTS is selected, it automatically defaults to **0.80**.

F5's speed parameter controls the duration predictor's mel frame count — lower values produce slower, more deliberate speech. Unlike Kokoro, F5 does not have a hard distortion floor, but below 0.70 the duration predictor's variance increases (less consistent pacing across chunks).

| Use Case | Speed Value |
|---|---|
| Fast/clear narration | 1.0 |
| Normal speaking pace | 0.90 |
| Gentle meditation pace | 0.85 |
| **F5 default (recommended)** | **0.80** |
| Deep relaxation / sleep | 0.75 |
| Very slow, spacious delivery | 0.70 |

**Critical:** The reference audio pace also influences output pace. If your reference was recorded at a fast conversational speed, the model will tend to generate faster speech even at lower speed values. Record your reference audio at the same deliberate, unhurried meditation pace you want in the output.

---

## Reference Audio Best Practices

The quality of F5-TTS output depends heavily on the reference audio. Follow these guidelines for optimal results.

### Format Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24,000 Hz |
| Channels | Mono |
| Duration | 10–12 seconds |
| Bit depth | 16-bit or 32-bit float |

Convert any recording:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 voice_name.wav
```

### Recording Guidelines

**Pace and delivery:**
- Speak at the exact pace you want in the output — F5's duration predictor learns from the reference
- Target approximately 100–120 words per minute (significantly slower than conversational ~150 WPM)
- Leave natural micro-pauses between sentences (do not rush from one sentence to the next)
- Use a calm, warm, compassionate tone — not whispered, not monotone

**Content:**
- Speak 2–3 complete, natural meditation sentences
- The content should be representative of the meditation style (e.g. body scan cues, breathing instructions, grounding phrases)
- Avoid filler words ("um", "uh") unless they are part of your natural delivery
- Examples of good reference text:
  - "Gently close your eyes. Allow your breath to find its natural rhythm. Feel the weight of your body being fully supported."
  - "Begin by finding a comfortable position. Let your shoulders drop away from your ears. Allow each exhale to carry away any tension."

**Environment:**
- Quiet room with soft furnishings (absorbs reflections)
- No HVAC, fans, or appliance noise
- Use a pop filter to prevent plosive bursts (which cause alignment failures)
- Microphone 20–30 cm from mouth, slightly off-axis
- Do not record in bathrooms or bare-walled rooms (excessive reverb degrades cloning)

**Transcript:**
- The transcript file must be **verbatim** — every word exactly as spoken, including any filler words
- Match punctuation to how the speech was delivered
- Do NOT add IPA or phonetic spelling
- A mismatch between transcript and audio causes metallic artefacts, pitch drift, or stutter

### What the Engine Does Automatically

You do not need to worry about:
- **RMS normalisation** — reference audio is automatically normalised to -20 dBFS
- **Whisper transcription** — if the transcript is empty, F5 auto-transcribes with Whisper
- **Clipping** — F5 clips the reference to ≤12 seconds internally

---

## Recommended Gradio UI Settings for F5-TTS

These are the recommended settings for top quality meditation audio generation with F5-TTS:

| Setting | Recommended Value | Notes |
|---|---|---|
| **TTS Voice Engine** | F5-TTS | Zero-shot voice cloning |
| **Voice Personality** | (select your voice) | Choose from registered voices |
| **Speaking Speed** | 0.80 | Auto-set when F5 is selected. Adjust 0.75–0.85 for preference. |
| **Voice Reverb** | 0.15 | Subtle room presence. Increase to 0.25 for spacious feel. |
| **Reverb Space** | Warm Studio | Intimate, short decay — best for meditation. Use Wooden Hall for longer sessions. |
| **Music Ducking** | -20 dB | Music drops 20 dB during speech. Use -25 dB if music is overpowering voice. |
| **Fade In** | 3 seconds | Gentle entry |
| **Fade Out** | 6 seconds | Gradual ending |
| **Output Format** | WAV | Lossless. Use MP3 only for distribution. |

### Music Engine Pairing

F5-TTS works well with any music engine. Recommended pairings:

| Music Engine | Best For |
|---|---|
| **ACE-Step** (Studio quality) | Rich, evolving ambient textures. Best quality but slower generation. |
| **MusicGen** | Simple ambient pads. Faster generation, more predictable output. |
| **Lyria** | Cloud-generated, high-quality ambient music. Requires API key. |

---

## Multi-Phase Voices

A single voice can have multiple **phases** for different sections of a meditation. This allows you to use slightly different vocal qualities for opening vs. closing, or different reference paces for active vs. settling sections.

### Defining Phases in voices.toml

```toml
# core/f5_tts/assets/voices.toml

[female_meditative_warm]
default = { audio = "female_meditative_warm.wav", transcript = "female_meditative_warm.txt" }
closing = { audio = "female_meditative_warm_closing.wav", transcript = "female_meditative_warm_closing.txt" }
```

Each phase needs its own reference audio + transcript pair in the assets directories.

### Using Phases in Scripts

```
Welcome to this meditation. [pause:3s]

Find a comfortable position and close your eyes. [pause:5s]

[voice:closing]

Thank you for joining. Carry this peace with you into the rest of your day. [pause:3s]
```

The `[voice:closing]` marker switches to the "closing" phase's reference audio for all subsequent speech chunks.

---

## Sentence Length & Structure Guidelines

F5-TTS chunks text at sentence boundaries (`.`, `!`, `?`, `…`) up to 300 characters per chunk. Each chunk is synthesised in a single model call within a 30-second context window.

### Optimal Sentence Structure

- **Ideal length:** 8–20 words per sentence
- **Maximum:** Keep sentences under 200 characters for best quality (300 is the hard limit)
- **Minimum:** Avoid 1–2 word standalone sentences — they produce unreliable audio
- **One idea per sentence** — do not nest complex subordinate clauses

**Too complex (avoid):**
```
As you breathe in slowly and deeply through your nostrils while feeling the cool air enter your body and simultaneously becoming aware of the expansion of your ribcage and the gentle rise of your belly, allow your mind to quiet itself.
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
| Between major sections | Use paragraph break (auto 6.5s) or `[pause:6s]` – `[pause:8s]` |
| Opening / setting the space | `[pause:3s]` – `[pause:5s]` |
| Closing / return to awareness | `[pause:3s]` – `[pause:5s]` |

---

## Anti-Patterns — What NOT to Do

| Anti-Pattern | What Goes Wrong | Fix |
|---|---|---|
| IPA injection `[word](/ˈaɪpə/)` | Spoken as literal text | Remove — F5 does its own G2P |
| ALL CAPS words like "RELAX" | Auto-lowercased, but stylistically avoid | Write "relax" in normal case |
| Digits ("breathe for 4 counts") | F5's G2P may stumble | Write "breathe for four counts" |
| Sentences > 300 chars | Split mid-sentence at arbitrary point | Break into shorter sentences |
| Stage directions in script | Spoken aloud by F5 | Remove all non-spoken text |
| Single newline for pause | Treated as a space, no pause effect | Use `[pause:Xs]` or double newline |
| Very short standalone sentences (1–2 words) | Unreliable audio quality | Attach to adjacent sentence |
| Fast-paced reference audio | Output will be fast despite low speed setting | Re-record reference at meditation pace |
| Reference transcript mismatch | Metallic artefacts, pitch drift, stutter | Ensure transcript is verbatim |
| Very long unbroken paragraphs | Many consecutive chunks without pauses | Break with explicit pauses every 2–3 sentences |

---

## Complete Example Script

This is a well-formed script demonstrating all features for F5-TTS:

```
Welcome to this moment of stillness. [pause:3s]

Allow yourself to arrive, just as you are. [pause:5s]

Find a comfortable position. [pause:2s] Let your body settle into the surface beneath you. [pause:4s]

Gently close your eyes. [pause:5s]

[breath]

Take a slow breath in through your nose... [pause:4s] and release through your mouth. [pause:6s]

[breath]

Again, breathe in... [pause:4s] and let go. [pause:6s]

Notice the weight of your body. [pause:3s] Feel how you are fully supported. [pause:5s]

There is nothing you need to do right now. [pause:4s] Nothing to fix, nothing to change. [pause:6s]

Let any thoughts that arise drift past, like clouds moving slowly across a wide sky. [pause:8s]

[breath]

With each exhale, release a little more. [pause:5s]

You are here. [pause:3s] You are safe. [pause:3s] You are completely at peace. [pause:10s]

When you are ready, begin to deepen your breath. [pause:4s]

Gently bring your awareness back to the room around you. [pause:5s]

Slowly open your eyes. [pause:3s]

Thank you for taking this time for yourself. [pause:3s]
```

**What this script produces (approximately):**
- At speed 0.80, approximately 3–4 minutes of audio
- Each `[pause:Xs]` creates room tone of that exact duration
- `[breath]` markers insert real breath audio samples (1.2s each)
- Automatic 0.8s room tone gaps + 300ms crossfades between speech chunks within the same paragraph
- Paragraph breaks (blank lines) add 6.5s pauses

---

## Audio Output Specification

The F5-TTS engine outputs:
- **Sample rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono (1 channel)
- **Data type:** float32, values in range -1.0 to +1.0
- **Voice activity mask:** Boolean array (True = speaking, False = silence) used for music ducking
- **Mixing sample rate:** 48,000 Hz (upsampled from 24 kHz with high-accuracy sinc resampling)

---

## Duration Estimation

At default speed 0.80, F5-TTS generates approximately 8–10 seconds of audio per 300 characters of text.

Rough rules of thumb:
- **~100–130 words of speech** ≈ 1 minute of audio (before adding pauses)
- Add all explicit pause durations to the speech duration
- `[breath]` markers add ~1.2 seconds each

**Target length examples:**
- 5-minute meditation: ~400–500 words of speech + 90–120s of total explicit pauses
- 10-minute meditation: ~800–1000 words of speech + 150–200s of total explicit pauses
- 20-minute meditation: ~1600–2000 words of speech + 300–400s of total explicit pauses
