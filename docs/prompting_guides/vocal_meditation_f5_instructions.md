<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/f5_tts/preprocessor.py` · `core/f5_tts/engine.py` · `core/f5_tts/voice_registry.py`
**Script tags:** `[pause:Xs]` · `[breath]` · `\n\n` paragraph break ({{PARA_PAUSE}}s) · `[voice:phase_name]` for multi-phase (hand-written scripts only)
**Chunk limit:** {{CHUNK_CHARS}} chars (auto-split at sentence boundaries)
**Speed:** {{SPEED}}, set by the Content Type preset (not by the script) · about {{WPM}} words per minute
**F5-specific text rules:** colons → commas · ellipses → periods · em-dashes → commas · hyphens in compounds removed
**Voice assets:** `assets/speakers/*.wav` (24kHz mono, ~10s) + matching `.txt` transcripts
**Multi-phase:** defined in `voices.toml` — opening / body / closing phases with different reference audio
**See also:** `docs/model_implementation_guides/f5_tts.md` · `CLAUDE.md#task-routing-guide`
<!-- ────────────────────────────────────────────────────────────────── -->

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
Each speech block split into ≤{{CHUNK_CHARS}}-character chunks at sentence boundaries
      │
      ▼ core/f5_tts/engine.py — F5Engine.synthesize()
For each "speech" chunk:
  - Normalise text (collapse whitespace, lowercase ALL_CAPS)
  - Select reference audio (always the static reference for consistent voice)
  - Call F5TTS infer() with speed={{SPEED}} (Content Type preset), nfe_step=32, cfg_strength=2.0, seed
  - Trim trailing silence, build activity mask
For each "pause" segment:
  - Insert digital silence
For each "breath" segment:
  - Insert real breath sample
      │
      ▼ Silero VAD — crop trailing non-speech, attenuate interior gaps to 15%
      ▼ Assembly — 0.4s silent gap + 300ms crossfade between speech chunks
      │
      ▼ Concatenate all chunks → voice_audio (float32, 24 kHz, mono)
         + voice_activity mask (True where voice is speaking)
```

**Critical rule:** The app never passes `[pause:Xs]` markers to F5-TTS. The script parser strips all pause markers and converts them to room tone arrays before any TTS call happens.

---

## Script Format Specification

### Pause Syntax

Write `[pause:Xs]`, where `X` is an integer or decimal number of seconds between {{PAUSE_MIN}} and {{PAUSE_MAX}}. Examples:

```
[pause:3s]       ← 3 seconds of silence
[pause:2.5s]     ← 2.5 seconds of silence
[pause:10s]      ← 10 seconds of silence
[pause:{{PAUSE_MIN}}s]     ← the shortest allowed pause
```

The preprocessor also normalises `[3 second pause]` and `[5 sec pause]` to `[pause:3s]` and `[pause:5s]`, but generated scripts must use the `[pause:Xs]` form only: the script validator rejects any other bracket tag.

### Breath Markers

```
[breath]       ← 1.2s natural breath sound
[inhale]       ← 1.5s inhale sound
[exhale]       ← 1.8s exhale sound
```

These insert real breath audio samples.

### Paragraph Breaks

A double newline (`\n\n`) is automatically converted to a `[pause:{{PARA_PAUSE}}s]` marker — a paragraph-level pause. Single newlines are treated as a space within the same speech segment.

**Exception:** If one side of a `\n\n` boundary contains only a pause marker (e.g. `[pause:5s]` on its own line), the paragraph break is treated as formatting and no extra pause is added.

**Next to an explicit pause:** A blank line adds {{PARA_PAUSE}}s only when no `[pause:Xs]` is adjacent to it; when a paragraph ends or begins with an explicit pause, the longer of the two wins (they are not added). So `the body. [pause:6s]` followed by a blank line gives 6 seconds, not 6 + {{PARA_PAUSE}}.

### Voice Phase Switching

```
[voice:phase_name]
```

Switches to a different reference audio phase mid-script. This requires multi-phase voice definitions in `voices.toml`. See the Multi-Phase Voices section below.

**Do not use `[voice:...]` in generated scripts.** The automated script validator accepts only `[pause:Xs]`, `[breath]`, `[inhale]` and `[exhale]`, and rejects any other bracket tag. Phase switching is for hand-written scripts in the Manual tab.

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
| Text expansion | Auto (digits, abbreviations) | Auto (digits, abbreviations, punctuation normalization) |
| Chunk limit | Token-based, one sentence per call | {{CHUNK_CHARS}} characters |
| Paragraph pause | Set by the Content Type preset (longer) | {{PARA_PAUSE}} seconds |
| Inter-sentence gap | 0.8s (1.2s after `...`) | 0.4s silent gap + 300ms crossfade |
| Breath sounds | `[breath]` | `[breath]`, `[inhale]`, `[exhale]` |
| Speed default | Content Type preset | {{SPEED}} (Content Type preset) |
| Pacing default | N/A | Natural rhythm (WPM disabled) |

**Important:** Do not use IPA phoneme injection syntax with F5-TTS. It will not be parsed and will be spoken as literal text.

---

## Writing Style for F5-TTS

F5-TTS performs its own grapheme-to-phoneme conversion from raw text. Write natural, conversational meditation prose.

### Do

- Write in natural prose — spell words as they should be read
- Keep sentences short: {{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}} words is ideal, and never more than {{SENT_MAX}} words or {{CHUNK_CHARS}} characters
- Use explicit `[pause:Xs]` markers for precise timing control
- Use `[breath]` markers between breathing instructions for natural rhythm
- Use `[pause:Xs]` instead of `...` for pauses — ellipses are normalized to periods by the preprocessor
- Use commas for subtle micro-pauses within sentences
- Write one clear idea per sentence

### Do Not

- Do not use IPA injection syntax — F5 ignores it and speaks it literally
- Do not use ALL CAPS for emphasis — the engine auto-lowercases them to prevent letter-by-letter G2P spelling
- Digits are auto-expanded (10 → ten), but writing words is still preferred for clarity
- Do not include stage directions or notes in the script — F5 will attempt to speak them
- Do not write sentences longer than {{CHUNK_CHARS}} characters — they will be split mid-sentence
- Do not rely on single newlines for pauses — they are treated as spaces
- Do not use parenthetical asides — they disrupt F5's prosody model

### Style

{{SLOP_GUIDANCE}}

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

1. Place a 24 kHz mono WAV file in `assets/speakers/your_voice_name.wav`
2. Place a verbatim transcript in `assets/speakers/transcripts/your_voice_name.txt`
3. Restart the app — the voice will appear in the dropdown automatically

The slug is derived from the filename (e.g. `your_voice_name.wav` → slug `your_voice_name`).

---

## Speaking Speed

### Speed — Primary Pacing Control

The **speed** parameter is the primary pacing control. It preserves the model's natural prosodic timing, allowing it to insert natural pauses, emphasis, and rhythm variation. Do not write speed into the script.

Pace is set by the Content Type preset (speed {{SPEED}}); the voice reads at about {{WPM}} words per minute.

### Pacing (WPM) — Optional Fixed Pacing

The **Pacing (WPM)** slider (default: 0 = disabled) provides optional fixed pacing via F5's `fix_duration` mechanism. When set above 0, the engine calculates exact output duration per chunk based on word count and target WPM. Script word budgets assume it is left at 0.

**Recommendation:** Leave WPM at 0 (natural rhythm) for most meditations. Fixed WPM is useful for timed breathing exercises where consistent pacing matters more than expressiveness. Fixed WPM constrains the model's prosodic freedom, which can flatten expression.

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
- Speak at a slow meditation pace, well below conversational speech
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
| **Pacing (WPM)** | 0 | Natural rhythm (recommended). |
| **Speaking Speed** | {{SPEED}} | Set by the Content Type preset. About {{WPM}} words per minute. |
| **Voice Reverb** | 0.15 | Subtle room presence. Increase to 0.25 for spacious feel. |
| **Reverb Space** | Warm Studio | Intimate, short decay — best for meditation. Use Wooden Hall for longer sessions. |
| **Music Ducking** | Content Type preset | Increase the ducking depth if the music overpowers the voice. |
| **Fade In** | Content Type preset | Gentle entry |
| **Fade Out** | Content Type preset | Gradual ending |
| **Output Format** | WAV | Lossless. Use MP3 only for distribution. |

### Music Engine Pairing

F5-TTS works well with any music engine. Recommended pairings:

| Music Engine | Best For |
|---|---|
| **Lyria** | Cloud-generated, high-quality ambient music. Requires API key. |
| **Background Music** | Curated instrumentals — no generation, instant. |

---

## Multi-Phase Voices

A single voice can have multiple **phases** for different sections of a meditation. This allows you to use slightly different vocal qualities for opening vs. closing, or different reference paces for active vs. settling sections.

### Defining Phases in voices.toml

```toml
# assets/speakers/voices.toml

[female_meditative_warm]
default = { audio = "female_meditative_warm.wav", transcript = "female_meditative_warm.txt" }
closing = { audio = "female_meditative_warm_closing.wav", transcript = "female_meditative_warm_closing.txt" }
```

Each phase needs its own reference audio + transcript pair in the assets directories.

### Using Phases in Scripts

In a hand-written script, a line containing only `[voice:closing]` switches to the "closing" phase's reference audio for all subsequent speech chunks. Generated scripts must not use it: the automated script validator rejects every tag except `[pause:Xs]`, `[breath]`, `[inhale]` and `[exhale]`.

### Breaking Monotony with Phases

Because F5-TTS uses the same reference clip for every chunk, long meditations can sound repetitive (same intonation curve repeated). Multi-phase voices solve this by giving the model different acoustic targets.

**Recommended phase types:**

| Phase | Character | Recording tip |
|---|---|---|
| `default` (or `opening`) | Welcoming, warm, inviting | Normal meditation pace, gentle smile |
| `body` | Grounding, steady, present | Slightly deeper register, even pacing |
| `whisper` | Intimate, soft, close | Closer to mic, quieter delivery (not breathy) |
| `closing` | Uplifting, grateful, warm | Slightly brighter tone, gentle energy |

Record each phase as a separate 10–12s WAV clip with the same speaker. Define them in `voices.toml`:
```toml
[female_meditative_warm]
default = { audio = "female_meditative_warm.wav", transcript = "female_meditative_warm.txt" }
body = { audio = "female_meditative_warm_body.wav", transcript = "female_meditative_warm_body.txt" }
whisper = { audio = "female_meditative_warm_whisper.wav", transcript = "female_meditative_warm_whisper.txt" }
closing = { audio = "female_meditative_warm_closing.wav", transcript = "female_meditative_warm_closing.txt" }
```

---

## Sentence Length & Structure Guidelines

F5-TTS chunks text at sentence boundaries (`.`, `!`, `?`, `…`) up to {{CHUNK_CHARS}} characters per chunk. Each chunk is synthesised in a single model call within a 30-second context window.

### Optimal Sentence Structure

- **Ideal length:** {{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}} words per sentence
- **Maximum:** {{SENT_MAX}} words; {{CHUNK_CHARS}} characters is the hard chunk limit
- **Minimum:** Avoid 1–2 word standalone sentences — they produce unreliable audio
- **One idea per sentence** — do not nest complex subordinate clauses

**Too complex (avoid):**
```
As you breathe in slowly and deeply through your nostrils while feeling the cool air enter your body and simultaneously becoming aware of the expansion of your ribcage and the gentle rise of your belly, allow your mind to quiet itself.
```

**Better:**
```
Breathe in through your nose. [pause:4s] Feel the cool air entering your body. [pause:3s] Notice your ribcage expanding. [pause:3s] Allow your mind to become quiet. [pause:5s]
```

### Pause Duration Reference

| Context | Recommended Duration |
|---|---|
| Between short phrases (same breath) | `[pause:2s]` – `[pause:3s]` |
| After a breathing instruction | `[pause:4s]` – `[pause:6s]` |
| After a body awareness cue | `[pause:3s]` – `[pause:5s]` |
| After a visualisation sentence | `[pause:5s]` – `[pause:7s]` |
| Deep stillness / spacious silence | `[pause:8s]` – `[pause:10s]` |
| Between major sections | Use paragraph break (auto {{PARA_PAUSE}}s when no explicit pause is next to it) or `[pause:6s]` – `[pause:8s]` |
| Opening / setting the space | `[pause:3s]` – `[pause:5s]` |
| Closing / return to awareness | `[pause:3s]` – `[pause:5s]` |
| A breath hold, if the practice uses one | Never longer than {{MAX_HOLD}} seconds, and shorter if the brief sets a lower limit |

Every pause must be between {{PAUSE_MIN}} and {{PAUSE_MAX}} seconds.

---

## Anti-Patterns — What NOT to Do

| Anti-Pattern | What Goes Wrong | Fix |
|---|---|---|
| IPA injection `[word](/ˈaɪpə/)` | Spoken as literal text | Remove — F5 does its own G2P |
| ALL CAPS words like "RELAX" | Auto-lowercased, but stylistically avoid | Write "relax" in normal case |
| Digits ("breathe for 4 counts") | Auto-expanded, but words are clearer | Write "breathe for four counts" |
| Sentences > {{CHUNK_CHARS}} chars | Split mid-sentence at arbitrary point | Break into shorter sentences |
| Stage directions in script | Spoken aloud by F5 | Remove all non-spoken text |
| Single newline for pause | Treated as a space, no pause effect | Use `[pause:Xs]` or double newline |
| Very short standalone sentences (1–2 words) | Unreliable audio quality | Attach to adjacent sentence |
| Fast-paced reference audio | Output will be fast even at speed 1.0 | Re-record reference at a slow meditation pace |
| Reference transcript mismatch | Metallic artefacts, pitch drift, stutter | Ensure transcript is verbatim |
| Very long unbroken paragraphs | Many consecutive chunks without pauses | Break with explicit pauses every 2–3 sentences |

---

## Complete Example Script

This is a well-formed script demonstrating all features for F5-TTS:

```
Welcome to these few quiet minutes, just for you. [pause:3s]

You can keep your eyes open or let them close, whichever feels easier, and you can shift your position at any time. [pause:4s]

Let your body settle into whatever is holding you. [pause:3s] Notice where you are touching the chair, the floor, or the bed. [pause:5s]

[breath]

If it feels right, take a slow breath in through your nose. [pause:4s] And let it go through your mouth. [pause:6s]

Notice the weight of your hands. [pause:3s] Notice the air on your face. [pause:5s]

Thoughts may come and go. That is fine. When one pulls you away, you can return to the feeling of your breath. [pause:8s]

[breath]

With each exhale, see if your shoulders can soften a little. [pause:6s]

Rest here for a few breaths. [pause:10s]

When you are ready, let your breathing deepen. [pause:4s] Notice the sounds in the room around you. [pause:4s] Move your fingers and your toes. [pause:3s]

Open your eyes when it suits you. Thank you for taking this time. [pause:3s]
```

**What this script produces (approximately):**
- At the preset speed of {{SPEED}} with a calm reference, approximately 3–4 minutes of audio
- Each `[pause:Xs]` creates silence of that exact duration
- `[breath]` markers insert real breath audio samples (1.2s each)
- Automatic 0.4s silent gaps + 300ms crossfades between speech chunks within the same paragraph
- Every spoken paragraph here ends in an explicit pause, so those blank lines add nothing extra: the longer of the explicit pause and the {{PARA_PAUSE}}s paragraph pause wins. The blank line after each `[breath]` line does add a {{PARA_PAUSE}}s pause, because a breath marker is not a pause marker

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

At the preset speed of {{SPEED}} (natural rhythm mode), the voice reads at about {{WPM}} words per minute.

Rough rules of thumb:
- **About {{WPM}} words of speech** ≈ 1 minute of audio (before adding pauses)
- Add all explicit pause durations to the speech duration
- A paragraph break adds {{PARA_PAUSE}} seconds only when no `[pause:Xs]` is next to it; otherwise the longer of the two counts, once
- `[breath]`, `[inhale]` and `[exhale]` add about 1.2, 1.5 and 1.8 seconds each

Budget words as minutes × {{WPM}}, then add your explicit pauses.
Count only the spoken minutes here: speech plus pauses, paragraph breaks and breath sounds must together land inside the target length.
