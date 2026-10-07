<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/kokoro_tts/preprocessor.py` · `core/kokoro_tts/engine.py`
**Script tags:** `[pause:Xs]` · `[breath]` (1.2s) · `\n\n` paragraph break ({{PARA_PAUSE}}s) · speed {{SPEED}} (Content Type preset), clamped to 0.65–1.0
**Synthesis unit:** per sentence (`split_into_sentences()`) — `merge_sentences_to_chunks()` ({{CHUNK_TOKENS}}-token chunks) exists in the preprocessor but is not on the synthesis path
**Speed floor:** 0.65 — below this Kokoro produces distorted output
**Prosody auto-applied:** comma injection at breath boundaries, sensory ellipses, IPA for Sanskrit/yoga terms
**Voice selection:** 6 presets or custom blend — see `voice_manager.py :: MEDITATION_PRESETS`
**See also:** `docs/model_implementation_guides/kokoro_tts.md` · `CLAUDE.md#task-routing-guide`
<!-- ────────────────────────────────────────────────────────────────── -->

# MoodScape — Vocal Script Instructions for Kokoro TTS

This document is the authoritative reference for writing meditation vocal scripts for MoodScape. It is intended to be given to an LLM to generate complete, production-ready vocal scripts that the app can process without errors or degraded audio quality.

---

## How the App Processes Your Script

The pipeline from raw script text to audio follows this exact sequence:

```
Raw script text
      │
      ▼ core/kokoro_tts/preprocessor.py — parse_script()
List of segments: {"type": "speech", "text": "..."} and {"type": "pause", "duration_sec": N.N}
      │
      ▼ core/kokoro_tts/preprocessor.py — preprocess_for_meditation()
For each speech segment:
  - Expand digits/abbreviations → spoken words
  - Convert formal phrasing → contractions (warmer prosody)
  - Inject IPA phonemes for Sanskrit/yoga terms
  - Enhance prosody punctuation (comma insertion at breath boundaries)
  - Inject contemplative ellipses before sensory words (warmth, peace, etc.)
  - Vary sentence lengths (promote long-clause commas to periods)
      │
      ▼ core/kokoro_tts/engine.py — KokoroEngine.synthesize()
For each "speech" segment:
  - Split into individual sentences at .!? boundaries
  - Merge sentences < 4 words with the next sentence
  - Per sentence: annotate_speed() adjusts pace (short phrases slower, questions slower)
  - Per sentence: add_voice_jitter() for subtle timbre variation
  - Call Kokoro KPipeline for each sentence → float32 audio
  - Insert room-tone pauses between sentences (0.8s / 1.2s after "...")
For each "pause" segment:
  - Insert room-tone noise (bandpass 100–800 Hz, -55 dBFS)
      │
      ▼ Concatenate → spectral gating
      │
      ▼ voice_audio (float32, 24 kHz, mono) + voice_activity mask
```

**Critical rule:** The app never passes `[pause:Xs]` markers to Kokoro. The script parser strips all pause markers and converts them to room-tone audio arrays before any TTS call happens. If you accidentally pass a pause marker string to Kokoro, it will try to pronounce the words "pause" and "3s" — which is never what you want.

---

## Script Format Specification

### The Only Allowed Pause Syntax

Write `[pause:Xs]`, where `X` is an integer or a decimal number (float) of seconds between {{PAUSE_MIN}} and {{PAUSE_MAX}}. Both forms are valid:

```
[pause:3s]       ← integer, 3 seconds of silence
[pause:2.5s]     ← decimal, 2.5 seconds of silence
[pause:10s]      ← integer, 10 seconds of silence
[pause:{{PAUSE_MIN}}s]     ← the shortest allowed pause
```

**Invalid forms that will not be parsed:**
```
[pause: 3s]      ← space before number — NOT parsed
[pause:3]        ← missing "s" — NOT parsed (will be spoken as text)
[Pause:3s]       ← capital P — NOT parsed (case-sensitive)
[pause:3sec]     ← "sec" not accepted — NOT parsed
(pause 3 seconds) ← parentheses — NOT parsed (will be spoken)
```

### Paragraph Breaks as Automatic Pauses

Any double newline (`\n\n`) in the script is automatically converted to a `[pause:{{PARA_PAUSE}}s]` pause before parsing. The {{PARA_PAUSE}}-second duration comes from the Content Type preset and was chosen for spacious pacing — it provides a generous breath between major script sections. Single newlines have no special effect — they are treated as a space within the same speech segment. Use explicit `[pause:Xs]` markers if you need a shorter or longer inter-paragraph pause.

```
This is one sentence on a single line. [pause:3s]      ← explicit 3-second pause

This is a new paragraph.                               ← blank line above → auto {{PARA_PAUSE}}s pause
```

### Consecutive Pause Merging

The parser **merges consecutive pauses** into one. If two pause markers appear back-to-back (or a paragraph break falls next to an explicit pause), their durations are summed:

```
[pause:3s] [pause:2s]    →  becomes a single 5-second pause
[pause:3s]               }
                         }  blank line adds a further {{PARA_PAUSE}}s,
Next sentence            }  so the total is longer than 3s
```

This means you should not rely on stacking small pauses; be explicit with a single adequate duration.

### Automatic Number & Abbreviation Expansion

The `core/kokoro_tts/preprocessor.py` module now automatically expands numbers (0–999) and common abbreviations before TTS inference:

- `4` → `four`, `120` → `one hundred and twenty`
- `sec` → `seconds`, `min` → `minutes`, `Hz` → `hertz`
- `e.g` → `for example`, `i.e` → `that is`

This means you can safely write `"breathe in for 4 seconds"` and the TTS engine will receive `"breathe in for four seconds"`. For numbers above 999, continue to spell them out manually.

---

## Available Voices

These are the exact voice IDs supported by the app (from `app.py`). The voice is passed as a string to Kokoro's `voice=` parameter.

| Display Label in UI | Voice ID to Use | Character |
|---|---|---|
| Heart + Nicole blend (meditation) | `af_heart,af_nicole` | Warm + calm ASMR blend — **default, best for meditation** |
| Heart — Female | `af_heart` | Warm, velvety, intimate |
| Nicole — Female (calm/ASMR) | `af_nicole` | Soft, ASMR-like, very calm |
| Bella — Female | `af_bella` | Warm, expressive |
| Sarah — Female | `af_sarah` | Natural, clear |
| Sky — Female | `af_sky` | Airy, gentle |
| Nova — Female (intimate) | `af_nova` | Smooth, intimate |
| Adam — Male | `am_adam` | Deep, grounded |
| Michael — Male | `am_michael` | Resonant |

All voices use `lang_code='a'` (American English). The model loaded is `hexgrad/Kokoro-82M` with `trf=True` (transformer-based G2P for highest phonemization quality).

### Voice Blending

A comma-separated string in the voice field creates a blended voice in Kokoro. The app passes the string directly to Kokoro's `voice=` parameter. The blend is equal-weight.

```
"af_heart,af_nicole"   ← blends af_heart and af_nicole equally
```

Only comma-separated pairs of American English voices work reliably. Do not attempt to blend across American and British voice prefixes.

---

## Speaking Speed

The `speed` slider in the UI ranges from **0.5 to 1.0**. Its value, **{{SPEED}}**, is set by the Content Type preset, not by the script. Do not write speed into the script.

The app internally clamps speed to a **minimum of 0.65** regardless of the slider value:

```python
speed = max(speed, 0.65)
```

Below 0.65, Kokoro produces distorted, robotic-sounding output.

Pace is set by the Content Type preset (speed {{SPEED}}); the voice reads at about {{WPM}} words per minute. On top of that, `annotate_speed()` slows individual sentences: short phrases (under six words), questions and sentences that trail off with an ellipsis are spoken a little slower, never below the 0.65 floor.

---

## How Kokoro Processes Each Speech Segment

Understanding this is essential for writing scripts that produce good audio.

### Sentence Splitting

Each `"speech"` segment (text between pause markers) is split into individual sentences by the regex:

```
(?<=[.!?])\s+
```

This splits **after** `.` `!` or `?` followed by whitespace. The sentence boundaries are:
- `.` (period) — falling intonation
- `!` (exclamation) — rising/excited intonation
- `?` (question) — rising intonation

Each sentence is then synthesized as a separate Kokoro call. This is intentional — it gives Kokoro enough context per call for natural prosody and avoids the 510-token limit.

### Short Sentence Merging

Sentences with **fewer than 4 words** are merged with the next sentence. This prevents Kokoro from generating poor-quality audio on very short utterances.

```
"Breathe in."        ← 2 words → merged with next sentence
"Hold for four counts."  → becomes "Breathe in. Hold for four counts."
```

**Implication for script writers:** Avoid standalone very short sentences like "Good." or "Yes." or "Now." as isolated lines. If you want a short affirmation, either:
1. Attach it to the following sentence, OR
2. Follow it immediately with a pause: `"Good. [pause:2s]"` — here the pause comes after, so they are in the same speech segment and Kokoro synthesizes the whole thing together before the silence.

### Inter-Sentence Pauses (Automatic)

The engine automatically inserts **silence between sentences within a single speech segment**:

- **0.8 seconds** between regular sentences ending in `.` `!` `?`
- **1.2 seconds** after sentences ending with `...` (ellipsis) or `…` (Unicode ellipsis U+2026)

These pauses are **in addition to** any explicit `[pause:Xs]` markers. They are inserted within a paragraph, not between paragraphs.

**Example of what actually gets generated:**

```
Script:  "Notice the weight of your body. Feel your breath rise and fall."

Rendered as:
  [audio: "Notice the weight of your body."]
  [0.8s silence ← automatic inter-sentence]
  [audio: "Feel your breath rise and fall."]
```

The 1.2s ellipsis pause only occurs when both sentences are in the **same speech segment** (no pause marker between them). A `[pause:Xs]` marker splits the segment, so the explicit pause replaces the automatic one. Prefer explicit `[pause:Xs]` markers over ellipses for timing: they are exact and they count toward the duration budget.

### Punctuation as Prosody Control

Kokoro's StyleTTS2 architecture interprets punctuation to shape intonation. Use these deliberately:

| Punctuation | Effect | Best Use in Meditation |
|---|---|---|
| `.` period | Falling intonation, definitive close | End of most sentences |
| `...` or `…` | Trailing, drifting quality; adds 1.2s auto-pause before next sentence | Inhale cues, dreamlike transitions, incomplete thoughts |
| `;` semicolon | Brief connected pause with anticipation | Linking related thoughts with a slight breath |
| `—` em dash | Rhythmic break mid-sentence | Natural speech rhythm variation |
| `?` question mark | Rising intonation | Use very sparingly; questions disrupt the meditative hypnotic effect |
| `!` exclamation | Energetic/rising | Avoid in meditation scripts |
| `,` comma | Very slight natural breath | Between clauses; subtle |

---

## Script Writing Rules

### Structural Template

A well-formed meditation script follows this arc:

```
1. Opening / grounding (set the space, 30–60 seconds)
2. Body settling cues (arrive in the body, 60–90 seconds)
3. Breathing instruction (anchor to breath, with timed pauses, 90–180 seconds)
4. Core meditation content (visualization, body scan, mantra, etc.)
5. Gradual closing (return to awareness, 30–60 seconds)
```

### Sentence Length Guidelines

- **Ideal sentence length:** {{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}} words
- **Maximum:** {{SENT_MAX}} words
- **Minimum:** 4+ words per sentence (shorter will be merged with the next)
- **Do not write complex nested subordinate clauses.** Break them into separate sentences with pauses.

**Too complex (avoid):**
```
As you breathe in and feel the air moving through your nostrils into your lungs while simultaneously becoming aware of the expansion of your ribcage, allow your mind to become quiet.
```

**Better:**
```
Breathe in slowly. [pause:3s] Feel the air entering your nostrils. [pause:2s] Notice your ribcage expand. [pause:3s] Allow your mind to become quiet. [pause:5s]
```

### Pause Duration Reference

| Context | Recommended Duration |
|---|---|
| Between short phrases (same breath) | `[pause:2s]` – `[pause:3s]` |
| After a breathing instruction (inhale cue) | `[pause:4s]` |
| After a breathing instruction (exhale cue) | `[pause:4s]` – `[pause:6s]` |
| After a body awareness instruction | `[pause:3s]` – `[pause:5s]` |
| After a visualization sentence | `[pause:5s]` |
| Deep stillness / spacious silence | `[pause:8s]` – `[pause:10s]` |
| Between major sections | `[pause:5s]` – `[pause:8s]` |
| Opening / opening the space | `[pause:3s]` – `[pause:5s]` |
| Closing / return to awareness | `[pause:3s]` – `[pause:5s]` |
| A breath hold, if the practice uses one | Never longer than {{MAX_HOLD}} seconds, and shorter if the brief sets a lower limit |

Every pause must be between {{PAUSE_MIN}} and {{PAUSE_MAX}} seconds.

Note: the automatic 0.8s inter-sentence pause is added on top of these. A `[pause:3s]` after a sentence ending in `.` will result in approximately 3.8 seconds of silence total if there is speech on both sides.

### Language Style for TTS

Kokoro performs best with:

- **Short, clear sentences** — one idea per sentence
- **Active voice** — "Notice your breath" not "Your breath should be noticed"
- **Simple vocabulary** — avoid medical or technical terms
- **Present tense and imperatives** — "Feel", "Notice", "Allow", "Let", "Breathe"
- **Rhythmic repetition** — Kokoro handles repeated phrases with natural variation ("Breathe in. Breathe out.")
- **Avoid parenthetical asides** — they disrupt prosody

### Style

{{SLOP_GUIDANCE}}

### Sanskrit and Foreign-Language Pronunciation

Kokoro uses `misaki` G2P library with espeak-ng fallback. For Sanskrit meditation terms, spell them phonetically in a way that guides correct pronunciation, or use IPA injection syntax:

```
[pranayama](/pɹɑːnɑːˈjɑːmə/)   ← IPA injection (misaki syntax)
[ujjayi](/uːˈdʒɑːji/)
[chakra](/tʃɑːkɹə/)
[savasana](/ʃɑːˈvɑːsənə/)
```

**Generated scripts must not use IPA injection.** The automated script validator accepts only `[pause:Xs]`, `[breath]`, `[inhale]` and `[exhale]`, and rejects every other bracket tag, including `[word](/ipa/)`. IPA injection is for hand-written scripts in the Manual tab.

If you do not use IPA injection, common Sanskrit terms that Kokoro typically mispronounces include: _pranayama_, _ujjayi_, _savasana_, _namaste_, _chakra_. Spell these phonetically if exact pronunciation matters: e.g. write "pra-nah-yama". In generated scripts, phonetic spelling is the only option.

---

## What NOT to Do — Anti-Patterns

| Anti-Pattern | What Goes Wrong | Fix |
|---|---|---|
| `[pause: 3s]` with a space | Not parsed; spoken as text "pause 3s" | Use `[pause:3s]` exactly |
| `(pause 3 seconds)` in parentheses | Spoken aloud by Kokoro | Use `[pause:3s]` |
| Writing very short standalone sentences < 4 words | Merged with next sentence, changing intended pacing | Attach to adjacent sentence or ensure the merged result is still natural |
| Stacking consecutive pauses | They merge — the total may not match your intent | Use a single pause with the total duration |
| Very long sentences (>{{SENT_MAX}} words) | Unnatural delivery | Break into shorter sentences |
| Putting stage directions in the script | Kokoro reads them aloud | Remove all stage directions; put only spoken text in the script |
| Relying on line breaks for pauses | Single `\n` has no pause effect; only `\n\n` (blank line) adds {{PARA_PAUSE}}s | Use explicit `[pause:Xs]` markers for precise control |
| Going below speed 0.65 | Distorted, robotic audio | Leave speed at the Content Type preset ({{SPEED}}); the app clamps at 0.65 |
| Numbers and symbols | Kokoro may or may not expand them correctly | Spell out: "four" not "4", "fifty percent" not "50%", "minus" not "−" |

---

## Complete Example Script

This is a well-formed script demonstrating all features correctly:

```
Welcome. This is a few quiet minutes, just for you. [pause:3s]

You can keep your eyes open or let them close, whichever feels easier, and you can shift your position at any time. [pause:4s]

Let your body settle into whatever is holding you. [pause:3s] Notice where you are touching the chair, the floor, or the bed. [pause:5s]

If it feels right, take a slow breath in through your nose. [pause:4s] And let it go through your mouth. [pause:6s]

Notice the weight of your hands. [pause:3s] Notice the air on your face. [pause:5s]

Thoughts may come and go, and that is fine. When one pulls you away, you can return to the feeling of your breath. [pause:8s]

With each exhale, see if your shoulders can soften a little. [pause:6s]

Rest here for a few breaths. [pause:10s]

When you are ready, let your breathing deepen. [pause:4s] Notice the sounds in the room around you. [pause:4s]

Open your eyes when it suits you. Thank you for taking this time. [pause:3s]
```

**What this script produces (approximately):**
- Total duration depends on the speaking speed setting
- At the preset speed of {{SPEED}}, this script is approximately 3–4 minutes of audio
- Each `[pause:Xs]` creates exact silence of that duration at 24 kHz
- Automatic 0.8s gaps appear between sentences within each paragraph

---

## Audio Output Specification

The TTS engine outputs:
- **Sample rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono (1 channel)
- **Data type:** float32, values in range −1.0 to +1.0
- **Voice activity mask:** Boolean array (True = speaking, False = silence) used for music ducking

---

## Duration Estimation

A rough rule of thumb at the preset speed of {{SPEED}}:
- **About {{WPM}} words of speech** ≈ 1 minute of audio (before adding pauses)
- Add all explicit pause durations (in seconds) to the speech duration
- Each paragraph break adds {{PARA_PAUSE}} seconds, and each sentence adds about 0.8 seconds of automatic gap

Budget words as minutes × {{WPM}}, then add your explicit pauses.
Count only the spoken minutes here: speech plus pauses, paragraph breaks and automatic gaps must together land inside the target length.
