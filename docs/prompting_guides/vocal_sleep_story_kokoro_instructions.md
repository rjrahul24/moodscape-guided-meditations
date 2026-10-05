<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/kokoro_tts/preprocessor.py` · `core/kokoro_tts/engine.py`
**Mode:** Set **Content Type → Sleep Story** in the UI (slows speed to ~0.75, gives spacious
4.0s paragraph pauses, softens the music bed). All controls remain adjustable.
**Script tags:** `[pause:Xs]` (seconds) · `[breath]` (1.2s) · `\n\n` paragraph break
(**4.0s in Sleep Story mode**, vs 6.5s for meditations)
**No tone tags:** `[soothing]`, `[dreamy]`, etc. are **not** supported — they would be
spoken aloud. Pace is the **Speech Speed** slider; emotion lives in word choice.
**Chunk limit:** 150 tokens (auto-merged/split at sentence boundaries)
**Speed:** clamped to a 0.65 floor; Sleep Story preset is 0.75
**See also:** `docs/model_implementation_guides/kokoro_tts.md` · `vocal_meditation_kokoro_instructions.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# MoodScape — Sleep Story Script Instructions for Kokoro

This document is the authoritative reference for writing **sleep story** vocal scripts
for the Kokoro engine. Give it to an LLM to generate complete, production-ready prose
that the app processes without errors.

A sleep story is **not** a guided meditation. It is a single continuous narrative — a
quiet, sensory journey that drifts, softens, and dissolves toward sleep. The delivery
is unhurried, gentle, and soft, with regular restful pauses letting the imagery linger
while the ambient music bed softly comforts the listener.

The delivery character is calmer and slower than daytime meditation, with deliberate,
spacious pauses throughout the narrative to create an atmosphere of profound rest.

---

## INPUTS — edit these, then send everything below to the LLM

```
TOPIC:            <what the story is about, 1–3 sentences — a setting, a gentle journey.
                   e.g. "A slow walk through a quiet forest at dusk">
VOICE:            <Kokoro voice, e.g. "balanced_calm", "deep_rest" — set in the UI, not the script>
TARGET_LENGTH:    <e.g. "about 10 minutes" or "~850-950 words" (~85-95 spoken words/minute)>
OVERALL_TONE:     <e.g. "deeply peaceful", "warm and cosy", "gently hypnotic">
NOTES (optional): <imagery to include, a motif, a feeling to land on, things to avoid>
```

---

## How the app processes your script (Sleep Story mode)

```
Raw prose
   │  ▼ core/kokoro_tts/preprocessor.py — parse_script(content_type="sleep_story")
   │    [pause:Xs] → silent room-tone; [breath] → breath sample;
   │    blank line (\n\n) → 4.0s pause (spacious scene breath)
   │  ▼ meditation prosody pass (comma/ellipsis phrasing applied automatically)
   │  ▼ token-aware chunking (~100–150 tokens, merged/split at sentence boundaries)
   │  ▼ KokoroEngine.synthesize() at the Sleep Story speed (~0.75)
   ▼ voice_audio (24 kHz mono)
```

**Critical rule (same as meditation):** never write the word "pause" expecting silence.
Use the bracket tag `[pause:Xs]` exactly. The parser strips these to silence *before*
TTS; a malformed tag will be spoken aloud.

## Pacing toolkit

### 1. Speed — handled by the UI
Selecting **Content Type → Sleep Story** sets **Speech Speed ≈ 0.75** (slow, calm,
and relaxing, well below standard speech). You do not write speed into the script.
Leave the slider where the preset puts it unless you want to fine-tune.

### 2. Paragraph breaks — your main structural pause
A blank line between paragraphs inserts a **4.0-second** pause in Sleep Story mode. This
is your primary "scene breath." Use paragraph breaks generously between scene shifts to
give the story room to breathe and let the music softly shine through.

### 3. Explicit pauses — `[pause:Xs]` for restful pacing
Use seconds syntax: `[pause:2s]`, `[pause:3s]`, `[pause:4s]`. Sleep stories require
regular pauses so the narration never feels like a hurried wall of text:

| Moment | Suggested | Example |
|--------|-----------|---------|
| Scene transition | `[pause:3s]`–`[pause:4s]` | `...the meadow falls behind you. [pause:3s] A quiet stream appears ahead.` |
| Landing a sensory image | `[pause:2s]`–`[pause:3s]` | `A single candle burns in the window. [pause:2.5s] Its golden flame is still.` |
| Body or breath awareness | `[pause:3s]`–`[pause:4s]` | `Your shoulders drop a little lower. [pause:3s]` |
| The dissolving end | `[pause:3s]`–`[pause:5s]` | `Drifting now. [pause:4s] Deep into sleep. [pause:5s]` |

**Frequency:** place an explicit pause every **2–4 sentences**, allowing each visual or
feeling to settle before continuing. In the final third (settling and release), increase pause
duration to 3–5 seconds as the story softly dissolves.

### 4. Punctuation — natural rhythm
Kokoro reads punctuation for intonation, and the app's prosody pass adds gentle phrasing.
- **Commas** at every natural clause boundary: "The air is cool, and very still."
- **Ellipses** (`...`) create a gentle drift on Kokoro (~1.2s settle after). Use them for
  trailing-off, especially near the end: "The light fades slowly... softly..."
- **Periods** keep sentences short and clean.

## Style — slow, sensory, continuous

### Sentence structure
- **12–20 words** per sentence; never over 25. Short renders best.
- Simple subject–verb–object syntax. Avoid nested clauses.
- Vary length gently — a few short sentences, then one longer, then short again.
- Fragments are welcome near the end: "So quiet. So still."

### Word choice
- **Present tense, second person.** "The air is cool." / "You sit down slowly."
- **Concrete sensory detail** — texture, temperature, light, sound, scent. Specific and
  physical is calming: "The stones beneath your feet are smooth and cool."
- **Emotion lives in the words**, never in tags — Kokoro's voice character is fixed.
- Spell out numbers ("three", not "3"). No ALL CAPS (spelled letter-by-letter). No emojis,
  no markdown, no headings, no speaker labels.

### Structure — progressive wind-down (weave naturally, don't label)
1. **Arrival** (~15%) — ground the listener in one specific, safe place. Two or three
   senses. Unhurried but not yet drifting.
2. **Exploration** (~35%) — move slowly through the setting; each paragraph shifts the
   scene slightly. Active but gentle imagery: walking, noticing, discovering.
3. **Settling** (~30%) — the pace eases; body awareness appears (warmth, weight,
   softness). Imagery becomes passive — things happen *to* the listener.
4. **Release** (~20%) — almost still. Short sentences, ellipses, slightly more frequent
   pauses. The story doesn't end; it fades.

### What to avoid
- **Instructions** ("close your eyes", "breathe in") — this is a story, not a meditation.
- **Questions or dialogue** — they wake the mind. Stay in narration.
- **Tension, conflict, or a climax** — no plot, only drift.
- **Abstract concepts** — stay concrete and sensory.

## Worked example (style reference — do not copy)

For topic "a quiet lakeside at twilight", ~2 minutes, tone "deeply peaceful":

```
The lake is very still this evening. The water holds the last light of the day, and the
colours are soft, pale gold and the faintest blush of rose. [pause:2.5s] You stand at the
water's edge, and the stones beneath your feet are smooth and cool.

The air is clean here, carrying the scent of pine and wet earth. [pause:2s] A few birds
call to each other across the water, quiet and unhurried sounds that seem to belong to the twilight.

You find a place to sit, where the grass meets the shore. The ground is soft, and it holds
you gently. [pause:3s] From here you can see the far trees reflected in the water, dark and
patient shapes standing perfectly still. [pause:3s]

The light is fading now, and the colours deepen. The gold becomes amber, the rose becomes
a soft grey, and the lake grows quieter. [pause:3.5s]

Everything is settling. The birds have gone silent... the water holds the sky, and the sky
holds nothing but soft, gathering dark. [pause:4s]

And you drift with it, gently... [pause:4s] gently... into sleep. [pause:5s]
```

Notice: unhurried narrative flow with gentle pauses every few sentences; 4-second paragraph breaks
let the music bed breathe; emotion in the imagery; progressive slowing toward the end.

## Before you output — self-check

- [ ] Plain prose only — no markdown, no headings, no speaker labels, no emojis.
- [ ] **No tone tags** — only `[pause:Xs]` (seconds) and optional `[breath]` in brackets.
- [ ] Regular calming pauses (roughly every 2–4 sentences, 2–4s), deepening to 3–5s near the end.
- [ ] Most sentences are 12–20 words; none exceed 25.
- [ ] Numbers/symbols spelled out. No ALL CAPS.
- [ ] Present tense, second person, concrete sensory detail; emotion in word choice.
- [ ] Progressive structure: arrival → exploration → settling → release.
- [ ] No instructions, questions, dialogue, or tension.
- [ ] Roughly matches `TARGET_LENGTH` (~85–95 spoken words per minute).

Now output the sleep story, and nothing but the story.
