<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/kokoro_tts/preprocessor.py` · `core/kokoro_tts/engine.py`
**Mode:** Set **Content Type → Sleep Story** in the UI (speed {{SPEED}}, spacious
{{PARA_PAUSE}}s paragraph pauses, softer music bed). All controls remain adjustable.
**Script tags:** `[pause:Xs]` (seconds) · `[breath]` (1.2s) · `\n\n` paragraph break
(**{{PARA_PAUSE}}s in Sleep Story mode**)
**No tone tags:** `[soothing]`, `[dreamy]`, etc. are **not** supported — they would be
spoken aloud. Pace is the **Speech Speed** slider; emotion lives in word choice.
**Synthesis unit:** one sentence per Kokoro call (the {{CHUNK_TOKENS}}-token chunker exists in the preprocessor but is not on the synthesis path)
**Speed:** {{SPEED}}, set by the Content Type preset; clamped to a 0.65 floor
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
TARGET_LENGTH:    <e.g. "about 10 minutes" (the voice reads about {{WPM}} words per minute)>
OVERALL_TONE:     <e.g. "deeply peaceful", "warm and cosy", "gently hypnotic">
NOTES (optional): <imagery to include, a motif, a feeling to land on, things to avoid>
```

---

## How the app processes your script (Sleep Story mode)

```
Raw prose
   │  ▼ core/kokoro_tts/preprocessor.py — parse_script(content_type="sleep_story")
   │    pause markers → silent room-tone; [breath] → breath sample;
   │    blank line (\n\n) → {{PARA_PAUSE}}s pause (spacious scene breath)
   │  ▼ meditation prosody pass (comma/ellipsis phrasing applied automatically)
   │  ▼ split into sentences; one Kokoro call per sentence, 0.8s gap between sentences
   │  ▼ KokoroEngine.synthesize() at the Sleep Story speed ({{SPEED}})
   ▼ voice_audio (24 kHz mono)
```

**Critical rule (same as meditation):** never write the word "pause" expecting silence.
Use the bracket tag `[pause:Xs]` exactly. The parser strips these to silence *before*
TTS; a malformed tag will be spoken aloud.

## Pacing toolkit

### 1. Speed — handled by the UI
Pace is set by the Content Type preset (speed {{SPEED}}); the voice reads at about {{WPM}}
words per minute. Short phrases and trailing sentences are slowed a little further,
never below the 0.65 floor. You do not write speed into the script.
Leave the slider where the preset puts it unless you want to fine-tune.

### 2. Paragraph breaks — your main structural pause
A blank line between paragraphs inserts a **{{PARA_PAUSE}}-second** pause in Sleep Story mode. A blank line adds {{PARA_PAUSE}}s only when no `[pause:Xs]` is adjacent to it; when a paragraph ends or begins with an explicit pause, the explicit pause stands and the paragraph pause is dropped. This
is your primary "scene breath." Use paragraph breaks generously between scene shifts to
give the story room to breathe and let the music softly shine through.

### 3. Explicit pauses — `[pause:Xs]` for restful pacing
Use seconds syntax: `[pause:2s]`, `[pause:3s]`, `[pause:5s]`. Sleep stories need regular
pauses so the narration never feels like a hurried wall of text, and the pauses grow as
the story winds down:

1.5–2.5 s in the first third, 2–4 s in the middle, 4–8 s in the final third, never more than 10 s; place a pause every 2–4 sentences, lengthening toward the end.

### 4. Punctuation — natural rhythm
Kokoro reads punctuation for intonation, and the app's prosody pass adds gentle phrasing.
- **Commas** at every natural clause boundary: "The air is cool, and very still."
- **Pauses:** prefer an explicit `[pause:Xs]` for any deliberate pause; it is exact and
  counts toward the duration budget. Ellipses (`...`) are rarely, if ever, needed:
  "The light fades slowly. [pause:4s] The water is still."
- **Periods** keep sentences short and clean.

## Writing — slow, sensory, continuous

### Sentence structure
- **{{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}} words** per sentence; never over {{SENT_MAX}}. Short renders best.
- Simple subject–verb–object syntax. Avoid nested clauses.
- Vary length gently — a few short sentences, then one longer, then short again.
- Fragments are welcome near the end: "So quiet. So still."

### Style

{{SLOP_GUIDANCE}}

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
4. **Release** (~20%) — almost still. Short sentences, fragments, longer and slightly
   more frequent pauses. The story doesn't end; it fades.

### What to avoid
- **Instructions** ("close your eyes", "breathe in") — this is a story, not a meditation.
- **Questions or dialogue** — they wake the mind. Stay in narration.
- **Tension, conflict, or a climax** — no plot, only drift.
- **Abstract concepts** — stay concrete and sensory.

## Worked example (style reference — do not copy)

For topic "a quiet lakeside at twilight", ~2 minutes, tone "deeply peaceful":

```
The lake is very still this evening. The water holds the last light of the day, and the
colours are soft, pale gold and the faintest blush of rose. [pause:2s] You stand at the
water's edge, and the stones beneath your feet are smooth and cool.

The air is clean here, carrying the scent of pine and wet earth. [pause:2s] A few birds
call to each other across the water, quiet sounds that seem to belong to the twilight.

You find a place to sit, where the grass meets the shore. The ground is soft, and it holds
you. [pause:3s] From here you can see the far trees reflected in the water, dark and
patient shapes standing perfectly still. [pause:3s]

The light is fading now, and the colours deepen. The gold becomes amber, the rose becomes
a soft grey, and the lake grows quieter. [pause:4s]

Everything is settling. The birds have gone silent. The water holds the sky, and the sky
holds nothing but the gathering dark. [pause:5s]

And you drift with it, slow and easy. [pause:6s] Down into sleep. [pause:8s]
```

Notice: unhurried narrative flow with pauses every few sentences that lengthen toward the end;
{{PARA_PAUSE}}-second paragraph breaks (where no explicit pause ends the paragraph) let the music bed breathe; no ellipses; emotion in the imagery; progressive slowing toward the end.

## Before you output — self-check

- [ ] Plain prose only — no markdown, no headings, no speaker labels, no emojis.
- [ ] **No tone tags** — only `[pause:Xs]` (seconds) and optional `[breath]` in brackets.
- [ ] Pauses every 2–4 sentences, growing from 1.5–2.5s early to 4–8s in the final third, none over 10s.
- [ ] Most sentences are {{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}} words; none exceed {{SENT_MAX}}.
- [ ] Numbers/symbols spelled out. No ALL CAPS.
- [ ] Present tense, second person, concrete sensory detail; emotion in word choice.
- [ ] Progressive structure: arrival → exploration → settling → release.
- [ ] No instructions, questions, dialogue, or tension.
- [ ] Roughly matches `TARGET_LENGTH`: budget words as minutes × {{WPM}}, then add your explicit pauses.

Now output the sleep story, and nothing but the story.
