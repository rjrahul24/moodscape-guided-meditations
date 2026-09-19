<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/f5_tts/preprocessor.py` · `core/f5_tts/engine.py` · `core/f5_tts/voice_registry.py`
**Mode:** Set **Content Type → Sleep Story** in the UI (slows speed, shortens paragraph
pauses, softens the music bed). All controls remain adjustable.
**Script tags:** `[pause:Xs]` (seconds) · `[breath]` · `\n\n` paragraph break
(**1.5s in Sleep Story mode**, vs 3.0s for meditations) · `[voice:phase_name]` multi-phase
**No tone tags:** `[soothing]`, `[dreamy]`, etc. are **not** supported. Pace is the
**Speech Speed** slider; the cloned voice's character is fixed to its reference clip.
**Chunk limit:** ~300 chars (auto-split at sentence boundaries)
**F5 text rules:** colons → commas · ellipses → periods · em/en-dashes → commas ·
hyphens in compounds removed · ALL CAPS → lowercase
**See also:** `docs/model_implementation_guides/f5_tts.md` · `vocal_meditation_f5_instructions.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# MoodScape — Sleep Story Script Instructions for F5-TTS

This document is the authoritative reference for writing **sleep story** vocal scripts for
the F5-TTS engine. Give it to an LLM to produce complete, production-ready prose the app
can synthesize without errors or degraded audio.

A sleep story is **not** a guided meditation. It is a single continuous narrative — a
quiet, sensory journey that drifts and dissolves toward sleep. The crux is the **story**.
There are far fewer pauses than in a meditation; the narration flows for long stretches
and the music sits softly behind it as a soothing add-on.

F5-TTS clones the voice, tone, and pace from a short reference clip. The delivery
character is fixed by that clip — so emotion must live in your **word choices**, not in
any tag.

---

## INPUTS — edit these, then send everything below to the LLM

```
TOPIC:            <what the story is about, 1–3 sentences — a setting, a gentle journey>
VOICE:            <F5 voice slug, selected in the UI, not the script>
TARGET_LENGTH:    <e.g. "about 10 minutes" or "~850-950 words" (~85-95 spoken words/minute)>
OVERALL_TONE:     <e.g. "warm and dreamy", "gentle and grounding">
NOTES (optional): <imagery to include, a motif, anything to include or avoid>
```

---

## How the app processes your script (Sleep Story mode)

```
Raw prose
   │  ▼ core/f5_tts/preprocessor.py — parse_script(content_type="sleep_story")
   │    [pause:Xs] → silence; blank line (\n\n) → 1.5s pause (NOT 3.0s — meditation mode)
   │  ▼ normalize_for_f5(): colons→commas, ellipses→periods, dashes→commas,
   │    compound hyphens removed, ALL CAPS lowercased
   │  ▼ split into ≤~300-char chunks at sentence boundaries
   │  ▼ F5TTS.infer() at the Sleep Story speed, cfg_strength, seed
   ▼ voice_audio (24 kHz mono), 0.4s gap + 300ms crossfade between chunks
```

## Constraints unique to F5 (read carefully)

- **Sentences under ~15 words.** F5 has the shortest comfortable sentence length of the
  engines; long run-ons garble at sleep pace. Two short sentences always beat one long one.
- **Periods and commas are the only reliable pacing punctuation.** Colons, ellipses
  (`...`), em-dashes (`—`) and en-dashes (`–`) are normalized away before synthesis — do
  **not** rely on them for pauses. Use `[pause:Xs]` instead. (This differs from Kokoro,
  where ellipses do create a drift.)
- **No hyphens in compound words** — they cause mispronunciation. Write "wellbeing" not
  "well-being", "goodnight" not "good-night".
- **No ALL CAPS** — they get lowercased / risk letter-by-letter spelling.

## Pacing toolkit

### 1. Speed — handled by the UI
**Content Type → Sleep Story** sets a slightly slower **Speech Speed** preset. Do not
write speed into the script. Leave **Pacing (WPM)** at 0 (natural rhythm) so F5 keeps its
own prosodic timing — fixed WPM flattens expression and is wrong for storytelling.

### 2. Paragraph breaks — your main structural pause
A blank line inserts a **2.0-second** pause in Sleep Story mode. This is your primary
scene breath. Use paragraph breaks between scene shifts so the narrative does not feel
continuous and rushed.

### 3. Explicit pauses — `[pause:Xs]` for restful pacing
Seconds only: `[pause:2s]`, `[pause:3s]`, `[pause:4s]`. Sleep stories require regular pauses
to give the listener space to rest and let the imagery settle:

| Moment | Suggested | Example |
|--------|-----------|---------|
| Scene transition | `[pause:2.5s]`–`[pause:3s]` | `The meadow falls behind you. [pause:3s] A quiet stream appears ahead.` |
| Landing a key image | `[pause:2s]`–`[pause:2.5s]` | `A single candle burns in the window. [pause:2s] Its light is warm and still.` |
| The dissolving end | `[pause:3s]`–`[pause:4s]` | `Drifting now. [pause:3s] Just drifting. [pause:4s]` |

**Frequency:** place an explicit pause roughly every **2–4 sentences**, slightly longer in the final
third. Do not exceed ~4s. Because F5 strips ellipses, `[pause:Xs]` is your *only* way to
create a deliberate beat beyond a sentence boundary.

## Style — slow, sensory, continuous

### Word choice
- **Present tense, second person.** "The air is cool." / "You sit down slowly."
- **Gentle, inviting imagery**, never commanding. Concrete sensory detail: "A warm light
  touches your shoulders," not "You feel relaxed."
- **Emotion lives in the words** — the cloned voice's tone is fixed.
- Spell out numbers ("three", not "3"). No symbols, no markdown, no headings, no speaker
  labels, no emojis.

### Structure — progressive wind-down (weave naturally, don't label)
1. **Arrival** (~15%) — set the scene gently; two or three senses; inviting.
2. **Exploration** (~35%) — move slowly through the setting; each paragraph shifts the
   scene slightly; unhurried imagery.
3. **Settling** (~30%) — pace eases; body awareness (warmth, weight, softness); imagery
   becomes passive — things happen *to* the listener.
4. **Release** (~20%) — almost still; very short sentences and fragments; the story fades
   rather than ends.

### Multi-phase voices (optional, advanced)
If a voice defines multiple phases in `voices.toml`, switch with `[voice:phase_name]` at a
paragraph start (e.g. a warmer "closing" reference for the final third). Most stories need
only the default voice — omit this unless you have a configured multi-phase voice.

### What to avoid
- **Instructions** ("close your eyes", "breathe in") — this is a story, not a meditation.
- **Questions or dialogue** — they wake the mind.
- **Tension, conflict, or a climax** — only drift.
- **Long sentences, colons, ellipses-for-pauses, compound hyphens, ALL CAPS** (see constraints).

## Worked example (style reference — do not copy)

For topic "a quiet forest at twilight", ~2 minutes, tone "warm and still":

```
The path is soft underfoot. Pine needles cushion each step. The air carries something
green and cool. You walk slowly, and there is nowhere to be.

Above you, the last light moves through the branches. It turns the leaves to gold.
[pause:1s] A bird calls once, far away. Then stillness.

You find a clearing. The grass is dry and warm. You sit down slowly, and the earth holds
you. Your shoulders soften. Your hands rest open.

The twilight deepens around you. Everything is gentle now. [pause:1.5s] The trees breathe.
You breathe.

Drifting now. [pause:2s] That is all there is.
```

Notice: very short sentences; present tense; sensory and concrete; only a few short
`[pause:Xs]` beats; no colons or ellipses; progressive wind-down from movement to stillness.

## Before you output — self-check

- [ ] Plain prose only — no markdown, no headings, no speaker labels, no emojis.
- [ ] **No tone tags** — only `[pause:Xs]` (seconds), optional `[breath]`, optional `[voice:phase]`.
- [ ] Sentences under ~15 words; long ideas split across two sentences.
- [ ] No colons, ellipses, em-dashes, compound hyphens, or ALL CAPS (F5 normalizes/mishandles them).
- [ ] Pauses are regular (roughly every 2–4 sentences, 2–4s), deepening near the end.
- [ ] Numbers/symbols spelled out. Present tense, sensory detail; emotion in word choice.
- [ ] Progressive structure: arrival → exploration → settling → release.
- [ ] No instructions, questions, dialogue, or tension.
- [ ] Roughly matches `TARGET_LENGTH` (~85–95 spoken words per minute).

Now output the sleep story, and nothing but the story.
