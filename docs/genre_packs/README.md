# Genre Packs — Field Contract & Prose Guidelines

Genre packs are TOML files that curate meditation-specific creative material and deterministic fields for the pipeline. Packs are validated at load time and read on every generation with no restart, so edits to `docs/genre_packs/` take effect immediately.

## Required Fields

Every pack must include these fields. Validation is strict; missing or malformed fields fail at load:

| Field | Type | Constraints | Consumed by |
|-------|------|-------------|-------------|
| `label` | string | User-facing name, under ~30 chars for UI layout | Gradio UI dropdown, genre_choices() |
| `family` | string | Grouping category (e.g., "Emotional", "Presence", "Breathwork") | UI dropdown grouping, genre_choices() |
| `content_type` | string | Must be a key of `CONTENT_PROFILES` (typically "meditation" or "sleep_story") | Pipeline routing; determines pause/pacing timing via content profiles |
| `music_tags` | [string, ...] | Array of 1–2 music tags from the known vocabulary (see **Tag Vocabulary** below) | Background music filter: conjunctive (both must match), falls back to full pool if no match |
| `pause_ratio` | float | Range 0.0–0.6; ratio of silence to total duration in the session | Preprocessor: scales pause durations during script generation |
| `technique` | string | 1–3 sentences naming a real, named method (not a mood) | Passed to planner model; informs script coherence |
| `arc` | [string, ...] | Ordered 4–6 short phrases describing the session's shape start to finish | Passed to planner; guides the meditation's narrative structure |
| `safety` | string | 2–4 sentences specific to THIS genre, beyond generic safety rules | Passed to planner; critical for high-risk genres (grief, anger, forgiveness, etc.) |
| `banned` | [string, ...] | *Optional.* 3–8 exact phrases a good writer in this genre would never use | Linter: matched as substrings, hard-fail if present |
| `angles` | [table, ...] | Exactly 3 distinct angle tables (see **Angles** below) | Genre rotation: pick_angle() selects one per run to vary repeated sessions |

## Tag Vocabulary

Tags are matched exactly against the known set. **Always use exactly two tags** for music filtering (one measured + one measured, or one measured + one declared).

### Why Exactly Two?

The background music library holds ~20 tracks. Music filtering is **conjunctive** (both tags must match). A third tag almost always matches nothing and silently falls back to the full pool, losing the genre's intended sonic profile. Two tags are the sweet spot: narrow enough to shape the bed, wide enough to find a match reliably.

### Measured Tags (Feature-Derived)

Derived from spectral/temporal analysis of the audio and re-derived when tracks are added:

- **Brightness:** "dark" (centroid < 600 Hz), "warm" (600–1050 Hz), "bright" (> 1050 Hz) — one per track
- **Motion:** "drone" (flux < 0.6), "evolving" (flux > 1.9), or neither
- **Rhythm:** "sparse" (onset < 1.5/s), "busy" (onset > 5/s), or neither — only if flux ≥ 0.6 (silent drone has spurious onsets)
- **Timbre:** "tonal" (spectral flatness < 0.15), "textured" (> 1.5), or neither — one per track
- **Articulation:** "struck" (percussive > 4%), "sustained" (percussive < 1%), or neither — one per track
- **Dynamics:** "steady" (RMS p95/p5 < 2.0), "dynamic" (> 3.3), or neither — one per track

### Declared Tags (Human-Written)

Written by you and never overwritten by analysis:

- "piano", "flute", "strings", "nature", "voice", "bells"

### Tag Thresholds (from `core/background_tags.py`)

These thresholds are calibrated against the 20-track library as of 2026-09-17 and are the measured distribution of the backgrounds. Re-derive with:

```bash
.venv/bin/python scripts/tag_backgrounds.py --report
```

**Brightness thresholds:**
- `CENTROID_DARK_BELOW = 600.0` Hz
- `CENTROID_BRIGHT_ABOVE = 1050.0` Hz
- Mid-range (600–1050 Hz) → "warm"

**Motion thresholds:**
- `FLUX_DRONE_BELOW = 0.6` (spectral flux, mean onset strength)
- `FLUX_EVOLVING_ABOVE = 1.9`
- Mid-range (0.6–1.9) → no motion tag

**Rhythm thresholds (only if flux ≥ 0.6):**
- `ONSET_SPARSE_BELOW = 1.5` onsets per second
- `ONSET_BUSY_ABOVE = 5.0` onsets per second
- Mid-range (1.5–5.0) → no rhythm tag

**Timbre thresholds:**
- `FLATNESS_TONAL_BELOW = 0.15` (spectral flatness × 1000)
- `FLATNESS_TEXTURED_ABOVE = 1.5`
- Mid-range (0.15–1.5) → no timbre tag

**Articulation thresholds:**
- `PERCUSSIVE_STRUCK_ABOVE = 0.040` (fraction of energy, HPSS percussive)
- `PERCUSSIVE_SUSTAINED_BELOW = 0.010`
- Mid-range (1–4%) → no articulation tag

**Dynamics thresholds:**
- `DYNAMICS_STEADY_BELOW = 2.0` (RMS p95 / p5)
- `DYNAMICS_DYNAMIC_ABOVE = 3.3`
- Mid-range (2.0–3.3) → no dynamics tag

## Prose Fields — The Creative Heart

### Technique (1–3 sentences)

Name a real, named method. Do not write "calm yourself" or "feel peaceful." The technique is what the meditation does, not how the listener should feel.

Examples of real techniques:
- Self-compassion break (Kristin Neff)
- RAIN (Recognise, Allow, Investigate, Nurture)
- Urge surfing (Marlatt & Gordon)
- Somatic resourcing (Peter Levine)
- Loving-kindness or metta (traditional)
- Cognitive defusion (ACT)
- Noting or labelling (Vipassana)
- Body scanning (Kabat-Zinn)
- STOP (Stop, Take a breath, Observe, Proceed)

### Arc (4–6 short phrases)

The session's shape from opening to close. Think of it as chapter headings. Use present participles ("arriving", "noticing", "returning") or noun phrases.

Example:
```
"arrival and permission",
"the body's weather",
"one memory, held lightly",
"self-kindness",
"return",
```

### Safety (2–4 sentences)

Genre-specific caveats beyond the generic rules in `content_safety_rules.md`. Anticipate what a bad interpretation might be and head it off.

Examples:

- **Grief & Loss:** "No stage models of grief and no timelines. Never instruct the listener to let go of the person, or to feel better."
- **Forgiveness:** "Never suggest the listener should forgive, owes the person forgiveness, or is failing by withholding it."
- **Loneliness:** "Never suggest that loneliness is the listener's fault or a failure of effort to socialise."
- **Anger & Frustration:** "Anger is a messenger. Never instruct the listener to suppress anger or 'let it go' on command."

### Banned (3–8 phrases)

Exact substring matches that hard-fail the linter. Prefer multi-word phrases over single words: a word like "relax" is common and will catch too many otherwise-fine drafts, forcing expensive repairs. Phrases like "let it go" or "breathe deeply" are better targets.

Phrases are matched as substrings, so "breath holds" will catch "I want you to hold your breath" and "consider holding your breath." Use them to block idioms the model might fall into.

Example:
```
banned = [
    "everything happens for a reason",
    "in a better place",
    "time heals",
    "move on",
    "closure",
    "at least",
]
```

### Angles (exactly 3, each with name and imagery)

An **angle** is a distinct sensory/metaphorical treatment of the genre. Each angle should be different enough that two repeated runs, using different angles, could not produce the same meditation.

Structure:
```toml
[[angles]]
name    = "the empty chair"
imagery = ["a chair by a window", "afternoon light on the floor", "a cup gone cold"]

[[angles]]
name    = "tidal"
imagery = ["a shoreline at dusk", "waves that arrive and withdraw", "wet sand holding a shape"]

[[angles]]
name    = "carrying"
imagery = ["a stone warmed in a pocket", "a coat that still smells of someone", "a path walked many times"]
```

Each angle needs:
- **name** (string): The metaphor or focus; searchable and memorable
- **imagery** (array of 3–4 strings): Concrete sensory details. Short and vivid, not abstract. One image per line.

**The angle-distinctness bar:** Two angles are not "calm water" and "a still lake" (same angle twice). Two angles are "a descending staircase" and "a tide going out" (both invoke descent/change, but through different bodies and rhythms). Distinct angles defend against repetition when the same genre is used multiple times.

**Imagery guidance:**
- Concrete, sensory, present-tense ("light on the floor", not "illumination")
- 3–4 items per angle (brevity keeps them memorable and usable)
- No abstraction ("impermanence" is too abstract; "wet sand holding a shape" is concrete)
- No affirmations ("you are strong" is abstract; "the spine supporting upright" is embodied)

## Validation Checklist

Before commit, ensure:

1. **TOML is valid** — no quotes missing, arrays are `[...]` not bare strings
2. **All required fields present** — label, family, content_type, music_tags, pause_ratio, technique, arc, safety
3. **Types are correct**:
   - Strings in quotes: `label = "Title"`
   - Arrays with brackets: `music_tags = ["tag1", "tag2"]`
   - Numbers without quotes: `pause_ratio = 0.34`
4. **content_type is in CONTENT_PROFILES** — usually "meditation" or "sleep_story"
5. **music_tags exist in MEASURED_VOCAB or DECLARED_VOCAB** — exactly 2 tags
6. **pause_ratio is 0.0–0.6**
7. **arc has 4+ steps**
8. **angles has exactly 3 entries** with name and imagery (3–4 images each)
9. **angle names are unique** — no duplicates
10. **Load test passes:**
    ```bash
    .venv/bin/python -c "from core.genres import load_all_packs; print(len(load_all_packs()), 'packs load')"
    ```

## Testing

Run the full test suite to validate your packs:

```bash
.venv/bin/python -m pytest tests/unit/test_genres.py -v
```

The `ShippedPacksTest` class validates all 46 packs load, have the right structure, and that angles are distinct and sensory.
