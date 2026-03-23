<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/pipeline.py :: _enhance_heartmula_prompt()`
**Framework:** Eight Pillars — Genre (95%) > Timbre (50%) > Mood (32%) > Instrument (25%) > Scene
**Tag limit:** 7–9 tags max · comma-separated style tags (not sentences)
**Structural lyrics:** `[interlude]` only (suppresses vocal bias) · `[intro]`/`[outro]` for structure
**Auto-appended:** `no drums, instrumental`
**Duration scaling:** ≤90s → "extremely slow" · ≤300s → "long soft pads that barely move" · >300s → "soundscape stays flat"
**See also:** `docs/model_implementation_guides/heartmula.md` · `CLAUDE.md#task-routing-guide`
<!-- ────────────────────────────────────────────────────────────────── -->

# HeartMuLa Prompting Guide

## Tag Syntax

HeartMuLa uses **comma-separated style tags**, not natural language sentences. Tags are short descriptive phrases that guide the model's musical output.

```
Ambient, Warm, Relaxing, Synthesizer
```

**Do NOT use full sentences:**
```
# BAD: "Create a warm ambient track with gentle synthesizer pads"
# GOOD: "Ambient, Warm, Relaxing, Synthesizer"
```

---

## The Eight Pillars Tag System

HeartMuLa's DPO training maps tags into eight categorical pillars, each with a different influence weight on the generated output. Tags from higher-influence pillars dominate the acoustic character.

| Pillar | Influence Weight | Optimal Meditation Selections | Notes |
|--------|-----------------|------------------------------|-------|
| **GENRE** | 95% (mandatory) | Ambient, New Age, Drone, Classical | Primary architectural anchor — defines the fundamental acoustic rules |
| **TIMBRE** | 50% | Warm, Soft, Dark, Ethereal, Smooth | Textural quality and frequency envelope character |
| **MOOD** | 32% | Relaxing, Peaceful, Meditation, Calm | Emotional/psychological resonance |
| **INSTRUMENT** | 25% | Synthesizer, Piano, Strings, Bowl | Dominant sound source guiding the Global Transformer |
| **SCENE** | 20% | Study, Cinematic, Nature | Intended listening context |
| **GENDER** | 37% | *Omit for instrumental* | Only relevant if vocals are desired |
| **REGION** | 12% | *Omit entirely* | Cultural harmonic influences — omit for neutral drone |
| **TOPIC** | 10% | Healing, Self-discovery, Hope | Thematic essence — lowest influence |

### The "Less is More" Principle

**CRITICAL:** Overloading prompts with many tags causes *probability interference* — the cross-attention mechanism's probability distribution fractures, producing muddy, generic output that oscillates between styles. One strong, unified anchor is far more effective than many competing signals.

**Optimal tag count: 5-7 tags total.** The pipeline's `_enhance_heartmula_prompt()` enforces this automatically.

---

## Generation Parameters

HeartMuLa has **no API knobs for tempo or softness** — all control is through tags. Three LM sampling parameters control how faithfully the model follows conditioning:

| Parameter | Value | Effect on Meditation Music |
|-----------|-------|---------------------------|
| `cfg_scale` | **3.0** | Classifier-free guidance strength. Higher = stricter adherence to tags. Default (1.5) was too loose — LM wandered from ambient. 3.0 provides moderate adherence. Values above 4.0 risk early EOS (truncated output) and mode collapse. |
| `temperature` | **0.9** | Token sampling temperature. Lower = more predictable, grounded output. Default (1.0) was slightly too random for sleep music. 0.9 preserves tonal variety while reducing drift. |
| `top_k` | **45** | Sampling pool size. Moderate restriction from default (50) keeps the LM in ambient territory without starving token diversity (which causes repetitive loops). |
| `codec guidance_scale` | **1.5** | HeartCodec flow-matching guidance during detokenization. Slightly higher than default (1.25) improves fidelity of sustained pads. |
| `codec num_steps` | **16** | Codec diffusion steps (was 10). More steps = cleaner reconstruction, less quantization noise. |

**Key insight:** `cfg_scale` is the most impactful parameter but must be used conservatively. The heartlib default (1.5) is too loose, but values above 4.0 cause the autoregressive LM to predict end-of-sequence tokens early, producing truncated output. 3.0 is a safe sweet spot that improves tag adherence without over-constraining.

---

## How the Pipeline Enhances Prompts

The `_enhance_heartmula_prompt()` function in `core/pipeline.py` transforms user input following the Eight Pillars system:

1. **Builds core anchors** from the four highest-influence pillars (filtered to avoid duplicating user tags):
   - Genre: `Ambient`
   - Timbre: `Warm`
   - Mood: `Relaxing`
   - Instrument: `Synthesizer`

2. **Adds a temporal descriptor** scaled to duration:
   - ≤90s: `extremely slow`
   - ≤300s: `long soft pads that barely move`
   - >300s: `soundscape stays flat`

3. **Appends user's original prompt** verbatim

4. **Adds negative constraint floor**: `no drums, instrumental`

5. **Builds structural lyrics** using `[interlude]` markers:
   - ≤90s: `[intro] → [interlude] → [outro]`
   - ≤300s: `[intro] → [interlude] → [interlude] → [outro]`
   - >300s: `[intro] → [interlude] → [interlude] → [interlude] → [outro]`

**Example output for user prompt "gentle piano, ethereal":**
```
Tags:    "Ambient, Relaxing, extremely slow, gentle piano, ethereal, no drums, instrumental"
Lyrics:  "[intro]\n\n[interlude]\n\n[outro]"
```
Note: `Warm` and `Synthesizer` defaults were skipped because the user provided `ethereal` (timbre) and `piano` (instrument).

---

## Temporal Descriptors

Explicit pacing language the model learned during DPO training. These descriptors set the internal clock for how slowly the track progresses:

| Descriptor | Effect |
|------------|--------|
| `extremely slow` | Below 50 BPM feel, minimal movement |
| `long soft pads that barely move` | Sustained textures with glacial evolution |
| `soundscape stays flat` | Static, non-evolving ambient plane |
| `matches 4-count breathing` | ~12-15 breaths/min pacing |
| `gentle pulses` | Soft, rhythmic breathing cue without percussion |
| `50 BPM` | Heart-rate entrainment, deep relaxation |
| `60 BPM` | Resting heart rate, standard meditation tempo |

---

## Meditation-Appropriate Tags

### Genre Anchors (95% influence)
| Tag | Effect |
|-----|--------|
| `Ambient` | Base ambient genre — essential for meditation |
| `New Age` | Warm, contemplative textures |
| `Drone` | Sustained, evolving tonal foundations |
| `Classical` | Gentle acoustic warmth |

### Timbre Tags (50% influence)
| Tag | Effect |
|-----|--------|
| `Warm` | Low-frequency emphasis, cozy |
| `Soft` | Low dynamic, gentle |
| `Dark` | Deep, subdued textures |
| `Ethereal` | Light, floating, otherworldly |
| `Smooth` | Even, frictionless textures |

### Mood Tags (32% influence)
| Tag | Effect |
|-----|--------|
| `Relaxing` | Primary relaxation anchor |
| `Peaceful` | Calm, tranquil |
| `Meditation` | Model learned associations with meditative content |
| `Calm` | Steady, undisturbed |

### Instrument Tags (25% influence)
| Tag | Effect |
|-----|--------|
| `Synthesizer` | Warm, evolving electronic textures |
| `Piano` | Gentle acoustic warmth |
| `Strings` / `Cello` | Deep, rich string resonance |
| `Singing Bowls` | Traditional meditation resonance |
| `Flute` | Breathy, ethereal woodwind |
| `Harp` / `Chimes` | Delicate, crystalline tones |
| `Organ` | Deep, sustained foundation |

### Frequency / Resonance Tags
| Tag | Effect |
|-----|--------|
| `432Hz` | Shifts attention toward drone-heavy, resonant output |
| `#binauralbeats` | Encourages binaural beat-like tonal content |

---

## Tags to AVOID

These tags introduce percussion, energy, or vocal elements inappropriate for meditation:

| Avoid | Reason |
|-------|--------|
| `upbeat`, `energetic` | Triggers fast rhythms and high-energy arrangements |
| `dance`, `EDM`, `techno` | Electronic dance genres include 4/4 kick patterns |
| `beat`, `rhythm`, `drums` | Positive percussion signals — use `no drums` instead |
| `rock`, `pop`, `hip-hop` | Strong rhythm section associations |
| `vocals`, `singing` | Puts voice in the music track |
| `bass drop` | Sudden dynamic changes |
| `intense`, `aggressive` | High-energy characteristics |
| `uplifting`, `euphoric` | Often generates builds and drops |

---

## Curated Presets by Meditation Style

All presets follow the Eight Pillars system (5-7 tags, pillar-ordered):

### Sleep / Deep Relaxation (recommended starting point)
```
Ambient, Soft, Relaxing, sustained pads, soundscape stays flat, no drums, instrumental
```

### Deep Breathing
```
Ambient, Warm, Peaceful, Synthesizer, matches 4-count breathing, no drums, instrumental
```

### Body Scan
```
Ambient, Soft, Calm, Piano, long soft pads that barely move, no drums, instrumental
```

### Sleep / Yoga Nidra
```
Drone, Dark, Relaxing, deep bass, extremely slow, no drums, instrumental
```

### Loving-Kindness
```
Ambient, Warm, Peaceful, Harp, gentle pulses, no drums, instrumental
```

### Focus / Concentration
```
Ambient, Smooth, Calm, Synthesizer, 432Hz, no drums, instrumental
```

### Nature / Walking
```
Ambient, Ethereal, Peaceful, Flute, nature sounds, no drums, instrumental
```

### Morning Awakening
```
New Age, Warm, Peaceful, Chimes, gentle pulses, no drums, instrumental
```

---

## Structural Lyrics ([interlude] Markers)

HeartMuLa uses structural section markers to guide tonal arc. For meditation, we use `[interlude]` instead of `[verse]`/`[bridge]` because:

- `[verse]` and `[bridge]` activate vocal associations in the LM (the model expects sung lyrics after these markers)
- `[interlude]` starves the vocal generation module — without phonetic tokens, the LM sustains instrumental pads and atmospheric textures

| Marker | Purpose |
|--------|---------|
| `[intro]` | Opening arrival — sets the tonal foundation |
| `[interlude]` | Sustained instrumental section — no vocal associations |
| `[outro]` | Closing dissolution — gentle fade |

For instrumental meditation, use markers **without text lines** between them:

```
[intro]

[interlude]

[interlude]

[outro]
```

Fewer sections = less model-induced change = more stillness. The pipeline uses minimal section counts automatically.

---

## Long-Form Generation Strategy

For tracks longer than 4 minutes (240s), the engine automatically:

1. Splits the total duration into segments of ≤240s each
2. Selects a **tonal key anchor** (e.g., "key of D minor") applied to all segments for harmonic continuity
3. Assigns structural lyrics per segment position:
   - First: `[intro] + [interlude]`
   - Middle: `[interlude] + [interlude]`
   - Last: `[interlude] + [outro]`
4. Joins segments with **8-second** equal-power cosine crossfades (longer overlap masks tonal differences between independently generated segments)

**Key anchoring** is the primary coherence mechanism for long-form tracks. Since the heartlib API does not support latent context feedback between segments, appending a musical key tag (e.g., "key of A minor") to all segments constrains the LM to generate in a consistent key, preventing the most jarring form of harmonic drift.

---

## Story Mode

In story mode, each stage gets its own tags, allowing deliberate tonal evolution:

```python
music_prompt_stages = [
    ("Ambient, Warm, Calm, Synthesizer, extremely slow, no drums, instrumental", 90.0),
    ("Drone, Dark, Relaxing, deep bass, soundscape stays flat, no drums, instrumental", 180.0),
    ("Ambient, Soft, Peaceful, Chimes, gentle pulses, no drums, instrumental", 90.0),
]
```

Each stage is generated independently and crossfaded. Stages longer than 240s are automatically split into sub-segments with key anchoring.

**Important:** Always include `no drums, instrumental` in each stage's tags — they are not inherited between stages.
