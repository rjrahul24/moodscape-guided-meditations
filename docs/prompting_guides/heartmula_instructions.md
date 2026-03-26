<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Engine files:** `core/pipeline.py :: _enhance_heartmula_prompt()`
**Framework:** Eight Pillars — `deep ambient` genre, `60bpm` tempo tag as core anchors
**Tag limit:** 5–6 tags max · comma-separated style tags (not sentences)
**Structural lyrics:** `[inst-medium]` only (suppresses vocal bias) · `[intro-medium]`/`[outro-medium]` for structure
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
| `cfg_scale` | **1.8** | DPO-trained model follows tags at lower CFG. 3.0 over-constrains → repetitive/robotic. |
| `temperature` | **0.75** | More stable token distributions for sustained drones/pads. |
| `top_k` | **30** | Tighter pool prevents repetitive loops in ambient territory. |
| `codec_guidance_scale` | **1.25** | Paper Table 2: 1.25 = "smoother and less harsh." Reduces metallic artifacts. |
| `codec_num_steps` | **12** | Reflow-distilled for 10 steps; 12 gives marginal improvement, 16 was wasteful. |

---

## How the Pipeline Enhances Prompts

The `_enhance_heartmula_prompt()` function in `core/pipeline.py` transforms user input following the Eight Pillars system:

1. **Builds core anchors** (2 tags):
   - Genre: `deep ambient`
   - Tempo: `60bpm`

2. **Appends user's original prompt** (deduplicated against core anchors)

3. **Adds negative constraint floor**: `no drums, instrumental`

4. **Builds structural lyrics** using `[inst-medium]` markers scaled to duration (one per ~20s):
   - ≤90s: `[intro-medium] → [inst-medium] → [outro-medium]`
   - ≤300s: `[intro-medium] → [inst-medium] → [inst-medium] → [outro-medium]`
   - >300s: `[intro-medium] → [inst-medium] → ... → [outro-medium]`

**Example output for user prompt "gentle piano, ethereal":**
```
Tags:    "deep ambient, 60bpm, gentle piano, ethereal, no drums, instrumental"
Lyrics:  "[intro-medium]\n\n[inst-medium]\n\n[outro-medium]"
```
Note: User tags are deduplicated — if the user provides `ambient`, it is not repeated since `deep ambient` already covers it.

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

## Structural Lyrics ([inst-medium] Markers)

HeartMuLa uses structural section markers to guide tonal arc. For meditation, we use `[inst-medium]` instead of `[verse]`/`[bridge]` because:

- `[verse]` and `[bridge]` activate vocal associations in the LM (the model expects sung lyrics after these markers)
- `[inst-medium]` starves the vocal generation module — without phonetic tokens, the LM sustains instrumental pads and atmospheric textures
- The `-medium` suffix provides a moderate energy level appropriate for meditation

| Marker | Purpose |
|--------|---------|
| `[intro-medium]` | Opening arrival — sets the tonal foundation at moderate energy |
| `[inst-medium]` | Sustained instrumental section — no vocal associations |
| `[outro-medium]` | Closing dissolution — gentle fade at moderate energy |

For instrumental meditation, use markers **without text lines** between them:

```
[intro-medium]

[inst-medium]

[inst-medium]

[outro-medium]
```

Fewer sections = less model-induced change = more stillness. The pipeline uses minimal section counts automatically.

---

## Long-Form Generation Strategy

For tracks longer than 4 minutes (240s), the engine automatically:

1. Splits the total duration into segments of ≤240s each
2. Selects a **tonal key anchor** (e.g., "key of D minor") applied to all segments for harmonic continuity
3. Uses **token-level continuation** — the last 250 tokens of segment N are fed as a prefix to segment N+1, providing tonal context across boundaries
4. Assigns structural lyrics per segment position:
   - First: `[intro-medium] + [inst-medium]`
   - Middle: `[inst-medium] + [inst-medium]`
   - Last: `[inst-medium] + [outro-medium]`
5. Joins segments with **STFT crossfade in log-magnitude domain** for seamless spectral transitions (falls back to cosine² if STFT fails)

**Key anchoring** combined with **token-level continuation** provides cross-segment coherence. The key tag constrains harmonic consistency while the token prefix provides tonal context, preventing drift across independently generated segments.

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
