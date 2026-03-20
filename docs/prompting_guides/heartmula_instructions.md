# HeartMuLa Prompting Guide

## Tag Syntax

HeartMuLa uses **comma-separated style tags**, not natural language sentences. Tags are short descriptive phrases that guide the model's musical output.

```
ambient, warm synthesizer pads, gentle drone, slow evolving, peaceful
```

**Do NOT use full sentences:**
```
# BAD: "Create a warm ambient track with gentle synthesizer pads"
# GOOD: "ambient, warm synthesizer pads, gentle drone, slow evolving"
```

## How the Pipeline Enhances Prompts

The `_enhance_heartmula_prompt()` function in `core/pipeline.py` transforms user input:

1. **Prepends meditation base tags** (filtered to avoid duplication):
   `ambient, meditation, calm, relaxing, peaceful, slow`

2. **Appends user's original prompt** verbatim

3. **Adds instrumental marker** if no vocal intent detected:
   `instrumental, no vocals`

4. **Builds structural lyrics** scaled to duration:
   - <=90s: `[intro] → [verse] → [outro]`
   - <=300s: `[intro] → [verse] → [bridge] → [verse] → [outro]`
   - >300s: `[intro] → [verse] → [bridge] → [verse] → [bridge] → [verse] → [outro]`

## Meditation-Appropriate Tags

### Genre Anchors
| Tag | Effect |
|-----|--------|
| `ambient` | Base ambient genre — essential for meditation |
| `new age` | Warm, contemplative textures |
| `meditation` | Model learned associations with meditative content |
| `healing` | Therapeutic, restorative tones |
| `spa` | Gentle, non-intrusive background music |
| `drone` | Sustained, evolving tonal foundations |

### Instrument Tags
| Tag | Effect |
|-----|--------|
| `synthesizer pads` | Warm, evolving electronic textures |
| `piano` | Gentle acoustic warmth |
| `cello` | Deep, rich strings |
| `singing bowls` | Traditional meditation resonance |
| `flute` | Breathy, ethereal woodwind |
| `harp` | Delicate, crystalline tones |
| `chimes` | Soft metallic accents |
| `organ` | Deep, sustained foundation tones |

### Texture Tags
| Tag | Effect |
|-----|--------|
| `slow evolving` | Gradual harmonic shifts |
| `spacious` | Wide stereo image, reverberant |
| `warm` | Low-frequency emphasis, cozy |
| `ethereal` | Light, floating, otherworldly |
| `floating` | Weightless, drift-like movement |
| `sustained` | Long notes, minimal rhythmic activity |
| `deep` | Sub-bass presence, grounding |
| `soft` | Low dynamic, gentle |
| `minimal` | Sparse arrangement |

### Tempo Tags
| Tag | Effect |
|-----|--------|
| `slow` | Below 80 BPM feel |
| `very slow` | Below 60 BPM feel |
| `50 BPM` | Explicit tempo hint |
| `breathing pace` | ~12-15 breaths/min tempo |

## Tags to AVOID

These tags introduce percussion, energy, or vocal elements that are inappropriate for meditation:

| Avoid | Reason |
|-------|--------|
| `upbeat`, `energetic` | Introduces fast rhythms |
| `dance`, `EDM`, `techno` | Electronic dance genres |
| `beat`, `rhythm`, `drums` | Percussion-heavy output |
| `rock`, `pop`, `hip-hop` | Strong rhythm sections |
| `vocals`, `singing` | Voice in the music track |
| `bass drop` | Sudden dynamic changes |
| `intense`, `aggressive` | High-energy characteristics |

## Curated Presets by Meditation Style

### Deep Breathing
```
ambient, warm synthesizer pads, gentle drone, slow evolving,
deep breathing space, sustained tones, 50 BPM, instrumental
```

### Body Scan
```
ambient, soft piano, gentle pads, warm, spacious,
body awareness, flowing, minimal, instrumental
```

### Sleep / Yoga Nidra
```
ambient, deep drone, very slow, dark warmth, floating,
sleep inducing, sustained bass, soft chimes, instrumental
```

### Loving-Kindness
```
ambient, warm harp, gentle piano, peaceful, soft strings,
compassion, heart-centered, flowing, instrumental
```

### Focus / Concentration
```
ambient, minimal, soft static, gentle pulse, clean tones,
neural focus, binaural feel, subtle shimmer, instrumental
```

### Nature / Walking
```
ambient, nature sounds, birdsong, gentle breeze, forest,
flowing water, organic textures, peaceful, instrumental
```

### Morning Awakening
```
ambient, gentle chimes, soft piano, morning light, warm,
awakening, gradual brightening, birds, instrumental
```

## Structural Lyrics (Section Markers)

HeartMuLa uses standard structural section markers to guide the tonal arc:

| Marker | Purpose |
|--------|---------|
| `[intro]` | Opening arrival — sets the tonal foundation |
| `[verse]` | Main body — sustaining ambient texture |
| `[bridge]` | Transitional — subtle harmonic shift |
| `[chorus]` | Emphasis — fuller arrangement (use sparingly for meditation) |
| `[outro]` | Closing dissolution — gentle fade |

For instrumental meditation, use markers **without text lines** between them:

```
[intro]

[verse]

[bridge]

[verse]

[outro]
```

The model interprets these as structural cues for energy and arrangement changes, even without lyrics.

## Long-Form Generation Strategy

For tracks longer than 4 minutes (240s), the engine automatically:

1. Splits the total duration into segments of <=240s each
2. Generates each segment with the **same tags** for style continuity
3. Assigns structural lyrics per segment position:
   - First: `[intro] + [verse]`
   - Middle: `[verse] + [bridge] + [verse]`
   - Last: `[verse] + [outro]`
4. Joins segments with 4-second equal-power cosine crossfades

**Tip:** Keep tags consistent across segments for seamless transitions. The structural lyrics provide natural tonal evolution without needing different tags.

## Story Mode

In story mode, each stage gets its own tags, allowing deliberate tonal evolution:

```python
music_prompt_stages = [
    ("ambient, calm breathing pads, soft sine waves, grounding", 90.0),
    ("ambient, deep sleep drones, very slow, dark warmth", 180.0),
    ("ambient, gentle awakening, morning light, soft chimes", 90.0),
]
```

Each stage is generated independently and crossfaded. Stages longer than 240s are automatically split into sub-segments.
