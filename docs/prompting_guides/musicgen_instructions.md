# MoodScape — MusicGen Prompt Instructions

This document is the authoritative reference for writing background music prompts for MoodScape. It is intended to be given to an LLM to generate music style prompts that the app can use to generate ambient meditation music via Meta's MusicGen model.

---

## How the App Uses Your Music Prompt

Your prompt is **not passed directly to MusicGen**. Before generation, the app wraps your prompt with genre anchors, conditional descriptors, and negative exclusions in `core/pipeline.py`:

```python
def _enhance_music_prompt(user_prompt: str) -> str:
    # Positive genre anchors — front-loaded for T5 encoder weighting
    genre_anchors = "deep ambient soundscape, evolving synthesizer pads, ethereal reverb"

    # Add contextual descriptors only if the user hasn't already mentioned them
    optional = [
        ("ambient", "ambient"),
        ("reverb", "spacious reverb"),
        ("evolving", "slow evolving"),
        ("warm", "warm"),
    ]
    user_lower = user_prompt.lower()
    extras = [desc for key, desc in optional if key not in user_lower][:2]

    # Negative exclusions trail — keeps positive intent at the front
    negative = "no drums, no percussion, no vocals, no beat, no rhythm, beatless"

    parts = [genre_anchors, user_prompt] + extras + [negative]
    enhanced = ", ".join(parts)

    # Cap at 45 words to stay within MusicGen's effective attention window
    words = enhanced.split()
    if len(words) > 45:
        enhanced = " ".join(words[:45])

    return enhanced
```

Token ordering matters: MusicGen's T5 text encoder weighs earlier tokens more heavily, so positive genre anchors are front-loaded, the user's description sits in the middle, up to 2 conditional descriptors are appended (only if the user hasn't already mentioned them), and negative exclusions trail at the end.

**Critical implication:** The genre anchors already include "deep ambient soundscape", "evolving synthesizer pads", and "ethereal reverb". The negative tail handles "no drums, no percussion, no vocals, beatless". You do not need to repeat these in your prompt. Your prompt's job is to supply **texture, instrument, and mood specifics** that sit between these bookends.

---

## Word Budget

MusicGen's attention mechanism degrades with prompts longer than **~45 words total** (hard-capped in code). The genre anchors are approximately 8 words, the negative exclusions are ~10 words, and up to 2 conditional extras add ~4 words. That leaves a budget of **roughly 20–23 words** for your user-facing prompt.

**Count your words.** A prompt like:

```
Soft felt piano, warm analog synth pads, gentle singing bowls, sustained strings, slow and evolving, spacious atmosphere
```

...is 17 words — within budget.

A prompt like:

```
Deeply resonant tibetan singing bowls with crystal harmonics, warm low-frequency sub-bass drone, gentle shimmering glass harmonica textures floating in expansive reverb, soft cello sustained pad, ambient binaural tones, very slow attack evolving throughout
```

...is 36 words — far too long. MusicGen's attention dilutes, and key constraints like "no drums" start to fail.

**Rule: keep your user prompt under 22 words.**

---

## What the App Actually Generates

### Model

The app uses **`facebook/musicgen-stereo-medium`** (1.5B parameters) as the primary model, with **`facebook/musicgen-small`** (300M) as fallback. The model runs on **CPU only** — MPS (Apple Silicon GPU) is disabled due to AudioCraft's EnCodec ELU activation bug that corrupts audio tensors on Apple Silicon.

```python
MODEL_ID = "facebook/musicgen-stereo-medium"
FALLBACK_MODEL_ID = "facebook/musicgen-small"
# Device is always CPU — MPS causes "broken radio" static via EnCodec ELU corruption
self.model = MusicGen.get_pretrained(MODEL_ID, device="cpu")
```

The stereo-medium model (1.5B params) produces richer, more layered ambient textures than the small model. Stereo output is generated internally, then downmixed to mono at the pipeline boundary.

### Generation Strategy: Sliding Window Continuation

The app uses MusicGen's **continuation API** to generate long-form audio via a sliding window approach:

```python
SEGMENT_DURATION = 30        # Seconds per MusicGen call (hard limit)
CONTEXT_DURATION = 10        # Seconds of audio context for continuation
CROSSFADE_DURATION = 2.0     # Seconds of crossfade at each segment seam
```

**How it works:**
1. **First segment** (30s): Text-only generation from the enhanced prompt
2. **Subsequent segments**: `generate_continuation()` with 10s of audio context from the previous segment, producing new audio that naturally continues the sonic character
3. **Stitching**: Context is stripped, and a 2.0s equal-power cosine crossfade is applied at each segment seam, plus a micro-crossfade pass at zero-crossings for click-free joins
4. **Story mode** (optional): Supply `prompt_stages` — a list of (prompt, duration) tuples — to evolve the text prompt across the timeline, matching meditation phase transitions

**What this means for prompts:**
- The music is **not looped** — each segment is a genuine continuation of the previous one, producing natural evolution
- Prompts that produce **slowly evolving, texture-heavy, non-melodic audio** work best across segment boundaries
- Prompts that introduce **distinct musical events** (a gong strike, an arpeggiated phrase) may occasionally create subtle repetition at segment seams, but far less than loop-based approaches
- Aim for **continuous, homogeneous texture** that evolves gradually

### Music Duration

The music is generated to cover:

```python
music_duration = voice_duration + 15  # pre-roll (4s) + post-roll (8s) + safety margin (3s)
```

The `+15` accounts for 4 seconds of music pre-roll (music plays alone before the narrator begins), 8 seconds of post-roll (music plays after the voice ends for a graceful outro), and a 3-second safety margin.

### Generation Parameters (Fixed in Code)

These parameters are set in `core/music_engine.py` and are not user-configurable:

| Parameter | Value | Effect |
|---|---|---|
| `duration` | 30 seconds | Each segment is 30 seconds; continuation extends to target length |
| `use_sampling` | True | Required — greedy decoding sounds robotic |
| `top_k` | 250 | Token diversity (standard for ambient) |
| `top_p` | 0.0 | Disabled — uses `top_k` exclusively for stability |
| `temperature` | 0.87 | Ambient sweet spot: evolving textures without artifacts |
| `cfg_coef` | 4.5 | Strong prompt adherence — "no drums" is taken seriously |

**Temperature 0.87** is calibrated for the ambient sweet spot (0.85–0.90 range). It produces stable, evolving textures with enough variation to avoid monotony while minimizing artifact risk.

**cfg_coef 4.5** means MusicGen strongly adheres to the prompt's style descriptors. Negative constraints ("no drums", "beatless") are reliably honored at this setting. Going above 6.0 distorts audio.

---

## Prompt Structure

### Formula

```
[Instrument/Source] + [Texture/Character] + [Mood/Atmosphere]
```

Keep it to 2–3 clauses. Each clause should be 4–6 words. Comma-separate them.

### The Ingredients

**Instruments / Sound Sources (pick 1–3):**

| Keyword | What MusicGen Produces |
|---|---|
| `soft felt piano` | Muted, warm piano; pads, not melody |
| `warm analog synth pads` | Slow attack synth texture |
| `tibetan singing bowls` | Metallic resonant tones with long decay |
| `crystal bowls` | Higher, glassier bowl tones |
| `sustained strings` | Slow bow, no vibrato, cello/viola texture |
| `cello pad` | Low, resonant bowed string drone |
| `glass harmonica` | Ethereal high-frequency glass tones |
| `low frequency drone` | Sub-bass hum, grounding, cave-like |
| `shakuhachi flute` | Breathy Japanese flute with long decays |
| `soft piano` | General piano texture (less muted than felt) |
| `warm hum` | Vocal-adjacent drone without words |
| `gong` | Single distant strike, fading reverb tail |
| `bell` | High, delicate bell tone with decay |
| `ambient guitar` | Slow, processed guitar without chord movement |

**Texture / Character Descriptors (pick 2–3):**

| Keyword | Effect |
|---|---|
| `slow evolving` | Gradual changes over the 30-second window |
| `spacious` | Wide sense of empty room around the sound |
| `atmospheric` | Environmental, immersive feel |
| `ethereal` | Light, ghostly, slightly otherworldly |
| `warm` | Lower frequency emphasis, comforting |
| `deep` | Emphasizes low-end resonance |
| `shimmering` | High-frequency subtle movement |
| `soft` | Low attack, gentle dynamics |
| `sustained` | Long note holds without releasing |
| `drone` | Continuous single-note or chord texture |
| `harmonic` | Overtone-rich, resonant |
| `floating` | Sense of weightlessness |

**Mood / Atmosphere Anchors (pick 1–2, optional since genre anchors set the tone):**

| Keyword | Effect |
|---|---|
| `healing` | Warm, welcoming, restorative |
| `meditative` | Slightly ceremonial depth |
| `sleep` | Darker, slower, more hypnotic |
| `serene` | Clean, clear, uncluttered |
| `sacred` | Reverent, slightly ceremonial |
| `introspective` | Quieter, inward quality |
| `weightless` | Buoyant, no heaviness |
| `dreamy` | Soft focus, blurred edges |
| `grounding` | Earth-toned, stable, low-register |

---

## Curated Prompt Presets by Meditation Style

These prompts are tuned to the ~22-word user budget and work within the genre anchors and negative exclusions:

**General / All-Purpose (App Default)**
```
Soft felt piano, warm analog synth pads, gentle singing bowls, sustained strings, slow and evolving, spacious atmosphere
```

**Deep Sleep / Relaxation**
```
Low frequency drone, warm cello pad, deep sub-bass hum, slow and dark, constant volume, very spacious reverb
```

**Mindfulness / Breath Focus**
```
Single sustained cello pad, occasional soft singing bowl, minimalist, vast silence, warm room, grounding
```

**Stress Relief / Anxiety**
```
Warm analog pads, soft tibetan singing bowls, floating atmosphere, slow evolving, healing tones, gentle shimmer
```

**Chakra / Energy Work**
```
Authentic tibetan singing bowls, resonant harmonic overtones, sacred atmosphere, sustained crystal tones, healing frequencies
```

**Morning / Gentle Awakening**
```
Soft morning pads, light shimmering bells, slowly rising warmth, hopeful and gentle, very spacious, slow
```

**Visualization / Journey**
```
Ethereal glass harmonica, shimmering ambient wash, soaring sustained pads, expansive, weightless, dreamy texture
```

**Body Scan / Grounding**
```
Warm deep drone, sustained low strings, earth-toned, stable, grounding, very slow attack, minimal movement
```

**Zen / Clarity**
```
Soft shakuhachi flute with long decay, warm synth drone, occasional distant gong, spacious, introspective, dry reverb
```

---

## How the Final Prompt Looks

For the default preset, the full prompt MusicGen actually receives is:

```
deep ambient soundscape, evolving synthesizer pads, ethereal reverb, Soft felt piano, warm analog synth pads, gentle singing bowls, sustained strings, slow and evolving, spacious atmosphere, slow evolving, no drums, no percussion, no vocals, no beat, no rhythm, beatless
```

Note: The conditional extras ("slow evolving") are added only because the user prompt doesn't already contain "evolving" as a standalone word — the check is substring-based, so "evolving" in the genre anchors doesn't suppress it. "ambient", "reverb", and "warm" are suppressed because they appear in the genre anchors or user prompt. Repetition of key attributes slightly increases adherence.

---

## What NOT to Put in the Prompt — Anti-Patterns

These inputs consistently produce music that does not suit meditation or breaks the ambient quality:

| Anti-Pattern | Why It Fails |
|---|---|
| `upbeat`, `energetic`, `lively`, `dance` | Introduces rhythmic feel even alongside "no drums" |
| `melody line`, `hook`, `catchy`, `piano melody` | Forces melodic structure that disrupts the ambient drone character |
| BPM numbers (`60 BPM`, `72 BPM`) | Model associates BPM with rhythmic structure; undermines "beatless" |
| `4/4`, `waltz`, `groove` | Time signatures introduce pulse even without explicit drums |
| Contradictory terms (`beatless 4/4 groove`) | Model tries to satisfy both, produces poor output |
| Genre labels (`jazz`, `classical`, `pop`, `EDM`) | Introduce genre-typical structures that clash with meditation needs |
| `vocals`, `singing`, `choir`, `voice` | Conflicts with "no vocals" in negative tail — model gets confused |
| More than ~22 words | Attention dilutes; "no drums" reliability drops significantly |
| Specific tuning references (`432Hz`, `528Hz`) | MusicGen does not actually tune to specific Hz values; wastes words |
| `binaural beats` | MusicGen does not generate binaural beats; just wastes words |
| `nature sounds`, `rain`, `ocean waves` | MusicGen occasionally produces these but very unreliably from text |
| `piano`, `harp`, `plucked strings`, `pizzicato`, `marimba` | Attack-heavy transients punch through the mix and draw attention away from the voice |

### Preferred Textures for Meditation

Use these descriptors to produce the most subliminal, non-distracting backgrounds:

| Preferred Term | Result |
|---|---|
| `sine wave`, `pure tone` | Smooth, harmonic-free drone |
| `sub-bass hum` | Deep, felt-not-heard foundation |
| `bowed cello`, `bowed strings` | Warm sustained texture without attack transients |
| `warm hum`, `vocal pad` | Rounded mid-range fill with no rhythmic pulse |
| `soft pad`, `analog pad` | Smooth synthesizer texture without modulation artefacts |

---

## Post-Processing Applied to the Music

After generation, the music passes through the following processing chain in `core/audio_processor.py` before mixing:

**Music FX chain (`make_music_chain()`):**
```
PeakFilter(300 Hz, +2.0 dB, Q=0.7)        — low-end warmth
PeakFilter(5500 Hz, +0.8 dB, Q=0.6)       — clarity/air presence
HighShelfFilter(8000 Hz, -3.0 dB)          — tames MusicGen 8-12 kHz shimmer artifacts
Limiter(-1.0 dBFS)                         — prevents clipping
```

The high-shelf cut at 8 kHz targets MusicGen's autoregressive decoding artifacts — a grainy shimmer in the 8–12 kHz band. The 5500 Hz peak adds perceived openness without amplifying the shimmer. This means your prompt does not need to compensate for this — do not add `bright` or `sparkle` descriptors expecting them to survive; they will be attenuated.

**Mixing behavior:**
- Music is normalized to **-20 LUFS** before mixing
- Music starts **4 seconds before** the narrator begins (pre-roll)
- Music continues **8 seconds after** the voice ends (post-roll / outro)
- Music baseline level is set to **-17 dB**
- During speech, music is **ducked by an additional -21 dB** (default) using an envelope-follower sidechain with 10ms attack, 800ms release, and 25ms lookahead
- During pauses, music returns to the -17 dB baseline with a smooth 800ms release
- Final mix is normalized to **-19 LUFS** (meditation standard)

**What this means for prompts:** The music will often be heard most clearly in the pause sections between narration. Textures that have interesting slow evolution sound better in pauses. Dense, always-present drones work well at all times.

---

## Sample Rate and Continuation Notes

- MusicGen generates at **32,000 Hz** natively
- The output is **resampled to 24,000 Hz** (or 44,100 Hz depending on the mix path) for the pipeline
- Sliding window continuation uses **10s audio context** from the previous segment for seamless extensions
- **2.0s equal-power cosine crossfade** at segment boundaries, plus micro-crossfades at zero-crossings
- Drone and pad textures with **slow attack and no strong transients** produce the most seamless segment joins
- Any distinct "event" (a bell strike, a swell peak) may occasionally create subtle repetition at segment seams

**Best continuation-friendly characteristics:**
- Continuous, non-directional texture
- No crescendo or decrescendo within a 30-second window
- No distinct tonal events at predictable time positions
- Homogeneous spectral content that evolves gradually

---

## Quick Reference

**Word budget:** ~20–22 words in the user prompt
**Structure:** `[Instrument] + [Texture] + [Mood]`
**Do include:** Instrument names, texture adjectives, slow/evolving/spacious
**Do not include:** Genre labels, BPM, rhythm terms, binaural, nature sounds, vocals
**Do not repeat:** no drums, no percussion, no vocals, beatless, ambient, reverb, evolving (already in genre anchors/negatives)
**Best for continuation:** Drones, pads, bowls, sustained strings — avoid single transient events
**Model in use:** `facebook/musicgen-stereo-medium` — supports rich, layered ambient textures
