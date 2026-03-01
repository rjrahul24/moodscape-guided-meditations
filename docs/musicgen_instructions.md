# MoodScape — MusicGen Prompt Instructions

This document is the authoritative reference for writing background music prompts for MoodScape. It is intended to be given to an LLM to generate music style prompts that the app can use to generate ambient meditation music via Meta's MusicGen model.

---

## How the App Uses Your Music Prompt

Your prompt is **not passed directly to MusicGen**. Before generation, the app wraps your prompt with fixed prefix and suffix strings in `core/pipeline.py`:

```python
def _enhance_music_prompt(user_prompt: str) -> str:
    prefix = (
        "warm ambient soundscape, slow evolving synth pads, spacious reverb, "
        "beatless, no drums, no percussion, no vocals, "
    )
    suffix = ", peaceful, calm, gentle"
    return prefix + user_prompt + suffix
```

The final prompt passed to MusicGen is:

```
warm ambient soundscape, slow evolving synth pads, spacious reverb, beatless, no drums, no percussion, no vocals, [YOUR PROMPT HERE], peaceful, calm, gentle
```

**Critical implication:** The prefix already includes the most important structural constraints (no drums, no percussion, no vocals, beatless). You do not need to repeat these in your prompt. The suffix already adds peaceful/calm/gentle mood anchors. Your prompt's job is to supply **texture, instrument, and mood specifics** that sit between these bookends.

---

## Word Budget

MusicGen's attention mechanism degrades with prompts longer than **~40 words total**. The fixed prefix is approximately 16 words and the fixed suffix is 3 words. That leaves a budget of **roughly 18–21 words** for your user-facing prompt.

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

**Rule: keep your user prompt under 20 words.**

---

## What the App Actually Generates

### Model

The app uses **`facebook/musicgen-small`** (300M parameters). This is the fastest model and runs on CPU. It is sufficient for ambient drone and pad textures that sit behind narration.

```python
self.model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
```

Note: The research guide in `docs/musicgen_meditation_guide.md` recommends `musicgen-medium` for higher quality, but the current implementation uses `musicgen-small`. Prompts should be calibrated to `musicgen-small`'s capability — simpler, drone-like textures work better than complex multi-layered arrangements.

### Generation Strategy: Single Segment + Looping

The app does **not** use MusicGen's continuation API. It generates **one 30-second segment** and then loops it to fill the required duration.

```python
SEGMENT_DURATION = 30      # Only one MusicGen call is made
LOOP_CROSSFADE = 5         # Equal-power crossfade at loop boundaries (seconds)
```

**What this means for prompts:**
- The music will repeat every ~30 seconds with a 5-second crossfade at the loop point
- Prompts that produce **slowly evolving, texture-heavy, non-melodic audio** loop most naturally — there are no melodic phrases that restart obviously
- Prompts that introduce **distinct musical events** (a gong strike at second 15, an arpeggiated phrase) will create audible repetition
- Aim for **continuous, homogeneous texture** that could plausibly continue indefinitely

### Music Duration

The music is generated to cover:

```python
music_duration = voice_duration + 10  # extra for pre-roll + fade-out
```

The `+10` accounts for a 2-second music pre-roll (music plays alone before the narrator begins) and the fade-out tail. The looping handles extending the 30-second segment to whatever length is needed.

### Generation Parameters (Fixed in Code)

These parameters are set in `core/music_engine.py` and are not user-configurable:

| Parameter | Value | Effect |
|---|---|---|
| `duration` | 30 seconds | One 30-second segment is generated |
| `use_sampling` | True | Required — greedy decoding sounds robotic |
| `top_k` | 250 | Token diversity (standard for ambient) |
| `top_p` | 0.0 | Disabled — uses `top_k` exclusively for stability |
| `temperature` | 0.8 | Balanced: evolving textures without artifacts |
| `cfg_coef` | 4.0 | Strong prompt adherence — "no drums" is taken seriously |

**Temperature 0.8** is intentionally conservative. It produces stable, predictable ambient textures with minimal jarring transitions. Higher temperatures introduce more variation but also more artifact risk.

**cfg_coef 4.0** means MusicGen strongly adheres to the prompt's style descriptors. Negative constraints ("no drums", "beatless") are reliably honored at this setting. Going above 6.0 distorts audio.

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
| `warm analog synth pads` | Slow attack synth texture (already in prefix) |
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
| `gentle` | Already in suffix — do not repeat |
| `floating` | Sense of weightlessness |

**Mood / Atmosphere Anchors (pick 1–2, optional since suffix adds calm/peaceful):**

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

These prompts are tuned to the ~18-word user budget and work within the fixed prefix/suffix:

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
warm ambient soundscape, slow evolving synth pads, spacious reverb, beatless, no drums, no percussion, no vocals, Soft felt piano, warm analog synth pads, gentle singing bowls, sustained strings, slow and evolving, spacious atmosphere, peaceful, calm, gentle
```

Note that "warm analog synth pads" and "spacious" appear in both the prefix and the user prompt in the default case — this is fine and actually reinforces the descriptor. Repetition of key attributes slightly increases adherence.

---

## What NOT to Put in the Prompt — Anti-Patterns

These inputs consistently produce music that does not suit meditation or breaks the ambient quality:

| Anti-Pattern | Why It Fails |
|---|---|
| `upbeat`, `energetic`, `lively`, `dance` | Introduces rhythmic feel even alongside "no drums" |
| `melody line`, `hook`, `catchy`, `piano melody` | Forces melodic structure that repeats obviously at loop points |
| BPM numbers (`60 BPM`, `72 BPM`) | Model associates BPM with rhythmic structure; undermines "beatless" |
| `4/4`, `waltz`, `groove` | Time signatures introduce pulse even without explicit drums |
| Contradictory terms (`beatless 4/4 groove`) | Model tries to satisfy both, produces poor output |
| Genre labels (`jazz`, `classical`, `pop`, `EDM`) | Introduce genre-typical structures that clash with meditation needs |
| `vocals`, `singing`, `choir`, `voice` | Conflicts with "no vocals" in prefix — model gets confused |
| More than ~20 words | Attention dilutes; "no drums" reliability drops significantly |
| Specific tuning references (`432Hz`, `528Hz`) | MusicGen does not actually tune to specific Hz values; wastes words |
| `binaural beats` | MusicGen does not generate binaural beats; just wastes words |
| `nature sounds`, `rain`, `ocean waves` | MusicGen occasionally produces these but very unreliably from text |

---

## Post-Processing Applied to the Music

After generation, the music passes through the following processing chain in `core/audio_processor.py` before mixing:

**Music FX chain (applied after generation):**
```
LowShelfFilter(cutoff=200 Hz, gain=+1.5 dB)   ← adds warmth
HighShelfFilter(cutoff=8000 Hz, gain=−3.0 dB)  ← tames digital shimmer
Limiter(threshold=−1.0 dB)                     ← prevents clipping
```

The high-shelf cut at 8 kHz is specifically designed to tame MusicGen's tendency to produce "digital shimmer" in the upper frequencies. This means your prompt does not need to compensate for this — do not add `bright` or `sparkle` descriptors expecting them to survive; they will be attenuated.

**Mixing behavior:**
- Music starts **2 seconds before** the narrator begins (pre-roll)
- Music is set to a **−3 dB baseline** level relative to the voice
- During speech, music is **ducked by an additional −4 dB** (default slider value) using a smooth Butterworth envelope follower (~300ms attack, ~800ms recovery)
- During explicit pauses, music returns to the −3 dB baseline level

**What this means for prompts:** The music will often be heard most clearly in the pause sections. Textures that have interesting slow evolution sound better in pauses. Dense, always-present drones work well at all times.

---

## Sample Rate and Looping Notes

- MusicGen generates at **32,000 Hz** natively
- The output is **resampled to 24,000 Hz** for the pipeline
- The 30-second segment is looped with **5-second equal-power crossfade** at boundaries
- Equal-power crossfade (sqrt curves) maintains constant perceived loudness at the loop point
- Drone and pad textures with **slow attack and no strong transients** are nearly undetectable at loop boundaries
- Any distinct "event" (a bell strike, a swell peak) will be audible repeating every 25–30 seconds after the crossfade region

**Best loop-friendly characteristics:**
- Continuous, non-directional texture
- No crescendo or decrescendo (the music is looped, so a fade-in would restart)
- No distinct tonal events at predictable time positions
- Homogeneous spectral content throughout the 30 seconds

---

## Quick Reference

**Word budget:** ~18–21 words in the user prompt
**Structure:** `[Instrument] + [Texture] + [Mood]`
**Do include:** Instrument names, texture adjectives, slow/evolving/spacious
**Do not include:** Genre labels, BPM, rhythm terms, binaural, nature sounds, vocals
**Do not repeat:** no drums, no percussion, no vocals, beatless, calm, peaceful (already in prefix/suffix)
**Best for looping:** Drones, pads, bowls, sustained strings — avoid single transient events
**Model in use:** `facebook/musicgen-small` — prefer simple textures over complex arrangements
