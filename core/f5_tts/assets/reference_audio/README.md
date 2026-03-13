# F5-TTS Reference Audio

Place your reference voice recordings here. Each file defines one voice personality.

## File Naming

Use a descriptive **slug** as the filename (lowercase, underscores, no spaces):

```
calm_brittney.wav
deep_marcus.wav
soft_aurora.wav
```

The slug is derived from the filename stem and must exactly match the corresponding
transcript file in `../reference_transcript/`.

## Format Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM, uncompressed) |
| Sample Rate | **24 000 Hz** |
| Channels | **Mono** |
| Duration | **10–12 seconds** (shorter clips reduce cloning fidelity) |
| Bit Depth | 16-bit or 32-bit float |

### Convert an Existing Recording

```bash
# From any format to the correct spec
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 calm_brittney.wav
```

## Recording Guidelines

- **Environment**: Soft-furnished quiet room (bedroom, closet). Avoid hard surfaces.
- **Microphone**: Any clean condenser or USB mic, 20–30 cm from mouth.
- **Tone**: Calm, unhurried, compassionate — the target meditation voice.
- **No background noise**: HVAC, fans, or traffic bleeds into every generated segment.

## Validation

A voice is only available for generation if **both** files exist and the transcript
is non-empty:

```
reference_audio/calm_brittney.wav        ← this file
reference_transcript/calm_brittney.txt  ← matching transcript
```

Run the registry scanner to check which voices are active:

```python
from core.f5_tts import voice_registry
print(voice_registry.scan())
```
