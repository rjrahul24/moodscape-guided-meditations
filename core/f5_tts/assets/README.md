# F5-TTS Reference Audio Assets

This directory stores the default reference audio file used by F5-TTS for zero-shot voice cloning.

## Required File: `ref_meditation.wav`

Place your reference audio file here as `ref_meditation.wav` before using F5-TTS in default mode.

### Format Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM, uncompressed) |
| Sample Rate | 24 000 Hz |
| Channels | Mono |
| Duration | 10–12 seconds (ideal) |
| Bit Depth | 16-bit or 32-bit float |

### Recording Guidelines

- **Environment**: Quiet room with minimal reverb. Avoid rooms with hard surfaces (tile, glass). A bedroom or closet with soft furnishings works well.
- **Microphone**: Any clean condenser or USB mic. Distance ~20–30 cm from mouth.
- **Speaking style**: Calm, unhurried, compassionate — the same tone you want in the meditation. Speak at ~0.75× your normal pace.
- **Content**: Any coherent prose at a meditative tempo. 3–4 natural sentences work well.
- **No background noise**: HVAC, fans, or traffic will bleed into every cloned segment.

### Example Reference Script (verbatim transcript for `ref_meditation.wav`)

```
Allow yourself to settle here, right where you are.
There is nothing to do, nowhere to go.
Simply rest in this moment, breathing gently.
```

If you use this script verbatim as your reference text, copy it exactly into the pipeline's `ref_text` constant in [engine.py](../engine.py).

### Why the Transcript Must Be Verbatim

F5-TTS aligns the reference audio against its transcript to extract prosodic and timbral features. Even a single word mismatch causes the alignment to shift, producing degraded cloning quality (metallic artifacts, pitch drift, stutter).

### Using a Custom Voice at Runtime

Users can upload their own reference clip via the F5-TTS Settings accordion in the Gradio UI. The uploaded file replaces `ref_meditation.wav` for that session only; the bundled default is unaffected.
