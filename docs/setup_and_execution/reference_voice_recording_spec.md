# Reference Voice Recording Spec

## Purpose

F5-TTS clones whatever the reference clip contains: room, noise, pacing and intonation. Clean, well-paced clips give a better clone. Recording a multi-phase set (opening, body, closing) breaks the repeated intonation contour that a single clip imposes on every phrase of a long meditation.

## Capture

- Record at 48 kHz / 24-bit WAV.
- Deliver as 24 kHz mono 16-bit: `ffmpeg -i in.wav -ar 24000 -ac 1 -sample_fmt s16 out.wav`
- Noise floor at or below -60 dBFS, RMS around -20 dBFS, peaks at or below -6 dBFS.

## Mic and room

- Large-diaphragm or dynamic mic, 15-20 cm from the mouth, slightly off-axis, with a pop filter.
- Treated or dead room. No bathrooms or bare walls.

## Clip content

- Length 8-11 s (F5 clips references at 12 s).
- Two complete sentences ending on a falling cadence, with at least 0.5 s of room tone after the last word.
- One natural inhale is allowed at the start, none mid-clip. Avoid dense sibilants.
- Save a verbatim transcript as `assets/speakers/reference_text/<slug>.txt`.

## Phases

| Phase | Pace | Delivery |
|---|---|---|
| opening | about 100 WPM | warm, slightly brighter |
| body | about 90 WPM | neutral calm |
| closing | about 80 WPM | softer, lower, still fully voiced (F5 cannot whisper reliably) |

Record three takes per phase and keep the best. Wire the phases up in `assets/speakers/voices.toml`:

```toml
[my_voice_slug]
default = { ref_audio = "my_voice_body.wav", ref_text = "The exact words in the body clip." }
opening = { ref_audio = "my_voice_opening.wav", ref_text = "The exact words in the opening clip." }
closing = { ref_audio = "my_voice_closing.wav", ref_text = "The exact words in the closing clip." }
```

Switch phases in a hand-written script with `[voice:phase]` markers (for example `[voice:closing]` at a paragraph start).

## How a new voice is assigned

`core/voice_selection.py` measures every clip (speaking rate, median pitch, pitch range, spectral brightness) and caches the result in `var/voice_traits.json`. Lower WPM, darker and lower-pitched voices score `calm`; a wider pitch range scores `expressive`. A voice lands where it scores highest. Override any family in `assets/speakers/voice_map.toml`; see `assets/speakers/voice_map.example.toml`.

## Licensing

Record only voices you have the rights to clone, with written consent from the speaker.
