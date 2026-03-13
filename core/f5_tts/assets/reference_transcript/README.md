# F5-TTS Reference Transcripts

Place verbatim transcripts here — one `.txt` file per voice slug.

## File Naming

The filename stem must **exactly match** the corresponding WAV in `../reference_audio/`:

```
calm_brittney.txt   ↔   calm_brittney.wav
deep_marcus.txt     ↔   deep_marcus.wav
```

## Content Requirements

The transcript must be the **verbatim spoken text** of the reference recording.
F5-TTS performs forced character-level alignment between the audio and transcript.
Even a single word mismatch shifts the alignment window, producing:

- Metallic or robotic artefacts
- Pitch drift mid-sentence
- Stutter at chunk boundaries

**Rules:**
1. Type exactly what the speaker says, including filler words ("um", "ah")
2. Match punctuation as spoken (commas for micro-pauses, periods for longer breaks)
3. Do not add phonetic spelling or IPA annotations
4. Plain UTF-8 text, no markdown, no line breaks required

## Example

```
Allow yourself to settle here, right where you are.
There is nothing to do, nowhere to go.
Simply rest in this moment, breathing gently.
```

## Encoding

Save as UTF-8 (no BOM). Most editors default to this on macOS and Linux.
