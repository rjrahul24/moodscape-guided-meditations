# Checkpoint Locations

Where each model's weights live, and how to obtain them.

| Engine | Backend | Path | How to get |
|--------|---------|------|------------|
| ACE-Step 1.5 | MLX + MPS | `./ACE-Step-1.5/checkpoints/` | Auto-downloaded by `ace-step` package on first run |
| Kokoro-82M | HF hub | `~/.cache/huggingface/` | Auto-downloaded by `kokoro` on first synth |
| F5-TTS | HF hub | `~/.cache/huggingface/` | Auto-downloaded on first synth |
| IndexTTS-2 | HF hub (manual) | `model_checkpoints/indextts2/` | `huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=model_checkpoints/indextts2` |
| HT Demucs | torch hub | `~/.cache/torch/hub/` | Auto-downloaded by `demucs` on first stem-sep call |
| Convolution IRs | local | `assets/impulse_responses/{warm_studio,wooden_hall,stone_chapel}.wav` | Committed to repo |
| F5 voice assets | local | `core/f5_tts/assets/reference_audio/` · `.../reference_transcript/` · `.../voices.toml` | Committed to repo |
| IndexTTS voice assets | local | `reference_audio/vocals/` (speakers) · `reference_audio/instrumental/` (emotions) | User-managed (gitignored) |

## Notes

- All `checkpoints/`, `model_checkpoints/`, and `reference_audio/` are gitignored — they are large and either auto-downloaded or user-supplied.
- ACE-Step paths are relative; the app must be run from the project root.
- For IndexTTS-2, the engine raises a `FileNotFoundError` with the exact CLI command if the checkpoint dir is missing.
