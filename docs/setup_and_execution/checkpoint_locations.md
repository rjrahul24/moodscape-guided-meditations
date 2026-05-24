# Checkpoint Locations

Where each model's weights live, and how to obtain them.

| Engine | Backend | Path | How to get |
|--------|---------|------|------------|
| ACE-Step 1.5 | MLX + MPS | `./models/acestep/checkpoints/` | Auto-downloaded by `ace-step` package on first run |
| Kokoro-82M | HF hub | `~/.cache/huggingface/` | Auto-downloaded by `kokoro` on first synth |
| F5-TTS | HF hub | `~/.cache/huggingface/` | Auto-downloaded on first synth |
| IndexTTS-2 | HF hub (manual) | `models/indextts2/` | `huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=models/indextts2` |
| HT Demucs | torch hub | `~/.cache/torch/hub/` | Auto-downloaded by `demucs` on first stem-sep call |
| Convolution IRs | local | `assets/impulse_responses/{warm_studio,wooden_hall,stone_chapel}.wav` | Committed to repo |
| Breath samples | local | `assets/breath_sounds/{breath,inhale,exhale}.wav` | Committed to repo |
| F5 voice assets | local | `assets/speakers/*.wav` + `assets/speakers/transcripts/*.txt` + `assets/speakers/voices.toml` | Committed to repo |
| IndexTTS voice assets | local | `assets/speakers/*.wav` (speakers) + `assets/emotions/*.wav` (emotions) | Committed to repo |

## Notes

- **`models/` is gitignored** — large weights, either auto-downloaded by the engine or fetched once with `huggingface-cli`.
- **`assets/` is tracked** — curated runtime assets (IRs, breath samples, speaker/emotion references) ship with the repo.
- The global `*.wav` / `*.mp3` ignore rules apply everywhere _except_ `assets/**` (negation rule in `.gitignore`), so generated meditation output never gets committed.
- ACE-Step paths are relative; the app must be run from the project root.
- For IndexTTS-2, the engine raises a `FileNotFoundError` with the exact CLI command if `models/indextts2/` is missing.
