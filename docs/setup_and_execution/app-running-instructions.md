# MoodScape — Running Instructions

This document covers how to start the MoodScape app.

---

## Starting the App

```bash
cd ~/Documents/moodscape-guided-meditations-1
source .venv/bin/activate

python app.py
```

Open the URL shown (typically `http://127.0.0.1:7860`) in your browser.

Select your preferred TTS engine:
- **Kokoro TTS** — fast, lightweight, preset voices (default)
- **Parler TTS** — slower, richer, description-controlled voices

No separate servers or API keys required — both engines run fully locally.

---

## Shutdown

Press `Ctrl+C` in the terminal to stop the app.

---

## Notes

- On first use of Parler TTS, the model will be downloaded from Hugging Face (~4.5 GB for Large, ~1.7 GB for Mini). Subsequent runs use the cached model.
- Both TTS engines and MusicGen are loaded sequentially to minimize memory usage.
- The app requires `espeak-ng` to be installed at the system level (see README for installation instructions).
