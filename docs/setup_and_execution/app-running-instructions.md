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

No separate servers or API keys required — the app runs fully locally.

---

## Shutdown

Press `Ctrl+C` in the terminal to stop the app.

---

## Notes

- Kokoro TTS and MusicGen are loaded sequentially to minimize memory usage.
- The app requires `espeak-ng` to be installed at the system level (see README for installation instructions).
