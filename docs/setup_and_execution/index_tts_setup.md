# IndexTTS-2 Installation & Setup

**Status**: ✅ **Fully installed and configured**

This document summarizes the IndexTTS-2 integration for MoodScape Guided Meditations.

## Installation Summary

### 1. Dependencies ✅
All required pip packages have been installed:
- `indextts` (from `git+https://github.com/index-tts/index-tts.git`)
- Supporting dependencies: `numpy`, `soundfile`, `torch`, `transformers`, etc.
- See `requirements.txt` for the git dependency
- **Pacing (optional, recommended):** `brew install rubberband` enables high-fidelity
  pitch-preserving time-stretch for meditation pacing (`INDEXTTS_PACE_RATE`). Without the
  `rubberband` CLI the engine falls back to librosa's phase-vocoder (lower fidelity).

### 2. Model Weights ✅
IndexTTS-2 model checkpoints are present:
- Location: `./models/indextts2/`
- Contents: Full model including GPT, acoustic features, vocoder, BPE tokenizer
- No additional downloads required — weights were pre-downloaded

### 3. Reference Audio ✅
Default reference audio samples have been created for immediate testing:

#### Speaker Voices (`assets/speakers/`)
| Voice | Description |
|-------|-------------|
| `calm_guide.wav` | Gentle, calm meditation voice |
| `warm_teacher.wav` | Warm, nurturing tone |
| `peaceful_voice.wav` | Peaceful, serene voice |

#### Emotion Presets (`assets/emotions/`)
| Emotion | Description |
|---------|-------------|
| `calm.wav` | Peaceful, restful emotion |
| `warm.wav` | Nurturing, loving-kindness emotion |
| `soothing.wav` | Gentle, sleep-inducing emotion |
| `energetic.wav` | Uplifting, energetic emotion |

**Note**: The default samples are synthetic (generated from formant patterns). Replace them with real meditation voice recordings (5–12 seconds, 24 kHz mono) for production use.

## Usage in the App

### Voice Selection
IndexTTS-2 is now available as a TTS engine in the Gradio UI. Voices are auto-discovered from `assets/speakers/`:
```python
from core.index_tts.engine import IndexTTSEngine

# Load a voice by slug
engine = IndexTTSEngine(voice_slug="calm_guide")
audio, sr = engine.synthesize(
    text="Begin your meditation...",
    emotion_slug="calm"  # Optional emotion control
)
```

### Adding Custom Voices

1. **Record a meditation voice clip**:
   - Duration: 5–12 seconds
   - Content: 2–3 complete meditation sentences
   - Pace: ~100–120 WPM (meditation tempo)
   - Tone: warm, calm, compassionate

2. **Convert to 24 kHz mono WAV**:
   ```bash
   ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 my_voice.wav
   ```

3. **Place in `assets/speakers/`**:
   ```
   assets/speakers/my_voice.wav
   ```

4. **Restart the app** — the voice will auto-appear in the UI dropdown.

### Adding Custom Emotions

1. **Record a 3–10 second emotion reference clip** (instrumental or voice)
2. **Convert to 24 kHz mono WAV**
3. **Place in `assets/emotions/`**:
   ```
   assets/emotions/my_emotion.wav
   ```

## Technical Details

### Engine Configuration
| Parameter | Value | Reason |
|-----------|-------|--------|
| `use_fp16` | `False` | Float32 required for MPS stability (bf16 causes NaN errors) |
| `use_deepspeed` | `False` | DeepSpeed is CUDA-only; disabled on Apple Silicon |
| `use_cuda_kernel` | `False` | CUDA kernel unsupported on MPS; CPU fallback used |
| `generation_mode` | `"free"` | Natural prosodic timing for meditation (uncontrolled) |

### Audio Contracts
| Property | Value |
|----------|-------|
| Output sample rate | 24,000 Hz |
| Mix sample rate | 48,000 Hz (upsampled in pipeline) |
| Channels | Mono |
| Bit depth | 32-bit float (pipeline), 16-bit PCM (WAV export) |

### File Locations
```
project_root/
├── models/indextts2/          — Model weights (no change needed)
├── assets/
│   ├── vocals/                           — Speaker references (add custom voices here)
│   └── instrumental/                     — Emotion references (add custom emotions here)
├── core/index_tts/
│   ├── engine.py                         — IndexTTSEngine wrapper
│   ├── preprocessor.py                   — Text normalization
│   ├── postprocessor.py                  — Audio post-processing (mastering, FX)
│   └── voice_registry.py                 — Voice & emotion asset scanning
└── docs/prompting_guides/vocal_indextts_instructions.md  — Prompt engineering guide
```

## Testing

Verify installation with:
```bash
source .venv/bin/activate
python -c "
from core.index_tts.voice_registry import scan_voices, scan_emotions
from core.index_tts.engine import IndexTTSEngine

voices = scan_voices()
emotions = scan_emotions()
print(f'Voices: {list(voices.keys())}')
print(f'Emotions: {list(emotions.keys())}')

engine = IndexTTSEngine(voice_slug='calm_guide')
print('✓ Engine initialized successfully')
"
```

## Common Issues

### Issue: "No IndexTTS-2 voices found"
- **Cause**: `assets/speakers/` is empty or doesn't exist
- **Fix**: Run the reference audio generation script or add custom `.wav` files to `assets/speakers/`

### Issue: Engine initialization fails with FileNotFoundError
- **Cause**: The `voice_slug` parameter doesn't match any file in `assets/speakers/`
- **Fix**: Check available voices with `scan_voices()` and use a valid slug

### Issue: Audio generation produces NaN or metallic artifacts
- **Cause**: `use_fp16=True` on Apple Silicon MPS
- **Fix**: Ensure `engine.py` has `use_fp16=False` and `PYTORCH_ENABLE_MPS_FALLBACK=1` is set

## Integration with Pipeline

The `MeditationPipeline` can use IndexTTS-2:
```python
from core.pipeline import MeditationPipeline

pipeline = MeditationPipeline(
    tts_engine="index_tts",
    tts_kwargs={"voice_slug": "calm_guide", "emotion_slug": "soothing"}
)
audio, sr = pipeline.generate(
    script="Begin by finding a comfortable position...",
    music_engine="acestep",  # or "lyria"
    duration_sec=300
)
```

## References

- **IndexTTS-2 GitHub**: https://github.com/index-tts/index-tts
- **MoodScape Architecture**: `docs/ARCHITECTURE.md`
- **IndexTTS Prompting Guide**: `docs/prompting_guides/vocal_indextts_instructions.md`
- **Voice Manager**: `core/index_tts/voice_registry.py`

---

**Installation Date**: 2026-05-24  
**Status**: Production-ready for meditation audio generation
