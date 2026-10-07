<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/qwen3_tts/engine.py` · `core/qwen3_tts/preprocessor.py` · `core/qwen3_tts/presets.py` · `core/qwen3_tts/postprocessor.py`
**Class:** `Qwen3Engine` — `load_model()` / `synthesize()` / `unload_model()` · mlx-audio on Apple Silicon · three modes (clone / custom / design)
**Constants:** `INTER_CHUNK_GAP_SEC=0.4` · `_RETRY_SEED_OFFSET=1000` · `DEFAULT_SPEAKER="serena"` · chunk size `MOODSCAPE_QWEN3_CHUNK_CHARS=200`
**Contract:** Output — 24 kHz mono float32 + bool voice-activity mask · reuses the F5 voice pool (`assets/speakers/`) in clone mode
**Status:** experimental, **Sandbox tab only** (registry entry `qwen3`, `experimental: True`) — hidden from the Manual and Auto-Generate engine dropdowns
**Tasks:**
- Switch mode / tune sampling → `MOODSCAPE_QWEN3_*` env flags (below), or the Sandbox tab's Qwen3 controls
- Edit delivery instructions → `presets.py :: INSTRUCT_PRESETS` (custom) / `DESIGN_PRESETS` (design)
- Tune chunking → `preprocessor.py :: split_into_chunks()` via `MOODSCAPE_QWEN3_CHUNK_CHARS`
- Tune voice FX → `postprocessor.py` (re-exports Chatterbox's mastering engine + voice chain)
**See also:** `docs/superpowers/notes/2026-10-qwen3-tts-findings.md` (measured runs) · `docs/model_implementation_guides/f5_tts.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# Qwen3-TTS — Implementation Guide

## What Is Qwen3-TTS?

Qwen3-TTS (Alibaba Qwen team, **Apache-2.0**) is an autoregressive 12 Hz-codec TTS model. MoodScape runs the 1.7B bf16 MLX conversions from `mlx-community` through [`mlx-audio`](https://github.com/Blaizzy/mlx-audio). Upstream publishes the model as three separate checkpoints, so the engine has three mutually exclusive **modes**, one per checkpoint:

| Mode | Repo id | What drives the voice | `instruct` |
|---|---|---|---|
| `clone` (default) | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | our F5 reference clip + transcript (`assets/speakers/`, same `voice_registry` as F5) | **not supported** — dropped with one warning |
| `custom` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | a built-in preset speaker (`MOODSCAPE_QWEN3_SPEAKER`, default `serena`) | delivery/style instruction |
| `design` | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | a free-text narrator description | voice description (required) |

The mlx-audio calls the engine makes (`engine.py :: Qwen3Engine._call`):

```python
# clone
model.generate(text=, ref_audio=, ref_text=, temperature=, lang_code="english")
# custom
model.generate_custom_voice(text=, speaker=, language="English", instruct=, temperature=)
# design
model.generate_voice_design(text=, language="English", instruct=, temperature=)
```

Each returns an iterable of results with `.audio` (`mx.array`) and `.sample_rate`; the engine concatenates every result and resamples to exactly 24 kHz (`postprocessor.condition_chunk`, which also trims edge silence below −45 dBFS and applies 5 ms edge fades).

Weights are ~4.2 GB per repo in the default HF cache (`~/.cache/huggingface/hub`). If one is missing, `load_model()` raises with the exact `huggingface-cli download <repo>` command.

## Env Flags

All flags are read at synthesis time, so the Sandbox tab can set them per render (`sandbox_tab.apply_qwen3_env`).

| Flag | Default | Effect |
|---|---|---|
| `MOODSCAPE_QWEN3_MODE` | `clone` | `clone` / `custom` / `design` — selects checkpoint and call shape. Anything else raises `ValueError`. |
| `MOODSCAPE_QWEN3_TEMPERATURE` | `0.7` | Sampling temperature for every mode (upstream default is 0.9). |
| `MOODSCAPE_QWEN3_CHUNK_CHARS` | `200` | Max characters per synthesis chunk; chunks split on sentence boundaries, over-long sentences on word boundaries. |
| `MOODSCAPE_QWEN3_STRETCH` | `1.0` (off) | Optional pacing arm: `librosa.effects.time_stretch` rate per chunk (`< 1` slows). |
| `MOODSCAPE_QWEN3_INSTRUCT` | empty (use preset) | Overrides the preset instruction (custom) or voice description (design). Ignored in clone mode with one warning. |
| `MOODSCAPE_QWEN3_SPEAKER` | `serena` | CustomVoice preset speaker (custom mode only). |
| `MOODSCAPE_QWEN3_DF_WET` | `0.5` | DeepFilterNet wet blend applied to the Qwen3 voice stem in the pipeline (0 = off). |

All Qwen-specific settings are folded into the TTS cache key through `compute_cache_key(..., extra=...)`, so changing any of them never reuses a stale render; existing F5/Kokoro cache keys are unchanged.

## Instruct Presets

Chosen by `content_type` (`presets.instruct_for`); anything other than `sleep_story` uses the meditation preset.

**Custom mode — `INSTRUCT_PRESETS` (delivery instruction):**

- `meditation`: "Speak slowly and softly in a warm, intimate, unhurried voice, with a gentle falling cadence at the end of each sentence, as a calm meditation guide."
- `sleep_story`: "Speak very slowly and quietly, soft and low, drowsy and soothing, like reading a bedtime story to someone who is falling asleep."

**Design mode — `DESIGN_PRESETS` (voice description):**

- `meditation`: "A warm adult female narrator in her thirties with a low, soft, fully voiced tone. Calm, unhurried and intimate, like a meditation teacher speaking close to the microphone."
- `sleep_story`: "A gentle adult female narrator with a low, hushed but fully voiced tone. Drowsy, soothing and very slow, like a bedtime storyteller."

## Synthesis Flow

1. `preprocessor.prepare_segments(script, content_type)` — reuses F5's script grammar (`[pause:Xs]`, `[breath]`, paragraph pauses from the content profile) and F5's text normalisation, lowercases ALL-CAPS words, then chunks speech to `MOODSCAPE_QWEN3_CHUNK_CHARS`.
2. Per speech chunk: seed MLX with `seed + speech_index`, call the mode's `generate*`, condition to 24 kHz. If the result is empty, non-finite or silent (peak ≤ 1e-4), **retry once with seed + 1000**, then raise `RuntimeError` naming the chunk — silence is never returned as speech.
3. Optional time-stretch; 0.4 s gap between consecutive speech chunks; pauses become exact silence; breaths come from `core.breath_sounds`.
4. Logs one INFO line on `moodscape.qwen3_tts`: `Qwen3 synth mode=… audio=…s wall=…s RTF=…`.
5. In the pipeline: 24 → 48 kHz upsample, DeepFilterNet at `MOODSCAPE_QWEN3_DF_WET`, Chatterbox's mastering engine and voice chain (`postprocessor.py` re-exports them), then the normal mix/master/export. Registry presets: reverb 0.10, duck −16 dB.

## Known Limits

- **Upstream `speed` is not implemented** in mlx-audio's Qwen3 model. The pipeline's `speed` argument is accepted and ignored; pacing comes from the script's pauses, the instruct text (custom/design) and the optional `MOODSCAPE_QWEN3_STRETCH` arm.
- **Clone mode has no instruct.** The Base checkpoint takes only `ref_audio` + `ref_text`; an instruct set by env or the UI is dropped with a single warning rather than passed (which would raise).
- **Custom-mode speakers are fixed upstream presets.** `serena` is listed upstream as a Chinese-native voice; the English-native presets (`ryan`, `aiden`) are male. See the findings note for what was measured.
- **Loading prints two harmless upstream warnings** (`model of type qwen3_tts to instantiate a model of type ''` and a tokenizer "incorrect regex pattern" notice). Neither stopped any run.
- **Sandbox-only until promoted.** `list_all_engines()` shows `qwen3` only with `include_experimental=True` (the Sandbox tab); Manual and Auto-Generate never list it.

## Memory

- Load on synthesis, unload after: `synthesize()` loads the mode's checkpoint if needed; the pipeline unloads the TTS before loading music; `unload_model()` drops the model, runs `gc.collect()` and `mx.clear_cache()`.
- Only one checkpoint is ever resident (mode is fixed per engine instance). It must never be resident beside an LLM or F5 (32 GB constraint).
- Measured peak RSS (`/usr/bin/time -l`, M1 Max 32 GB, 2026-10-06): see the table below.

Whole-pipeline process (TTS + DeepFilterNet + FX + export), 5-minute scripts, `/usr/bin/time -l`:

| Mode | Peak memory footprint (incl. Metal) | Max RSS |
|---|---|---|
| clone | 19.2-20.7 GB | 4.6-5.1 GB |
| custom | 19.0-21.5 GB | 5.1-8.0 GB |
| design | 18.6-18.8 GB | 7.2-8.3 GB |
| F5 (reference) | 32.7-39.9 GB | 6.2-6.5 GB |

The footprint is far above the 4.2 GB of weights, probably MLX buffer-cache growth across chunks; `mx.set_cache_limit` is an untried follow-up. TTS RTF was 0.24-0.47 (F5: 1.23-1.66). Full detail in the findings note.

## How to Promote

The Sandbox tab's **Promote Model to App** button (`sandbox_tab.handle_promote_model` → `engine_registry.promote_model`) writes a record to `var/promoted_models.json`, which then appears in the Manual/Auto dropdowns as "[Promoted]".

**Not yet wired for Qwen3.** As of this writing a promoted record with `base_engine: "qwen3"` is not handled by `engine_registry.get_engine()` (it handles promoted `kokoro`/`f5`/`chatterbox`/`custom` only) and falls back to Kokoro; the promote handler also does not store the Qwen3 mode/temperature/speaker/instruct/stretch settings, which the engine reads from env. Before promoting Qwen3, add a `base == "qwen3"` branch to `get_engine()` and persist the `MOODSCAPE_QWEN3_*` settings in the promoted record's presets (and apply them at render time). Promotion also needs the user's listening sign-off from the R0 A/B against F5.
