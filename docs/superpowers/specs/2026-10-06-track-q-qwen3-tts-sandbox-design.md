# Track Q: Qwen3-TTS Sandbox Engine Design

**Date:** 2026-10-06
**Program:** [Generation Quality Roadmap](2026-10-06-generation-quality-roadmap.md)
**Branch / worktree:** `qwen3-tts` (isolated from Track R work)
**Status:** Design approved; awaiting spec review

## Purpose

F5-TTS weights are CC-BY-NC, and commercial use is coming. Qwen3-TTS
(Alibaba, Jan 2026, **Apache-2.0**) is the most promising commercial-safe
challenger: strong reported WER (1.24 on Seed-TTS en for 1.7B),
3-second cloning, natural-language delivery control, and native MLX support.
Track Q makes it a working, meditation-tailored engine in the **Sandbox tab
only**, so it can be judged by ear and on the R0 harness against F5.

## Non-goals

- Not exposed in the Manual or Auto-Generate tabs, and no change to the
  F5 + upload golden path. Promotion goes through the existing sandbox
  "promote model" flow later, after it wins an A/B.
- No fine-tuning.
- No changes to the shared mix/master stages beyond routing the new
  `base_engine`.

## Facts this design rests on (verified in the installed env)

- `mlx-audio` 0.5.4 is installed and ships `mlx_audio/tts/models/qwen3_tts`.
  No new dependency, unless a newer mlx-audio is needed for a fix (pin the
  version deliberately if so).
- Its `generate()` has three mutually exclusive model variants:
  - **Base:** clones from `ref_audio` + `ref_text`. **No `instruct`.**
  - **CustomVoice:** preset speakers + an `instruct` style string.
  - **VoiceDesign:** `instruct` is a voice *description*.
- `speed` is accepted but **not implemented** ("not directly supported yet").
  Pacing must come from text, pauses and instructions.
- Default sampling temperature is 0.9.

## Architecture

New package `core/qwen3_tts/`, mirroring `core/chatterbox_tts/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Explicit exports (`Qwen3Engine`, `prepare_segments`, `build_qwen3_voice_chain`) |
| `engine.py` | `Qwen3Engine(SpeechEngine)`: load/unload/synthesize; mode selection; per-chunk generation; activity mask |
| `preprocessor.py` | Script → segments; chunking for Qwen |
| `postprocessor.py` | Output conditioning and voice FX chain |
| `presets.py` | Meditation/sleep instruction strings, sampling presets, model ids per mode |

### Modes (`MOODSCAPE_QWEN3_MODE`, also a sandbox dropdown)

| Mode | Model (MLX, 1.7B where available) | Voice source | Delivery control |
|---|---|---|---|
| `clone` (default) | `Qwen3-TTS-12Hz-1.7B-Base` | Our `assets/speakers/` reference clips + transcripts (same `voice_registry` as F5) | Text and pauses only |
| `custom` | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | A preset speaker (sandbox dropdown, default a warm female preset) | `instruct` preset by content type |
| `design` | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `instruct` = narrator description | Description includes delivery |

Exact HF repo ids (MLX-converted vs original) are resolved during
implementation from what `mlx_audio` loads. They are recorded in `presets.py`
and in `CLAUDE.md`'s model list.

### Meditation tailoring (`presets.py`)

- **Instruct presets:**
  - Meditation: "Speak slowly and softly in a warm, intimate, unhurried voice, with gentle falling cadence at the end of each sentence, as a calm meditation guide."
  - Sleep story: "Speak very slowly and quietly, soft and low, drowsy and soothing, like reading a bedtime story to someone falling asleep."
- **VoiceDesign narrator descriptions:** one per content type (a warm adult
  female narrator with low, breathy-soft but fully voiced delivery).
- **Sampling:** temperature default 0.7 (stability over variety). A/B arms
  0.6/0.8 via `MOODSCAPE_QWEN3_TEMPERATURE`.
- **Seeds:** per-chunk seed = base + index, mirroring F5.

### Preprocessing

- Reuses the existing script grammar: `[pause:Xs]`, `[breath]/[inhale]/[exhale]`,
  and paragraph pauses by content type (`content_profiles`, using the F5
  values since both are cloning engines).
- Text normalisation reuses F5's `normalize_for_f5` rules (colons, ellipses
  and dashes become commas/periods; compound hyphens removed; ALL CAPS
  lowercased; digits spelled), wrapped in its own function so it can diverge
  later.
- **Chunking:** sentence-boundary chunks targeting ≤ 200 characters
  (`MOODSCAPE_QWEN3_CHUNK_CHARS`). Qwen3-TTS is autoregressive with long
  context, so chunks can be longer than F5's. Shorter chunks reduce runaway or
  hallucination risk; the right size is measured during implementation (see
  smoke tests).

### Synthesis and output contract

- **Contract:** same as other engines: `synthesize(segments, ...) -> (audio
  float32 mono @ 24 kHz, voice_activity bool mask)`.
- **Sample rate:** Qwen3's native output is resampled to 24 kHz with soxr VHQ
  if it isn't already 24 kHz.
- **Per chunk:**
  - Trim leading and trailing silence.
  - Apply a 5 ms fade at edges.
  - Insert the inter-chunk gap: 0.4 s for meditation, matching F5, with a
    constant to tune.
  - Build the activity mask (same threshold approach as F5).
- **Pauses and breaths:** identical handling to F5 (digital silence and breath
  samples).
- **Pacing fallback:** an optional time-stretch arm,
  `MOODSCAPE_QWEN3_STRETCH` (default `1.0` = off). It uses a high-quality
  phase-vocoder (`librosa.effects.time_stretch`, or pedalboard's
  `time_stretch` if available), because `speed` is unimplemented upstream.
  Off by default because stretching can add artifacts.

### Post-processing and pipeline routing

- `engine_registry.BUILTIN_ENGINES["qwen3"]`:
  - `base_engine: "qwen3"`, `experimental: True`, default voice = the F5
    default slug.
  - Presets: speed (ignored or stretch), reverb 0.10, duck −16, df_wet 0.5,
    temperature 0.7, mode clone.
- `pipeline.py` routes `base_engine == "qwen3"` the same way as Chatterbox:
  - F5-grammar preprocessor via the qwen3 package.
  - Its own mastering/voice chain.
  - DeepFilterNet wet from `MOODSCAPE_QWEN3_DF_WET` (default 0.5: Qwen output
    is codec-decoded and likely cleaner than F5).
  - Every other step (mix, master, export) is unchanged.
- **Voice chain:** start from Chatterbox's dry chain (HPF 80 Hz, gentle
  corrective EQ, de-ess), with reverb at the preset amount. Tuned by ear
  during smoke tests.
- **TTS cache:** the key includes engine id, mode, instruct, temperature and
  seed, so cached results never mix across configurations.

### Sandbox UI

Engine dropdown gains "Qwen3-TTS (experimental)". When it's selected, show:
- Mode dropdown.
- Preset speaker (custom mode).
- Instruction textbox, pre-filled from the content-type preset and editable.
- Temperature slider.
- Stretch slider.

Hidden for other engines. Follows the existing `on_engine_change` pattern in
`core/sandbox_tab.py`.

## Memory and performance

- Load on synthesise, unload after (`mx.clear_cache()` + `gc`). Never resident
  alongside an LLM or F5. Expect ~4–5 GB for 1.7B.
- Log the real-time factor per render. Wall-clock time is reported in the
  findings note.

## Error handling

- **Missing model weights:** raise a clear error naming the HF repo and the
  download command. Never fall back to another engine silently.
- **Empty or NaN output for a chunk:** retry once with seed + 1000, then raise
  with the chunk text.
- **Unsupported argument:** a mode/argument mismatch (e.g. `instruct` passed in
  `clone` mode) is dropped with a one-time warning, not sent to
  `mlx_audio`.

## Testing

- **Unit (mocked `mlx_audio` model):**
  - Preprocessor segment output and chunk sizes.
  - Mode → model id and argument mapping (no `instruct` in clone mode).
  - Seed progression.
  - Resample-to-24 kHz contract.
  - Activity mask shape.
  - Registry entry.
  - Pipeline routing branch.
  - Cache-key separation.
  - Retry-on-empty behaviour.
- **Smoke (real model, gated by `MOODSCAPE_QWEN3_LIVE=1`):** one ~60 s
  meditation excerpt per mode via the sandbox pipeline. Check: non-silent,
  ZCR sanity, no NaN, duration plausible.
- The full suite `tests/unit/` must stay green. No golden-path behaviour
  changes.

## Deliverables

- `core/qwen3_tts/`, the registry entry, pipeline routing, sandbox UI controls,
  and tests.
- `docs/model_implementation_guides/qwen3_tts.md`: modes, presets, flags and
  known limits.
- **Findings note** `docs/superpowers/notes/2026-10-qwen3-tts-findings.md`:
  - Per-mode listening impressions on the same 2–3 scripts as an F5 render.
  - Real-time factor and RAM.
  - Artifacts heard.
  - Recommended mode and settings for the R0 A/B against F5.
- One commit series on branch `qwen3-tts`, **not merged** until the user
  reviews.
