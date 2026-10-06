# R0: Evaluation Harness Design

**Date:** 2026-10-06
**Program:** [Generation Quality Roadmap](2026-10-06-generation-quality-roadmap.md)
**Status:** Design approved; awaiting spec review

## Purpose

Every later sub-project (R1–R5, Track Q) changes something whose effect on
quality is uncertain. R0 lets each change be judged two ways:

1. **Automatic metrics** catch regressions and gate obviously broken output.
2. **Blind A/B listening** makes the final ship/no-ship call. The research is
   explicit that metrics can disagree with human ears (Voxtral TTS report), so
   metrics reject bad output, and ears choose between good outputs.

## Non-goals

- No UTMOS or speaker-similarity models. They are new dependencies, and they
  arrive in R2, where they gate per-chunk re-renders.
- No LLM rubric scoring of scripts. That is the R4 selector; R0 records only
  deterministic script metrics.
- No changes to the generation or audio path. R0 is read-only with respect to
  `MeditationPipeline` and `auto_generate`.

## Architecture

```
scripts/eval_genres.py  (extended)            scripts/ab_export.py   scripts/ab_score.py
        │  renders runs (existing)                     │                      │
        │  --metrics → core/eval/*                     │ pairs two run dirs   │ tallies choices
        ▼                                              ▼                      ▼
<out>/<config>/<genre>-<i>/                   <ab>/pairs/NN_{A,B}.wav    sign test, win rate
   <name>.wav  .script.txt  .meta.json        <ab>/key.json (hidden)
   metrics.json   ← new
<out>/results.json + summary.md ← new
```

### Package `core/eval/`

| Module | Responsibility | Public functions |
|---|---|---|
| `script_metrics.py` | Deterministic text metrics | `script_metrics(script, *, draft=None, engine, content_type) -> dict` |
| `slop.py` | Slop lexicon + density | `SLOP_LEXICON`, `slop_hits(text) -> list[Hit]`, `slop_density(text) -> float` (hits per 150 words) |
| `speech_metrics.py` | Intelligibility and artifact metrics on rendered speech | `transcribe(audio, sr) -> Transcript` (pluggable backend), `speech_metrics(audio, sr, expected_text) -> dict` |
| `mix_metrics.py` | Loudness and duration on the final file | `mix_metrics(path, *, estimated_sec=None) -> dict` |
| `testset.py` | Loads `eval/testset.toml` | `load_testset(path) -> list[TestCase]` |
| `ab.py` | Excerpt cutting, level matching, pairing, scoring | `cut_excerpts(...)`, `build_pairs(...)`, `score(...)` |

Each module is importable without torch/Gradio (lazy imports inside
functions), so the unit tests stay fast and never touch `app.py`.

#### `script_metrics`

- **Counts:** `words`, `sentences`, `mean_sentence_words`, `max_sentence_words`.
- **Silence:** `pause_seconds` (summed with the same `prepare_segments()` the
  renderer uses, so paragraph pauses count), `breath_count`, and
  `pause_ratio_est = pause_seconds / estimate_duration_sec(...)`.
- **Linter:** `lint_fatal` and `lint_advisory` (code lists from `linter.check`).
- **Slop:** `slop_density`, `slop_top` (the 5 most frequent hits).
- **Draft vs final:** `edit_ratio` (token-level `difflib` ratio between draft
  and final, when the draft is available). This detects judge flattening; the
  research suggests alerting above 25%.

#### `slop` lexicon (seed; R1 refines it)

The lexicon is the research's artifact (d), in three tiers:
- **Hard-discourage:** any hit counts.
- **Rate-limited:** the counted unit is per 150 words.
- **Pattern rules:** regexes such as "more and more", triple-adjective stacks
  and "it's not X, it's Y".

It is stored as plain data in `slop.py`. Matching is case-insensitive and
word-boundary based, run on tag-stripped prose with curly apostrophes
normalised (the same normalisation the safety linter uses).

#### `speech_metrics`

- **Transcriber interface:** `Transcriber.transcribe(audio, sr) -> Transcript(text, words=[(word, start, end)])`.
- **Default backend:** `transformers` pipeline with `openai/whisper-large-v3-turbo`
  (already in the HF cache and used by F5's reference-transcription fallback),
  `return_timestamps="word"`, device MPS with CPU fallback, loaded lazily and
  unloaded after use.
- **Text normalisation** for CER/WER: lowercase, strip punctuation,
  spell digits, collapse whitespace. Script tags are removed from the expected
  text first.
- **Metrics:**
  - `cer`, `wer`: edit distance via `rapidfuzz.distance.Levenshtein`, added to
    `requirements.txt` (if not already present). It's pure C++ with no model.
  - `internal_gaps`: count of non-speech spans >250 ms that fall *inside* a
    word span from Whisper word timestamps, plus spans >250 ms between two
    words of the same sentence. This catches F5's mid-word gaps and stalls.
  - `sec_per_char_p50`, `sec_per_char_iqr`: per-sentence speaking-rate spread,
    measured from word timestamps.
- **Input:** a run's voice stem when one is exported, otherwise the final mix.
  The bed is quiet enough under speech for Whisper. The exporter records which
  input was used.

#### `mix_metrics`

- `integrated_lufs` as written, plus `integrated_lufs_dual_mono` (mono
  duplicated to 2 channels and re-measured), so the R5 loudness bug and its fix
  are both visible.
- `lra` (loudness range), `true_peak_dbtp` (4× oversampled, the same method as
  `mixer.true_peak_limit`), and `actual_sec`.
- `duration_ratio = actual_sec / estimated_sec`.

### Fixed test set: `eval/testset.toml` (tracked)

- **12 genres:**
  - Stress & Anxiety: anxiety_relief, panic_sos
  - Sleep & Rest (meditation-style): yoga_nidra, racing_mind_at_night
  - Sleep stories: deep_sleep, fall_asleep
  - Breathwork: box_breathing, coherent_breathing
  - Body & Movement: body_scan, workout_cool_down
  - Emotional: self_compassion, grief_and_loss
- **Bands:** short and medium, giving 24 jobs.
- **Fixed per-job seeds** for angle choice, background choice and the TTS base
  seed. Even so, F5 is nondeterministic (≤ ~4 LU short-term drift), so
  comparisons are made within a run pair, never against a stored historical
  number.
- **Steer:** empty.

### Runner changes: `scripts/eval_genres.py`

- `--testset eval/testset.toml` as an alternative to `--genres/--band` (each
  case carries its own band and seed).
- `--metrics` computes all three metric groups after each run and writes
  `metrics.json` beside the audio.
- `--export-stems` passes `do_export_stems=True` so speech metrics run on the
  clean voice stem.
- `--engine {f5,qwen3,...}` sets `AutoConfig.tts_engine`, so Track Q can join.
- **At the end:** `results.json` (with the existing fields plus metrics) and a
  `summary.md` table with per-config medians: CER, internal gaps, slop
  density, edit ratio, |1 − duration ratio|, dual-mono LUFS, and pass rate.
- Existing behaviour without the new flags is unchanged (existing tests keep
  passing).

### Blind A/B: `scripts/ab_export.py` and `scripts/ab_score.py`

- **`ab_export.py`** `--a <run_dir> --b <run_dir> --out <ab_dir> [--excerpt-sec 75]`:
  1. Matches jobs across the two dirs by test-case id.
  2. Cuts excerpts at 10% (start), 50% (middle) and 85% (end) of each file. The
     end excerpt samples long-session monotony and the sleep fade.
  3. Level-matches each pair to −20 LUFS (dual-mono measurement) so loudness
     doesn't bias choices.
  4. Randomises A/B order per pair with a recorded seed and writes
     `pairs/NN_A.wav`, `pairs/NN_B.wav` plus `key.json` (pair → arm, case,
     position).
  5. Writes `sheet.csv` with an empty `choice` column (`A`/`B`/`same`) and a
     `reason` column.
- **`ab_score.py`** `--ab <ab_dir>`:
  - Reads `sheet.csv` and `key.json`.
  - Reports wins/losses/ties per arm, overall and by position
    (start/middle/end) and content type.
  - Runs a one-sided sign test on non-tie pairs.
  - Prints SHIP if the challenger wins ≥ 60% of non-tie pairs and p < 0.05;
    otherwise NO-SHIP or INCONCLUSIVE.

## Error handling

- A metric failure (e.g. a Whisper load error) records
  `{"error": "..."}` for that metric group and never fails the run row.
  This mirrors the existing "calibration must never fail a job" rule.
- `ab_export` refuses to pair cases that are missing on either side and lists
  them, instead of silently shrinking the set.
- Excerpts shorter than requested (short files) are cut to the available
  length, and this is flagged in `key.json`.

## Testing

- **Unit (fast, no models):**
  - Slop hit and density on fixtures.
  - Script metrics on a known script (pause seconds match `prepare_segments`).
  - CER/WER normalisation, including digits and tags.
  - Internal-gap detection on synthetic audio (tone bursts with a planted 400 ms
    gap) plus fake word timestamps.
  - Dual-mono LUFS ≈ mono + 3.01 LU on pink noise.
  - True-peak on a known intersample-peak signal.
  - Excerpt positions, level matching and pairing determinism given a seed.
  - Sign-test thresholds (17/24 → SHIP).
  - Testset loading.
- **Fake transcriber** injected via the `Transcriber` interface for
  `speech_metrics` unit tests.
- **Integration (slow, gated by `MOODSCAPE_EVAL_LIVE=1`):** real Whisper on a
  short TTS fixture (CER < 10%).

## Deliverables

- The `core/eval/` package and tests.
- `eval/testset.toml`.
- Extended `scripts/eval_genres.py`, new `scripts/ab_export.py` and
  `scripts/ab_score.py`.
- A `docs/auto_generation/README.md` section "Evaluating a change", with the
  workflow: baseline run → change → run → `ab_export` → listen → `ab_score`.
- A first baseline run of the current golden path, saved under
  `var/eval/baseline-<date>/` (gitignored). It is the reference that R1–R5 and
  Track Q compare against.
