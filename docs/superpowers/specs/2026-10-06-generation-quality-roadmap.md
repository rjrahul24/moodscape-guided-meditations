# Generation Quality Program: Roadmap

**Date:** 2026-10-06
**Status:** Approved structure; per-sub-project specs follow
**Source:** `artifact_research_output.md` (deep-research report, Oct 2026), verified against this codebase.

## Goal

Maximise the *perceived* quality of auto-generated meditations and sleep
stories: pacing and silence that feel intentional, a warm voice free of
artifacts across 3–15 minutes, specific non-clichéd writing, and a
broadcast-grade mix. Reliability and speed matter, but less.

## Decisions

- **Engine strategy: optimise F5 now, stay engine-agnostic.** Commercial use
  is coming. F5-TTS weights are CC-BY-NC, so speech QA, timing and evaluation
  are built against the `SpeechEngine` interface, not F5 internals. A
  commercial-safe engine (Qwen3-TTS, Apache-2.0) is evaluated in parallel
  (Track Q).
- **Golden-path audio changes are flag-gated, default OFF**, until they win a
  blind A/B on the R0 harness. Clear bug fixes ship ON with a kill-switch.
- **Measure before tuning.** Most numbers in the research are "A/B starting
  points", so the harness (R0) is built first.

## Research claims verified against the code

| Claim | Verdict |
|---|---|
| 250-char chunks overflow F5's ~30 s window | **Cause corrected.** `f5-tts` 1.1.22 `infer_process` already re-splits `gen_text` with `max_chars = ref_bytes/ref_s × (22 − ref_s) × speed`. Our slow references (~9 chars/s) plus the 1 s ref-pad give **max_chars ≈ 60–85 bytes**. So each 250-char chunk is silently cut at commas, colons and sentence ends into 3–4 sub-chunks, joined with a fixed 0.15 s crossfade. Our gap logic, VAD, per-chunk seeds, short-phrase pacing and QA never see these sub-chunks. Fix: own the chunking (R2). |
| Mono export at −16 LUFS is ~3 LU hot | **Confirmed.** `export_audio` normalises mono to −16. The mono adjustment in `calculate_loudness_gain` fires only when the target is exactly −18. |
| Guides contradict the code | **Confirmed:** 300 vs 250 chars, speed 0.88 vs 0.80/0.75, 100–130 vs 85 WPM, sleep paragraph pause 1.5 vs 2.0 s, and the example script uses ellipses and "You are completely at peace". |
| Avoid-recent-tracks does nothing in the UI | **Confirmed:** `AutoConfig` is rebuilt per click. |
| `deep_sleep` pause_ratio 0.28 is unreachable under the sleep guide | **Confirmed by arithmetic:** the ≤4 s cap every 2–4 sentences gives ~12–18%. |
| F5 weights are CC-BY-NC | Consistent with known upstream terms. Re-check before commercial launch. |

## Sub-projects

Each gets its own spec → plan → implementation cycle, in this order.

| # | Sub-project | Scope | Depends on |
|---|---|---|---|
| **R0** | Eval harness | Script, speech and mix metrics; fixed test set; blind A/B exporter + scorer | — |
| **R1** | Script-gen fixes | Guides generated from code constants; per-stage model-card sampling; slop lexicon + ADVISORY density check; safety regex additions (rapid breathing, pain-elimination, grief platitudes, energetic healing, "you are safe" as fact in trauma/panic packs, post-exhale holds >4 s in panic/anxiety packs); sleep pause_ratio fix; per-family voice choice; reference-clip recording spec (doc only) | R0 |
| **R2** | F5 chunk ownership + speech QA | Breath-group chunking sized to F5's own budget so upstream never re-splits; punctuation-dependent gaps; engine-agnostic per-chunk QA (Whisper CER, internal gaps, sec/char outliers, speaker-similarity drift, UTMOS gate); best-of-3 re-render with 7-NFE EPSS | R0 |
| **R3** | Timing engine | `[cycle:technique:N]` / `[breathe:in|out]` tags padded to exact phase lengths (coherent, box, 4-7-8, cyclic sighing); voiced-then-silent cycles; per-family progressive pause curves; duration control (pause scaling ±15% → targeted re-write); per-voice duration regression | R0, R2 |
| **R4** | Script selection | Planner returns `<intent>` + pipe-format beat table; best-of-3 drafts; pairwise, order-swapped, rubric-anchored selector (hosted or local); judge reduced to safety/format + capped line edits; beat-signature semantic originality | R0, R1 |
| **R5** | Mix, loudness, sleep, music | Loudness convention fix (meditation −16 dual-mono / −19 mono; sleep −18 / −21, LRA ≤ 6); sleep post-roll 60–180 s + 20–30 s fade; bed LPF/level darkening over the final 30% of sleep; persisted recent tracks; CLAP/MuQ text-prompt music matching; duck-depth A/B arm (20–24 vs 30.5 LU); DeepFilterNet wet A/B arm | R0 |
| **Q** | Qwen3-TTS sandbox engine (parallel, worktree `qwen3-tts`) | See its own spec | — (joins R0 comparisons once R0 lands) |

## Parked

- **Multi-phase reference set:** needs recordings (spec written in R1).
- F5 / LLM fine-tuning (needs 300+ approved scripts).
- Stable Audio 3.0 generated beds.
- LLM safety classifier promoted to FATAL (it starts ADVISORY in R1/R4).
- 20–30-minute sleep stories (after R2 + R3).

## Protected invariants (from CLAUDE.md / GOTCHAS)

Every sub-project must keep:

- The FATAL/ADVISORY split.
- Writer/judge model independence.
- `<problems>` stripping before judge parsing.
- `unload()` after each LLM stage, with sequential model loading under the 32 GB ceiling.
- No pedalboard `Limiter`.
- Upload stem-separation skip.
- The `MOODSCAPE_ADAPTIVE_BED` kill-switch.
