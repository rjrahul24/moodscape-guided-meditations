# Generation Quality Program: Roadmap

**Date:** 2026-10-06
**Status:** Closed 2026-10-23 — see `docs/superpowers/notes/2026-10-program-closeout.md`
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
  commercial-safe engine (Qwen3-TTS, Apache-2.0) was evaluated in parallel
  (Track Q) and parked on 2026-10-08. The commercial engine question is
  still open.
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
| **Q** | Qwen3-TTS sandbox engine | **Parked 2026-10-08** on branch `dev-qwen-tts-temp` (worktree `../moodscape-qwen3-tts`). It lost the listening check: F5 scored 5/5, the best Qwen mode 3/5 (too fast, accent, voice drift mid-file). The user retries it separately, and removes it if that fails. Its spec, plan and findings live only on that branch. | — |

## Execution order after the R1 evaluation (2026-10-08)

The R1 blind A/B (`var/eval/r1/RESULTS.md`) set the order. Round 1 was a
null result: 5 R1 wins, 6 baseline wins and 10 ties. Round 2 (writing only)
went 6–4–2, leaning R1. The listener's notes, from both arms, say:

- "robotic / not natural / low clarity" (7 reasons);
- "music gets loud suddenly" and "music poor";
- clichés barely matter (2 ticks in 33 pairs).

A stem check confirmed both causes: the voice alone is still somewhat
robotic, and the music masks it.

R1 closed with a follow-up on `dev`:
- Brittney is the default voice again; per-family choice is opt-in via `MOODSCAPE_VOICE_BY_FAMILY=1`.
- The self-compassion inner-child angle was removed.
- yoga_nidra's rotation now scales to session length, and its sleep-equivalence claim is gone.
- Two more natural disclaimer wordings are accepted.

R1's script changes ship. The length fix and the safety codes are
corrections, not taste calls.

**Order: E → R5a → R2 → R3 → R4 → R5b.**

Every phase keeps the existing cycle: brainstorm → spec → plan →
subagent-driven development with per-task review → a final review on the
most capable model → a blind A/B. Evals run only with the user's go-ahead,
hold `var/heavy-model.lock`, and unload models afterwards.

### E. Eval hardening (small; do first)

Every remaining phase changes audio, and Round 1 showed the current A/B has
too little power. Different scripts per arm add variance, and half the pairs
tie.

- **`--scripts-from <arm_out>` render-only mode** (deferred R0 Important 6). Both arms render the *same* scripts and backgrounds, so an audio A/B measures audio only. Skipping the LLM stages also cuts a job to roughly the F5 render (8–11 min) and less fan time.
- **Shuffle pair order** in `ab_export` (deferred R0 minor), so a genre's three pairs are not consecutive.
- **Add a "lean A / lean B" answer** to the listening page. The scorer counts leans as half-weight wins in a second, reported-only statistic; the SHIP rule is unchanged. Several "same" answers in R1 named a side.
- Exit: unit tests, plus a one-case render-only smoke under the RAM lock.

### R5a. Mix clarity and loudness (moved up from R5)

This is the cheapest fix for the "music masks the voice" half of the problem.

- **Loudness convention fix.** Meditation exports at −16 LUFS dual-mono (−19 mono), and sleep at −18 (−21). A clear bug, so it ships ON with a kill-switch.
- **Bed rise in pauses.** Measure how fast and how far the bed swells in long pauses, then bound both rise rate and ceiling. This targets "music gets loud suddenly".
- **Masking A/B arms on the same scripts:**
  - the existing `MOODSCAPE_SPECTRAL_DUCK=1`;
  - duck depth of 20–24 LU against today's 30.5 LU.
- Exit: a blind A/B win for any golden-path default change. The loudness fix needs only a metric check.

### R2. F5 chunk ownership and speech QA

This targets the "robotic / glitches / gaps" half of the problem.

- **Chunking.** Breath-group chunks sized to F5's own budget, so its internal re-splitter never fires. Gaps depend on punctuation.
- **Per-chunk QA.** Whisper CER, internal gaps, sec/char outliers, speaker drift and UTMOS, with best-of-3 re-render (7-NFE EPSS) on failures only.
- **Pending research flags as same-script A/B arms.** These cost one extra render each:
  - `MOODSCAPE_F5_CFG=1.2` ("warmer, less identical");
  - `MOODSCAPE_F5_REF_PRESERVE_DYNAMICS=1`.
- **RAM is a design constraint.** F5's whole-pipeline peak footprint measured 33–40 GB on the 32 GB machine, and R1's morning_start render died of an MPS OOM. R2 must measure peak memory and keep re-renders inside today's peak, for example by freeing caches between chunks and by loading Whisper only after F5 is unloaded.
- After R2, re-test per-family voices: Delilah's doubled gap count may simply be the stall R2 fixes.

### R3. Timing engine and duration control

As scoped above. Duration control turns the ADVISORY duration drift into a
targeted fix: pause scaling ±15%, then a re-write. That covers overshoots
like yoga_nidra without making duration FATAL.

### R4. Script selection

As scoped above. The evidence to aim at:
- Round 2 lost on morning_start and box_breathing (wording, "mid-sentence pauses").
- The listener rejected content (inner-child imagery) that no regex anticipated. That argues for the ADVISORY LLM content and safety classifier.
- Slop density stays ADVISORY; listeners did not hear it.

### R5b. Sleep and music

The rest of R5:
- sleep post-roll with a long fade;
- bed darkening over the final 30% of a sleep story;
- persisted recent tracks;
- CLAP/MuQ text-prompt music matching;
- the DeepFilterNet wet A/B arm.

### Not in this program

The commercial-engine question (F5 is CC-BY-NC). It needs its own research
spike once the user has finished retrying Qwen.

## Phase 3 (2026-10-10, after the R2 eval)

See `docs/superpowers/specs/2026-10-10-phase3-design.md`. Results that drove it are in
`var/eval/r2/RESULTS.md`. The phase covers:
- content sensitivity (our R1 disclaimer rule produced the medical wording the listener rejected);
- the F5 lead-in trim and duration control (R3a);
- the LLM content check (R4a);
- the memory peak;
- the R3 eval (spectral duck, R2 on fresh scripts, lead-in trim, candidate stack).

R3b, R4b and R5b follow, in an order set by that eval.

## Close-out (2026-10-23, after the R3 eval)

R3 was null or leaning toward today's defaults in every round, and every
listener note named robotic F5 delivery as the remaining problem. The program
closed with the listening-driven fixes only (bed 3 LU quieter, one track
removed, a duration-repair fix). R3b, R4b and R5b move to Parked: none
addresses the binding limit, which is now a voice-engine question (a blind F5
vs ElevenLabs v4 round, or new reference recordings). Full record:
`docs/superpowers/notes/2026-10-program-closeout.md`.

## Parked

- **R3b / R4b / R5b** as listed above (breath-cycle tags, best-of-3 drafts, sleep post-roll, music matching).
- **Blind F5 vs ElevenLabs v4** on the same scripts (paid; the harness supports `--engine elevenlabs`).

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
