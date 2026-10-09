# E + R5a + R2 — deferred minors and rulings

Carried forward from the subagent-driven ledger (2026-10-08/09). Spec: `docs/superpowers/specs/2026-10-08-e-r5a-r2-design.md`; findings: `docs/superpowers/notes/2026-10-r2-findings.md`.

## Rulings
- Ruling: T7's `test_off_is_identical` is replaced by a test proving the limiter is NOT applied when the flag is off (a 10 s pause reaches ≥ duck+15 dB, i.e. today's full rise + lift) — a two-identical-calls test asserts nothing about behaviour — cost if wrong: none (strictly stronger test).
- Ruling: T11 budget uses the segment's base `speed` (not the per-group short-phrase speed): short-phrase pacing only fires on ≤12-char groups, which are far below any budget, so there is no circularity — cost if wrong: a ≤12-char group at speed 0.5 could exceed a halved upstream budget only if budget < 24 bytes, impossible with f5_char_budget's minimum of 20 × 2… negligible.
- Ruling: T10/T11 `ref_duration_sec` must be the duration of the CONDITIONED (ref-padded) reference file F5 actually loads (`assets["audio"]`); implementer verifies `assets["duration_sec"]` is that, else measures it — upstream computes max_chars from the file it loads — cost if wrong: budget ~10% high (the 0.9 safety factor absorbs it).
- Ruling: Execution on `dev` in the main checkout, no worktree (program's established preference) — cost if wrong: commits movable to a branch.
- Ruling: Task 2 ⚠️ — log_estimate_accuracy now runs only when estimated_sec is truthy (plan-mandated); run() always supplies a positive estimate, so its behaviour is unchanged; render-only sources without an estimate simply skip the calibration log — cost if wrong: a few missing WPM-calibration log lines.
- Ruling: Task 5 ⚠️ — radio `value=` attributes are built in JS at runtime, so the test asserts the `leanB` value string is in the page data rather than a literal `value="leanB"`; the controller's real-data smoke confirmed the page works — cost if wrong: none functional.
- Ruling: Task 7 test fixtures use sr=16000 instead of the plan's 8000 — the pre-existing reactive band-pass (200–4000 Hz) cannot be designed at an 8 kHz Nyquist, independent of this change; production runs at 48 kHz — cost if wrong: none.
- Ruling: Task 7 ⚠️ Review Focus 4 (continuous voice flag-on == flag-off) is not literally true and not tested through compute_breathing_gain_db: with the flag on, the regions before the first and after the last phrase (intro/outro bed) are also capped at duck+12 (−4 dB) — this is the spec's intended "long pauses plateau 4 dB under today's"; helper-level test covers the no-rise case — cost if wrong: intro/outro bed 4 dB quieter in the rise arms, which the A/B will judge.
- Ruling: QA INTERNAL_GAP switches from the signal-only rule to the Whisper word-gap rule (gap_breakdown in-word + between-words-not-at-clause-end > 0), aligning QA with the eval's gaps metric — spec R2-3 defined the signal rule; evidence shows it is ~65% false-positive — cost if wrong: misses silent stalls Whisper doesn't timestamp (CER/rate checks remain).
- Ruling: verification re-run hit the TTS cache (same flags as the first rise_r2) and served the stale voice + stale speech_qa — cache keys cover env flags, not code; busting with MOODSCAPE_F5_QA_DEVICE=mps (the default). Real eval unaffected (fresh scripts) — cost if wrong: none.
- Ruling: stereo sleep stories exporting at −18 LUFS is intended — the playback convention is per content type; "stereo unchanged" in the spec meant "no mono offset" — document it — cost if wrong: stereo sleep exports 2 dB quieter than before.

## Deferred minors
- Task 1: minor (deferred): no test for differing non-empty fingerprints / hard-coded legacy hash
- Task 1: minor (deferred): reuse_last_voice (get_latest_tts) ignores the env fingerprint — UI-only path, eval never uses it
- Task 2: minor (deferred): nested if flattenable; no test for estimate_ratio / skipped log; speech_qa getattr truthy on a MagicMock pipeline (guard isinstance dict)
- Task 3: minor (deferred): src_meta re-read from disk instead of carried on SourceCase; tests don't assert seed/voice reach render_script or exercise --metrics in render-only mode
- Task 4: minor (deferred): positions=[] via direct API yields 0 pairs silently (CLI safe) — raise ValueError; long setdefault line
- Task 5: minor (deferred): no test that hostile `</script>`/quotes are neutralised; title placeholder replace order; multi-round import not atomic; tag-only answer blanks an existing choice; relative audio paths not URL-quoted; same-title pages share localStorage (use unique titles)
- Task 6: minor (deferred): TestExportLoudness appended after the __main__ guard in test_mixer.py (pytest runs it; direct `python` run skips it) — same pattern as R1 deferred minors; no (1,n)-shape test
- Task 7: minor (deferred): no continuous-voice compute-level test; lift_db override for sleep undocumented in docstring (document in Task 14); malformed env float raises mid-render; loose falls assertion
- Task 8: minor (deferred): malformed env offset fails only at mix step (after TTS); one-var override also widens sleep clamp — document in Task 14; commit trailer says Haiku 4.5 (cosmetic)
- Task 10: minor (deferred): budget floor 20 can exceed upstream's raw max_chars only for ~22 s references; f5_char_budget ZeroDivision on 0 s ref (caller guard); whitespace normalised; abbreviations end units (cosmetic)
- Task 11: minor (deferred): flag-off test checks only gen_text/seed (identity confirmed by reviewer's mechanical extraction diff); flag-on untested: pacing on group text, target_wpm groups, inter-segment join; empty segment → zero infer calls with flag on; stale assembly comment; broad except around empty_cache
- Task 12: minor (deferred): untested edges — exact 0.6 s gap boundary, all-silent chunk, zero-length audio, empty-text TRUNCATED/CER
- Task 13: minor (deferred): untested "original stays best" and mid-loop exception paths; candidate outlier rule duplicates speech_qa.rate_outliers via private _RATE_LO/_RATE_HI (expose is_rate_outlier)
- Task 14: minor (deferred): residual "-16 LUFS" mentions in pre-mix/QA contexts (ARCHITECTURE.md, audio_processing.md:129,162) — not export contract
- Task 15: minor (deferred): +13 unit-test warnings are third-party import deprecations (typer/matplotlib/torchao) pulled in by tests importing f5_tts.infer.utils_infer; plus our own ResourceWarning: unclosed file in tests/unit/test_eval_listen.py:54
- Task 16: fix wave re-review — all 5 addressed; new: order-dependent test failure (env leak from eval_genres main() in test_eval_render_only.py) → fixed 6553eaa, verified both orders + 936 suite; minor (deferred): chain.sh last `log "export exit=$?"` reports the if-block status

## Findings outside this round's scope
- Finding (out of scope, for R3): every F5 chunk begins with ~1.5–2.2 s of near-silence (ref-pad leak, A1 ships ON) on the golden path too — dead air before each sentence beyond the script's pauses.
- Debug memory: flags-off render-only of the same script peaks 43.2 GB (1037 s) vs rise_r2 42.8 GB → R2 adds no peak; the ~43 GB whole-pipeline peak is pre-existing (F5 stage alone 17.1 GB) — likely cause of R1's MPS OOM; out of scope, flagged to user as top reliability item. Same-script F5 nondeterminism: gaps 5 (base) vs 10 (mem_base) on identical inputs.
