# Phase 3 — rulings and deferred minors

From the subagent-driven ledger (2026-10-10). Spec: `docs/superpowers/specs/2026-10-10-phase3-design.md`; plan: `docs/superpowers/plans/2026-10-10-phase3.md`; findings: `docs/superpowers/notes/2026-10-r2-findings.md` (M1).

## Rulings
- Ruling: Execution on `dev` in the main checkout (program preference) — cost if wrong: commits movable.
- Ruling: The gentle line drops all medical conditions entirely (user rule 2026-10-10) instead of keeping a softened medical caveat — the user called the medical wording near-offensive; the remaining safety intent (stop if uncomfortable) is preserved — cost if wrong: listeners with conditions get no explicit medical caveat.
- Ruling: Duration length repair, content-check repairs and fatal repairs share the one MOODSCAPE_SCRIPT_MAX_REPAIRS budget (default 2) — no new budget knob; content check findings degrade to ADVISORY when the budget is spent — cost if wrong: a script with both a fatal problem and an overshoot may skip the length repair (pause scaling still applies).
- Ruling: SENSITIVE_TOPIC (deterministic) is FATAL; LLM content-check findings are FATAL-for-repair but ADVISORY after budget — a model false positive must never stop production, the lexicon covers the known classes — cost if wrong: a novel sensitive phrasing could ship after 2 failed repairs (logged in meta).
- Ruling: Tasks 1+2 dispatched as one batch (same linter/test files; T1's 'old medical line fails' depends on T2's rule) — one review covers both — cost if wrong: a larger single review.
- Ruling: energising_breath pack's safety text listed medical contraindications (high blood pressure, eye conditions, injuries) to the writer; replaced with the gentle line per the user rule — cost if wrong: the writer no longer knows those contraindications exist (scripts couldn't have voiced them anyway).
- Ruling: SENSITIVE_TOPIC excludes the yoga pose name "child's pose" (pose name, not child imagery; would waste repairs in movement packs) and adds clear-cut misses: sick(ness), hospital(s), asthma, newborn(s), infant(s), "little one(s)"; "ill" alone and "baby steps" stay as-is (ill → "ill at ease" risk; baby steps is rare collateral, repair rewrites it) — folded into Task 3 with a pinned test for each choice — cost if wrong: a stray idiom costs one repair.
- Ruling: Tasks 5+6 dispatched as one batch (T6 consumes T5's scale_pauses and adds total_pause_sec in the same module) — cost if wrong: larger review.
- Ruling: accept implementer's addition — if fatal violations remain after the length repair and the budget is spent, fall back to the last fatal-free script instead of failing the job (a length rewrite must never be why a job fails); pause scaling kept only if the scaled script stays fatal-free — cost if wrong: an over-long script ships (ADVISORY) rather than failing.
- Ruling: Task 7 runs the content check AFTER the length repair and BEFORE pause scaling (so a content rewrite is what gets scaled), and adds duration_control["length_repair_reverted"] (review minor 1, same code) — cost if wrong: none.
- Ruling: content-check categories keep self-harm and weight/body shape (spec K1, concrete classes) alongside the user's medical/pregnancy/child classes — cost if wrong: occasional extra repair on body-focused packs.
- Ruling: Task 11 fix round re-review done by the controller against code (engine.py:70-80 docstring, auto_generate order) instead of a subagent — three doc lines, verified line by line — cost if wrong: a doc wording slip.
- Ruling: MOODSCAPE_DF_STREAMING default flips to "0" (offline) — the +23 GB in M1 was dominated by F5 still being resident, now fixed by the early unload; streaming stays opt-in for very long renders; Task 12 smoke verifies peak ≤ 28 GB with offline DF — cost if wrong: if the peak is still high, enable streaming (+~10 min per 12-min render).
- Ruling: Round C stays a two-render pacing comparison (trim happens inside F5 before the TTS cache, so no same-voice variant exists without a redesign); the page tells the listener to judge pacing only — cost if wrong: render noise blurs the trim signal; a later post-cache trim could make it same-voice.
- Ruling: MISSING_DISCLAIMER accepting "strain"/"too much" stays (gentle wording variants; slightly looser) — cost if wrong: a vaguer safety line passes.
- Task 13: parked — length repair asks for ~1 word when pause seconds ≥ window midpoint — Ruling: real but rare (pause-dominated scripts) and harmless (one wasted repair; pause scaling + ADVISORY follow); fix in R3b by skipping the length repair when pauses ≥ mid — cost if wrong: a pause-heavy script burns one repair slot.
- Ruling: content-check weight category narrowed to body image/dieting with grounding "weight" explicitly allowed (commit e38213c, 6 prompt lines, controller-verified) — real smoke false positive that would strip grounding imagery across the eval — cost if wrong: none.

## Parked / deferred minors
- Task 3: minor (deferred): energising_breath rewrap left one long line; self_compassion "earlier stages of life" abstract (linter backstops)
- Task 4: minor (deferred): malformed _LEAD_MS/_DB env raises mid-synthesis (same as neighbours); lead_ms<10 fades inside onset (clamp); weak first-sample assertion
- Tasks 5+6: minor (deferred): adjacent-pause merge makes factor approximate (clamped, harmless); sub-1 s pauses floored up when shrinking
- Task 7: minor (deferred): truthy non-string technique (MagicMock) would hit _WS.sub outside try; salvaged replies indistinguishable from complete; progress fractions non-monotonic; failure artifacts omit content-check state
- Task 10: minor (deferred): config name other than "eval" breaks hardcoded /eval paths (spec uses eval=); arm names "chain"/"export" would clobber logs (reserve); absolute paths embedded; no RED run
- Task 13: parked — length repair asks for ~1 word when pause seconds ≥ window midpoint — Ruling: real but rare (pause-dominated scripts) and harmless (one wasted repair; pause scaling + ADVISORY follow); fix in R3b by skipping the length repair when pauses ≥ mid — cost if wrong: a pause-heavy script burns one repair slot.
- Task 13: minor (deferred): auto_generate imports private linter._strip_tags (make public)

## Smoke evidence
- Task 12: smoke (box_breathing): base peak 24.1 GB, r2_trim 22.4 GB (was 42.8–43.2) ✓; gentle line in opening ✓; 0 SENSITIVE_TOPIC ✓; duration est 359.5 (window 180–360) via pause_scale 0.902 ✓ (actual 6.31 min); content_check ran 1→0 ✓ but finding was a FALSE POSITIVE ("body rests in its own weight" under weight/body-shape); r2_trim 45 s shorter, gaps 7→2, QA 9/35 failed → 2 still failing; LUFS −16 both.
