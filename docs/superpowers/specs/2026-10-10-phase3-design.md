# Phase 3: content sensitivity, timing (R3a), content check (R4a), memory, and the R3 eval

**Date:** 2026-10-10
**Status:** The user approved the direction in chat ("content fixes first … use
your best judgement … another round of eval after concrete changes"). Design
decisions are recorded as rulings in
`docs/superpowers/notes/2026-10-phase3-deferred-minors.md`.
**Inputs:**
- `var/eval/r2/RESULTS.md` (R2 blind eval, 2026-10-10)
- `docs/superpowers/notes/2026-10-r2-findings.md`
- `docs/superpowers/specs/2026-10-06-generation-quality-roadmap.md`

## What the R2 eval told us

| Signal | Evidence | Consequence |
|---|---|---|
| **Our own content rule produced offensive wording** | The "If you are pregnant, have a heart or respiratory condition…" disclaimer came from R1's planner line, the linter's FATAL requirement and global safety rule 14. Rule 14 leaked it into body_scan. | **Part C, first and blocking.** |
| Spectral duck preferred over the shallow duck | Weighted 0.63; user: "ducking felt really good" | Confirm it against today's mix in the next eval. |
| R2 voice path preferred | 3–0 clear wins, weighted 0.77; Whisper gaps 39 → 21 | Confirm on fresh scripts, then promote. |
| Bounded bed swell: no difference | Weighted 0.46 | Stays OFF. Not re-tested. |
| cfg 1.2: tiny lean | Weighted 0.60 | Stays OFF until the voice is less robotic. |
| Voice still "somewhat robotic" | Notes on most pairs | Part T: remove the 1.5–2 s ref-pad lead-in before every sentence. |
| Scripts overshoot their band | yoga_nidra 10.2 min and box_breathing 7.0 min for a 3–6 min request | Part T: duration control. |
| Whole pipeline peaks around 43 GB on 32 GB | R2 findings; likely cause of R1's crash | Part M. |

## Part C: content sensitivity (bug fix, ON)

**C1. Gentle breathwork safety line.**
- The planner's canonical line becomes: **"If anything feels uncomfortable at
  any point, simply breathe naturally."**
- `MISSING_DISCLAIMER` (FATAL for packs with `requires_disclaimer`) accepts
  only non-medical conditions: uncomfortable, unwell, lightheaded, dizzy,
  strained, "too much". The `pregnan*`, heart, lung and respiratory branches
  are removed. The action branch is unchanged.
- The violation message quotes the new line.

**C2. Safety rules.**
- Global rule 14 is rewritten to the gentle line, for breathwork only.
- New rule 15, **Sensitive topics**:
  - never mention illness, disease, medical conditions, pregnancy, diagnoses,
    medication or surgery;
  - never ask the listener to picture a child, children, a baby or a
    younger self.
- These rules are shown to every writer and judge.

**C3. A global `SENSITIVE_TOPIC` hard-block.** This is a FATAL entry in
`linter._SAFETY_RULES`, so every genre is repaired by the existing loop.
Matches are case-insensitive and whole-word on tag-stripped prose.

| Group | Matches |
|---|---|
| Pregnancy | `pregnan\w*`, `miscarriag\w*` |
| Medical conditions | `(heart\|lung\|respiratory\|medical\|health\|chronic) (condition\|disease\|problem\|issue)s?` |
| Illness and treatment | `illness(es)?`, `diseases?`, `cancer`, `diagnos\w*`, `medications?`, `surgery`, `surgeries` |
| Children | `child`, `children`, `kids?`, `bab(y\|ies)`, `toddlers?` |
| Younger self | `inner child`, `younger (self\|you)`, `younger version of (yourself\|you)` |

Not matched: `childhood`, `heart` alone, "medical" alone.

**C4. Pack cleanup.**
- `self_compassion`: drop the pack-level `child` and `younger` bans (C3 now
  covers them) and restore `time heals`.
- `deep_sleep`: "branches that have held you since childhood" becomes
  "branches that have held steady for a hundred years".
- A test scans every pack, prompting guide and example for C3 matches. The
  safety-rules document is exempt because it names the terms.

**C5.** The listening page gains a tag: **"wording problem / sensitive"**.

## Part T: timing (R3a)

**T1. Lead-in trim** (flag `MOODSCAPE_F5_TRIM_LEADIN`, default OFF):
- **Problem:** every F5 chunk starts with about 1.5–2.2 s of near-silence
  (the ref-pad leak).
- **Fix:** right after `_trim_trailing_silence` in `_infer_chunk`, cut leading
  frames quieter than `peak_dB − 35` down to a 60 ms lead, with a 10 ms
  fade-in.
- **Scope:** both the chunking-off and chunking-on paths. The QA rate check
  then sees the true speech time.
- **Expected effect:** about 1.5 min shorter per 5-minute file, closer to
  the script's own pause timing. `DEFAULT_WPM` calibration will shift; the
  next eval measures it.
- `_LEAD_MS` and `_DB` are tunable through `MOODSCAPE_F5_TRIM_LEADIN_*`, so
  the TTS cache fingerprint picks them up.

**T2. Duration control** (bug fix, ON; kill-switch
`MOODSCAPE_DURATION_CONTROL=0`). In `generate_script`, once a script has no
FATAL violations:
1. **Length repair.** If the estimate is more than 15% outside the window and
   the repair budget allows, send the judge one length repair. The violation
   message names the target, e.g. "estimated 612 s; target 180–360 s — cut to
   about N words, keep every [pause] marker style". This happens at most once
   and counts against `MOODSCAPE_SCRIPT_MAX_REPAIRS`.
2. **Pause scaling.** If the estimate is still outside the window, scale every
   `[pause:Xs]` value by a single factor clamped to [0.85, 1.15]. Each pause is
   kept at 1 s or more and rounded to 0.5 s. Then re-estimate. A pure
   function, `scale_pauses(script, factor) -> str`, does this in
   `script_gen/duration.py`.
3. **Still advisory.** Whatever remains stays ADVISORY, so it never fails a
   job.

`meta.json` records `"duration_control": {"length_repair": bool, "pause_scale": float}`.

## Part K: LLM content check (R4a)

**K1.** After the lint loop passes, one call to the **judge** model (already
loaded) uses a fixed rubric prompt. It asks for any passage that mentions
illness, medical or physical conditions, pregnancy, children or a younger
self, death or bereavement (unless the genre's pack is about grief),
self-harm, weight or body shape, or anything a general wellness audience
could find alienating. The output is strict JSON:
`[{"quote": "...", "reason": "..."}]`.

**K2.** Each finding becomes a `SENSITIVE_CONTENT` violation that goes to the
repair loop as fatal-for-repair, within the shared budget.
- If the budget runs out, the finding stays as an **ADVISORY** violation in
  `meta.json`, so the job still renders. A false positive must not stop
  production.
- C3 is the deterministic FATAL backstop for the known classes.

**K3.** Flag `MOODSCAPE_CONTENT_CHECK`, default **1**.
- Malformed JSON gives one retry, then the check is skipped with a warning.
- A finding is dropped unless its quote is present in the script (this
  guards against hallucination).
- The pack slug is passed so grief packs can talk about loss.

## Part M: memory reliability

**M1. Measure first.** One flags-off render-only run with per-stage
footprint logging: `resource` and `psutil` RSS plus
`torch.mps.driver_allocated_memory()` after each pipeline step (2, 3, 4, 6,
7, 8, 9, 10, 12).

**M2. Fix the top two contributors.**
- The likely ones are float64 promotions and full-array copies in the
  48 kHz mix and export path, Demucs and music arrays kept after use, and
  Whisper metrics loading while big arrays are still alive.
- Fixes must keep audio output identical within 1e-6 peak difference, which
  is the test.
- **Target:** whole-pipeline peak ≤ 28 GB on the box_breathing smoke script.
- If a contributor needs a structural change, park it with a ruling.

## Eval tooling polish (E2)

- **E2a.** `make_review.py` becomes a tracked script
  (`scripts/ab_review.py`) and the chain is generated from a TOML spec of arms
  and rounds (`scripts/ab_chain.py`), not hand-edited.
- **E2b.** The C5 tag.

## The R3 eval (built after Parts C, T, K and M; rendered on the user's go-ahead)

- **Cases:** the same 7 cases and seeds. The scripts are new and benefit from
  C, T2 and K.

**Arms:**

| Arm | Env | F5 render? |
|---|---|---|
| `base` | defaults (C, T2, K are ON) | yes |
| `spectral` | `SPECTRAL_DUCK=1` | no (cache) |
| `r2` | `F5_CHUNKING=1 F5_CHUNK_QA=1` | yes |
| `r2_trim` | + `F5_TRIM_LEADIN=1` | yes |
| `cand` | `r2_trim` env + `SPECTRAL_DUCK=1` | no (`r2_trim` cache) |

**Rounds** (2 positions, so 14 pairs each):

| Round | Compares | Question |
|---|---|---|
| A | base vs spectral | Does the spectral duck beat today's mix? |
| B | base vs r2 | Does R2 hold up on fresh scripts? |
| C | r2 vs r2_trim | Does removing the lead-in dead air help? |
| D | base vs cand | Is the whole candidate stack better? |

**Decision rule after the eval:**
- Promote a flag to default-ON if its round wins under the strict rule.
- If the round only leans, the user decides with the full numbers.

## Later phases (order set by this eval)

- **R3b:** deterministic breath-cycle tags (`[cycle:box:4]` padded to exact
  phase lengths), per-family pause curves, per-voice duration regression.
- **R4b:**
  - planner beat table;
  - best-of-3 drafts with a pairwise, order-swapped, rubric judge (targets the
    wording losses on morning and breathwork);
  - beat-signature originality.
- **R5b:**
  - sleep post-roll of 60–180 s with a 20–30 s fade;
  - darkening the bed over the last 30% of sleep stories;
  - persisted recent tracks;
  - CLAP/MuQ music matching;
  - the DeepFilterNet wet A/B.
- **Voice:** the multi-phase reference-clip recordings (spec written in R1),
  and the commercial-engine question.

## Protected invariants

All are unchanged:
- the FATAL/ADVISORY split (K findings are ADVISORY after the budget, C3 is
  FATAL);
- writer/judge independence;
- `<problems>` stripping;
- `unload()` after each LLM stage, so the K call runs before the judge
  unloads;
- no pedalboard `Limiter`;
- the upload stem-separation skip;
- `MOODSCAPE_ADAPTIVE_BED`;
- golden-path audio changes only behind default-OFF flags or after a blind
  win. T1 is a flag. C, T2 and K change scripts, not audio processing.
