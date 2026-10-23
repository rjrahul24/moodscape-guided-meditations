# Generation quality program: close-out (2026-10-23)

The program ran from 2026-10-06 to 2026-10-23. Its roadmap is
`docs/superpowers/specs/2026-10-06-generation-quality-roadmap.md`. It made four
blind listening evals (R1, R2, R3, plus the Qwen3-TTS check) on top of an eval
harness built for the purpose. This note records what shipped, what the ears
decided, what stays opt-in, and what is left.

## What shipped (default ON)

| Area | Change | Evidence |
|---|---|---|
| Eval harness (R0, E) | Script/speech/mix metrics, fixed test set, blind A/B with shuffled pairs and leans, listening page with tags and comments, answer import, sign-test scoring, TOML spec → chain → `REVIEW.md` | Used for R1–R3 |
| Script generation (R1) | Planner/writer/judge fixes, sampling per stage, slop lexicon, safety rules | R1 null by ear; kept as correctness fixes |
| Content safety (Phase 3) | Gentle breathwork line; global FATAL `SENSITIVE_TOPIC` (no medical conditions, pregnancy, illness, hospitals, children, younger self); LLM content check | R3: no wording complaints across 56 pairs (R2 had two) |
| Duration control (Phase 3) | One length repair, then pause scaling ×0.85–1.15 | R3: worst case 6.5 min for a 3–6 min request (R2: 10.2 min) |
| Memory (Phase 3) | TTS unloaded before denoise; MLX cache freed | Whole-pipeline peak 43 GB → 24 GB on 32 GB |
| Loudness (R5a) | Playback-referenced export (−16 / sleep −18; mono 3 dB lower) | Every R3 file −16.0 LUFS as played |
| Mix level (close-out) | Bed 3 LU further under the voice (meditation 33.5/17.5 LU, sleep 25/19) | R3: "music loud in every test" |
| Library (close-out) | `Valley of Voices` removed (moved to `var/removed_backgrounds/`) | R3: unusable bed regardless of level |
| No vocal sounds | Breath markers render as silence in every engine (ElevenLabs track, cbed7ea) | R3: "sound of breathing out" came from breath samples |
| Voice provider | ElevenLabs v4 cloud TTS behind the Voice Provider toggle, with Remix | Separate spec: `specs/2026-10-12-elevenlabs-v4-tts-design.md` |

## What the ears decided (all stay OFF, opt-in by env)

| Flag | R2 | R3 | Decision |
|---|---|---|---|
| `MOODSCAPE_SPECTRAL_DUCK` | 0.63 for (vs a shallow duck) | 0.385 against (vs today) | Not adopted |
| `MOODSCAPE_F5_CHUNKING` + `_CHUNK_QA` | 3–0 clear, 0.77 | 2–4 clear, 0.375 | Not adopted; halves machine-counted mid-word gaps but no by-ear win |
| `MOODSCAPE_F5_TRIM_LEADIN` | — | 0.615, p 0.31 | Not adopted |
| `MOODSCAPE_BED_RISE_LIMIT` | 0.46 | — | Not adopted |
| `MOODSCAPE_F5_CFG=1.2` | 0.60 | — | Not adopted (UI slider still offers it) |
| Full candidate stack | — | 0.312 | Today's defaults stay the golden path |
| Per-family voices (`MOODSCAPE_VOICE_BY_FAMILY`) | R1 null | — | Brittney stays default |
| Qwen3-TTS | F5 5/5 vs Qwen ≤3 | — | Parked on `dev-qwen-tts-temp` (pushed) |

Never tested by ear on the golden path, and closed as opt-in experiments:
`MOODSCAPE_SHARED_REVERB`, `MOODSCAPE_F5_MICROPROSODY`,
`MOODSCAPE_F5_REF_PRESERVE_DYNAMICS`, the Kokoro jitter/parallel-compression
flags. The program's evidence is that F5's robotic quality is the binding limit,
and no local post-processing knob has moved it in a blind test.

Scored results: `var/eval/r1/RESULTS.md`, `var/eval/r2/RESULTS.md`,
`var/eval/r3/RESULTS.md` (machine-local, gitignored).

## Rulings at close-out (cost if wrong)

1. **The bed moves 3 LU quieter for both content types, without a blind round.** The listener's note was unanimous across rounds, and the duck depth is unchanged. Cost if wrong: the music reads as too subtle; `MOODSCAPE_BED_*_OFFSET_LU` restores any level without code.
2. **Sleep stories get the same −3 LU, although R3 was meditation-only.** Cost if wrong: sleep beds are slightly too quiet; same env override.
3. **The track is moved out of the library, not deleted.** The mp3s are gitignored, so git cannot restore them. Cost if wrong: none.
4. **Pause-dominated length repair: proportional target, not a skip.** The parked fix ("skip when pauses ≥ midpoint") would also have skipped a long script built of repeated paragraphs, where cutting passages works. The repair now asks to cut whole passages with their pauses, at the whole-script ratio. Cost if wrong: one repair whose target is approximate.
5. **No experiment is promoted, and none is deleted.** Every losing flag stays as an opt-in env var, documented with its result. Cost if wrong: some dead-by-default code paths to remove later.
6. **The close-out ran inline with TDD, plus one final review**, instead of the full spec → plan → subagent cycle. It was four small, listening-driven changes, the same precedent as the R1 follow-up. Cost if wrong: less review depth on about 80 changed lines.

## Robotic delivery: where it stands

- F5 local tuning is exhausted by the evidence. Chunking, QA re-renders,
  cfg, lead-in trim and the mix changes all came out null or slight by ear, and
  every note still says "robotic".
- The remaining local lever is the **reference recording**. F5 copies the
  prosody of its reference clip, so a longer, genuinely meditative and expressive
  read, possibly multi-phase (`assets/speakers/voices.toml`), is the only
  untried F5 input of substance. It needs a human recording session.
- The **ElevenLabs v4 provider** is the answer to the commercial-engine question,
  and is already integrated. The next decision is a blind F5 vs ElevenLabs round
  on the same scripts. `eval_genres.py --engine elevenlabs` supports it, and the
  ElevenLabs chunk cache makes re-runs free. It bills per character.

## Known deferred minors

Every accepted minor is listed with its reason in the per-phase notes:
`2026-10-r0-deferred-minors.md`, `2026-10-r1-deferred-minors.md`,
`2026-10-e-r5a-r2-deferred-minors.md`, `2026-10-phase3-deferred-minors.md`.
None affects the default generation path's correctness. The phase-3 items
"parked length repair" and "private `_strip_tags` import" are fixed at
close-out.
