# R1 script-gen fixes — deferred minors and rulings

Carried forward from the R1 subagent-driven ledger (2026-10-07).

## Rulings
- Ruling: Execute on `dev` in the main checkout (user's established preference for this program) — cost if wrong: commits movable to a branch.
- Ruling: No real-model runs in R1 (unit tests + one cheap librosa voice measurement only); the evaluation run (fresh baseline + 2 challengers) is NOT started — the user stopped background renders over fan noise and wants to give the go-ahead — cost if wrong: evaluation starts later.
- Ruling: Task 4 reviewer rated safety false negatives Minor (plan-mandated regexes); controller upgrades the clear variants ("as quickly as you can/possible", "until it fully disappears", "until the pain is gone", "pain will fade/melt away / be gone", "energy can/could heal") to a fix round because these are mental-health guardrails and the spec intent covers them — cost if wrong: slightly broader FATAL rules (false-positive tests guard it).
- Ruling: Fold Task 6 minors #2-#6 into fix round 1 (duration clause from DEFAULT_WPM, Kokoro 3.8s note, slop in examples, Kokoro ellipsis recommendation, line-anchored fence regex covering all 4 guides) — they directly alter what the writer is told, which is R1's purpose; planner clause uses min-max of DEFAULT_WPM — cost if wrong: a slightly larger fix diff.
- Ruling: Fix wave = Important 1+2, Minors 5 (repair prompt self-contradiction), 6 ("Welcome." opener), 7 ("...body." literal), 9 (voice in eval rows), 10 (shared STYLE_CODES), plus T1/T8 temperature-none doc/behaviour clarity. Important 3 handled by interpretation: judge the duration criterion on the Brittney attribution arm — cost if wrong: none (no code).
- Ruling: The reviewer's recommended real-model script-only smoke test (presence_penalty 1.5 vs markup) is NOT run now — user is away and asked to stop heavy background work; surfaced as the first step of the evaluation plan — cost if wrong: a markup regression would surface in the eval instead of earlier.
- Ruling: Deviate from "one fix wave" for a single narrow corrective dispatch, because the residual is a safety-gate regression introduced by the fix wave itself (load-bearing): drop rhythm|pace|own from the action branch, require condition + action within the same sentence, add the reviewer's 4 negative openings; fold in the Kokoro "Welcome." opener and reword the planner hold line as a ceiling ("Do not add breath holds unless the technique calls for them; any hold stays at or under N seconds"). Then one scoped re-review — cost if wrong: one extra small round.

## Deferred minors
- Task 1: minor (deferred): TEMPERATURE=none silently falls back to 1.0 (untested/undocumented); nan/inf accepted from env
- Task 2: minor (deferred): TestSamplingForwarding appended after `if __name__ == "__main__"` in test_generator_judge.py (pytest still collects); import order; >100-char lines
- Task 3: minor (deferred): new test classes appended after __main__ guard (test_script_linter.py, test_auto_generate.py); ("SLOP_DENSITY","SLOP_HARD") tuple duplicated in linter + auto_generate; check_slop long lines / double hit computation
- Task 4: minor (deferred): residual regex false negatives ("until the pain is completely gone", "twenty rapid breaths" by design, "pant quickly"); contrived false positives ("energy repairs itself"); safety-rules doc list indentation for 10-13. A future LLM safety classifier (R4) is the real backstop.
- Task 5: minor (deferred, candidate for final fix wave): MISSING_DISCLAIMER action regex misses "breathe naturally"/"regular breathing" -> false FATAL costs a repair; condition+action matched independently across opening (loose); report said self_compassion=10 (actually 9)
- Task 6: minor (deferred): PARA_PAUSE drift-test assertion weak ("2","3","4" match anything); "Welcome." 1-word opener; sleep "<~15 words" stricter than linter band
- Task 6: minor (deferred, final-wave candidates): F5 meditation guide inline "...body." literal; Kokoro sleep guide section 4 still recommends ellipses "sparingly" (contradicts its own no-ellipses note)
- Task 7: minor (deferred): voice cache never prunes removed voices; non-atomic cache write
- Task 7: minor (deferred): real_voice_selection marker unused; cache-isolation test partly tautological
- Task 8: minor (deferred): README "none unsets" not true for TEMPERATURE (falls back to 1.0); recording spec omits energy/steady derivation; pre-existing voices.toml comment says reference_transcript/ (real dir reference_text/)
- Final: minor (deferred): disclaimer condition misses "heart issue(s)", "heart or lung condition", "heart ailment" (costs one repair; canonical line passes); a disclaimer split across two sentences fails by design
