# Qwen3-TTS (Track Q) — Findings, 2026-10-06

Sandbox evaluation of Qwen3-TTS on mlx-audio against the F5 golden-path voice, run on the M1 Max (32 GB). **Nobody has listened to these renders yet.** Everything below is measured by machine; the by-ear section is a checklist for the user's listening test, not a report of one.

## Model repos used

| Mode | Repo | Size on disk |
|---|---|---|
| clone | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | 4.2 GB |
| custom | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | 4.2 GB |
| design | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 4.2 GB |

All three resolved and loaded on the first try; no `MODEL_REPOS` change was needed. The mlx-audio `generate`, `generate_custom_voice` and `generate_voice_design` signatures accept every keyword the engine passes (`temperature`, `language`/`lang_code`, `instruct`, `speaker`), so no call-shape change was needed either. Loading prints two upstream warnings (a model-type notice and a tokenizer "incorrect regex pattern" notice); neither caused a failure.

## Live smoke tests

`MOODSCAPE_QWEN3_LIVE=1 .venv/bin/python -m pytest tests/integration/test_qwen3_live.py -v -s` (three-sentence script, two pauses totalling 7 s, voice Brittney, seed 42):

All three modes passed (`3 passed in 28.59s`, weights already in the HF cache so no download happened). Per-test wall time includes model load and unload:

| Mode | Result | Test wall (s) | Audio |
|---|---|---|---|
| clone | PASS | 15.87 | between 11 s and 37 s, no NaN, ZCR < 0.4, voice activity present |
| custom | PASS | 6.64 | same checks |
| design | PASS | 5.87 | same checks |

The exact audio duration per mode is not printed by the test (it only asserts the 11-37 s window). The `Qwen3 synth ... RTF=` INFO line was not captured because pytest `-s` does not enable that logger; per-mode RTF comes from the full-pipeline renders below.

## Full-pipeline renders

Same code path as the Sandbox tab: `MeditationPipeline().generate(script=BENCHMARK_SCRIPTS[k], content_type=k, tts_engine=…, f5_voice_slug="Brittney", reverb_amount=0.10, duck_amount_db=-16.0, music_model="upload", generation_mode="Vocals Only", seed=42, use_tts_cache=False)`, with `speed=1.0` for Qwen3 (ignored upstream) and `speed=0.80` for the F5 reference. Defaults otherwise (temperature 0.7, chunk 200, stretch 1.0, DF wet 0.5, speaker `serena`, preset instructs). Each render ran in its own process under `/usr/bin/time -l`.

### Speed and memory

TTS RTF = TTS synthesis wall time ÷ audio seconds (lower is faster). "Peak footprint" is `/usr/bin/time -l`'s *peak memory footprint*, which includes Metal/GPU allocations; *max RSS* does not, so it understates MLX/MPS use. GB = 10⁹ bytes. Both figures are for the whole pipeline process (TTS + DeepFilterNet + FX + export), not the TTS model alone.

| Render | Audio (s) | TTS wall (s) | TTS RTF | Pipeline wall (s) | Max RSS (GB) | Peak footprint (GB) |
|---|---|---|---|---|---|---|
| clone · meditation | 265.4 | 86.7 | 0.33 | 109.3 | 4.56 | 19.2 |
| clone · sleep story | 226.8 | 106.4 | 0.47 | 124.8 | 5.10 | 20.7 |
| custom · meditation | 291.1 | 71.3 | 0.25 | 94.9 | 8.03 | 19.0 |
| custom · sleep story | 331.1 | 136.1 | 0.41 | 163.5 | 5.07 | 21.5 |
| design · meditation | 281.9 | 67.3 | 0.24 | 89.2 | 8.28 | 18.6 |
| design · sleep story | 279.4 | 106.5 | 0.38 | 128.6 | 7.22 | 18.8 |
| **F5 · meditation** | 367.9 | 452.3 | 1.23 | 490.4 | 6.18 | 32.7 |
| **F5 · sleep story** | 371.4 | 617.5 | 1.66 | 658.3 | 6.48 | 39.9 |

- Qwen3 synthesises **3–7× faster than F5** (RTF 0.24–0.47 vs 1.23–1.66) and the whole pipeline finishes in 1.5–3 minutes instead of 8–11.
- Qwen3's peak footprint (18.6–21.5 GB) is well below F5's (32.7–39.9 GB; above 32 GB means compressed memory was in use, though `time` reported 0 swaps). No Qwen3 run came near the 32 GB ceiling, but ~19 GB is far more than the 4.2 GB of weights — most likely MLX's buffer cache growing across chunks. `mx.set_cache_limit` is a possible follow-up if Qwen3 ever has to share the machine with anything else.

### Chunk failures and retries

Zero. No render logged a `Qwen3 produced empty audio … retrying` line, and no Qwen3 render logged any `WARNING` at all. Every render passed the pipeline QA (the QA LUFS check reads −21.0, which is the known mono-LUFS quirk; the exported files measure −16.0 LUFS integrated, −1.3 dBFS sample peak).

The 200-character chunk cap was only binding on the sleep story: its 15 speech blocks (longest 260 chars) became 20 chunks. The meditation's 21 blocks (longest 119 chars) were never split.

### Objective proxies (instead of listening)

Measured on the raw 24 kHz TTS output (before FX) unless noted. Speech = 50 ms frames above −40 dBFS. WPM uses the script's word count. Pitch is `librosa.pyin` (60–400 Hz) on the first 60 s.

| Render | Final (s) | vs F5 | Speech (s) | Speech ratio | WPM (speech only) | Chunk-to-chunk level std / range (dB) | Frame level std (dB) | Median f0 (Hz) | f0 IQR (semitones) |
|---|---|---|---|---|---|---|---|---|---|
| clone · meditation | 265.4 | −28% | 88.0 | 0.33 | 206 | 1.04 / 5.3 | 3.9 | 193 | 2.8 |
| custom · meditation | 291.1 | −21% | 106.0 | 0.36 | 171 | 1.78 / 10.4 | 5.4 | 221 | 3.6 |
| design · meditation | 281.9 | −23% | 98.2 | 0.35 | 185 | 2.27 / 9.4 | 6.7 | 162 | 3.3 |
| F5 · meditation | 367.9 | — | 115.2 | 0.31 | 157 | 1.96 / 9.5 | 6.4 | (unreliable) | (unreliable) |
| clone · sleep story | 226.8 | −39% | 122.4 | 0.54 | 202 | 1.38 / 6.5 | 4.0 | 192 | 2.6 |
| custom · sleep story | 331.1 | −11% | 183.2 | 0.55 | 135 | 1.82 / 9.4 | 5.5 | 225 | 2.3 |
| design · sleep story | 279.4 | −25% | 151.4 | 0.54 | 163 | 3.34 / 15.0 | 6.9 | 189 | 3.0 |
| F5 · sleep story | 371.4 | — | 165.0 | 0.44 | 150 | 1.83 / 9.4 | 6.5 | (unreliable) | (unreliable) |

What the numbers suggest (to be confirmed by ear):

- **Pacing.** Clone mode speaks fastest (~205 WPM while speaking, about 30% faster than F5's 150–157). It has no instruct, so it likely inherits the reference clip's conversational pace. Custom mode responds to its instruct: 171 WPM for the meditation and 135 WPM for the "very slowly" sleep-story preset, the only Qwen3 render slower than F5. Design is in between (163–185).
- **Length.** Every Qwen3 render is shorter than F5's (−11% to −39%) and the 5-minute target (300 s); clone's sleep story is only 227 s. The pauses are identical across engines, so the whole gap is speech pace.
- **Level consistency between chunks.** Clone is the most even chunk-to-chunk (std 1.0–1.4 dB). Design's sleep story has the widest spread (std 3.3 dB, 15 dB range), which is a risk of audible jumps in level at chunk boundaries.
- **Dynamics / monotony proxy.** Clone has the flattest within-speech level (frame std 3.9–4.0 dB vs F5's 6.4–6.5) and a narrow pitch IQR (2.6–2.8 st). That fits either "steady and calm" or "monotonous over 5 minutes"; only listening can tell which.
- **Voice.** Custom's `serena` sits highest (median ~220 Hz); design's narrator is lowest on the meditation (162 Hz). Clone tracks Brittney's reference at ~192 Hz.
- **Pitch for F5 was not comparable.** pyin's output on F5's audio is bimodal (IQR ~20 semitones, i.e. constant octave confusion, probably from its breathier output), so no pitch comparison against F5 is made.

### Intelligibility (WER)

Whisper `openai/whisper-large-v3-turbo` (fp16, MPS, long-form decoding) transcribed the raw 24 kHz TTS output of the four **meditation** renders. WER is against the script text with `[...]` markers removed, lower-cased with punctuation stripped (302 reference words; no jiwer, plain word-level edit distance). One script per configuration, one seed.

| Render | WER | Hypothesis words |
|---|---|---|
| F5 · meditation | 0.0% | 302 |
| clone · meditation | 0.33% (1 word) | 302 |
| custom (`serena`) · meditation | 0.33% (1 word) | 302 |
| design · meditation | 0.33% (1 word) | 302 |

All four are essentially perfectly intelligible to Whisper; the one-word differences are within noise for a single run and were not inspected. This says nothing about naturalness. The sleep stories were not transcribed.

## By ear — pending the user's listening test

Not done; I cannot listen. WAVs (48 kHz, vocals only, gitignored) are in `var/qwen3_renders/` of the `qwen3-tts` worktree:

| | Meditation | Sleep story |
|---|---|---|
| F5 (reference) | `var/qwen3_renders/f5_meditation.wav` | `var/qwen3_renders/f5_sleep_story.wav` |
| clone | `var/qwen3_renders/clone_meditation.wav` | `var/qwen3_renders/clone_sleep_story.wav` |
| custom | `var/qwen3_renders/custom_meditation.wav` | `var/qwen3_renders/custom_sleep_story.wav` |
| design | `var/qwen3_renders/design_meditation.wav` | `var/qwen3_renders/design_sleep_story.wav` |

Checklist per render, compared against F5:

- [ ] **Warmth**: body and closeness of the voice; any thin or codec-y timbre.
- [ ] **Pacing**: does clone's ~205 WPM feel rushed for a meditation? Is custom's 135 WPM sleep story too slow or just right?
- [ ] **Artifacts**: clicks, glitches, slurred or swallowed words, stray syllables at chunk edges, level jumps between chunks (especially design · sleep story).
- [ ] **Monotony over the full 5 minutes**: does clone's flat level and pitch become droning?
- [ ] **Pronunciation**: names, "inhale/exhale", and any mangled words.
- [ ] **Accent (custom mode)**: `serena` is listed upstream as a Chinese-native preset. Listen for a non-native English accent. Machine check: Whisper recovered the `serena` render with 0.33% WER (the same as clone and design), so there is no sign of mispronunciation that breaks recognition; subtle accent is below what WER can detect, so this still needs an ear. If it is noticeable, alternatives are `vivian` (also Chinese-native, female), `ono_anna` (Japanese-native, female) or `sohee` (Korean-native, female); the only English-native presets (`ryan`, `aiden`) are male. Design mode, which describes the narrator in English, sidesteps the preset-speaker question altogether.
- [ ] **Identity**: does clone sound like Brittney (the F5 voice)?

## Not measured

- WER on the sleep stories, on other seeds, or on a second reciter model.
- Any listening judgement (warmth, naturalness, accent, monotony): pending the user.
- Different temperatures, chunk sizes or `MOODSCAPE_QWEN3_STRETCH` values: every render used the defaults (0.7 / 200 / 1.0).
- Other custom speakers than `serena`, and other clone voices than Brittney.
- Render-to-render variance: one seed (42) per configuration. F5 is not seeded by the pipeline (known nondeterminism).
- Memory of the TTS model in isolation: the figures cover the whole pipeline process.
- Music-bed behaviour: all renders were "Vocals Only".

## Recommendation (provisional, until the user listens)

**Provisional: based only on the objective data above. Nobody has listened, and this changes once the user does.**

- **Mode for the R0 A/B: `custom` (speaker `serena`, default instruct) first, `design` second.** Reasons from the data: both were intelligible (WER 0.33%), both have more F5-like dynamics (frame level std 5.4-6.9 dB vs F5 6.4-6.5), and custom is the only mode whose pace responds to the instruct (171 WPM meditation, 135 WPM sleep story vs F5's 150-157). Design is the safer fallback if the `serena` accent is audible, since it needs no preset speaker and sat lowest in pitch. Clone is the only mode that reproduces the F5 voice (Brittney), which makes it the fairest voice-identity comparison, so it should stay in the A/B as the third arm even though it is the least likely to win on pacing.
- **Temperature 0.7 (current default), chunk 200: keep both.** Zero chunk failures at these values, and the 200-character cap only split the sleep story's long sentences (15 blocks into 20 chunks). Other values were not tried, so there is no evidence to change them. If design's sleep story shows audible level jumps between chunks (its chunk-level std was 3.3 dB, the widest), the first thing to try is a lower temperature (0.5-0.6) before changing the chunk size.
- **Stretch: suggestion only, not measured.** Clone speaks about 205 WPM against F5's 150-157, and its runs are 11-39% shorter than F5's. If the user finds clone rushed, add one clone arm with `MOODSCAPE_QWEN3_STRETCH=0.80` (roughly 205 x 0.80 = 164 WPM, if the stretch behaves linearly, which was not checked, and with the risk that time-stretching adds artifacts that were not listened for). Custom and design should be A/B'd at stretch 1.0.
- **Practical argument for pursuing it:** Qwen3's TTS RTF is 0.24-0.47 vs F5's 1.23-1.66, and its peak footprint is 19-21.5 GB vs F5's 33-40 GB. Even a modest listening result may justify promotion; a clear loss on warmth should end the experiment.
