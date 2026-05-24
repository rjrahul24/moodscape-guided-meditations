# Kokoro TTS Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply six targeted Kokoro TTS improvements — pyworld scope fix, G2P lexicon overrides, stress markers, pure_calm voice preset, conservative FX tweaks, and documentation reconciliation — to produce more expressive, studio-quality guided meditation audio.

**Architecture:** All changes are localized to `core/kokoro_tts/` (engine, preprocessor, postprocessor, voice_manager) and `app.py`. No changes touch F5-TTS, Chatterbox, ACE-Step, HeartMuLa, Lyria, mixer, or QA monitor. Documentation updates reflect code reality.

**Tech Stack:** Python 3.11, Kokoro/misaki G2P, pyworld ≥0.3.4, Pedalboard ≥0.9, PyTorch, pytest (run via `.venv/bin/python -m pytest unit-tests/ -v`)

**Spec:** `docs/superpowers/specs/2026-05-24-kokoro-tts-pipeline-improvements-design.md`

---

## File Map

| File | Change Type | What Changes |
|------|------------|-------------|
| `core/kokoro_tts/engine.py` | Modify | Move `humanize_voice` to top-level import + per-chunk call; add `_apply_meditation_lexicon()` |
| `core/kokoro_tts/preprocessor.py` | Modify | Add `_STRESS_REDUCE`, `_STRESS_BOOST`, `_apply_stress_markers()`; call in `preprocess_for_meditation()` |
| `core/kokoro_tts/voice_manager.py` | Modify | Add `blend_with_extrapolation()`; add `pure_calm` preset; update `get_voice()` |
| `core/kokoro_tts/postprocessor.py` | Modify | Compressor threshold -22 → -28 dB; reverb default 0.15 → 0.18 |
| `app.py` | Modify | Add `pure_calm` to `KOKORO_VOICE_CHOICES` |
| `unit-tests/test_tts_engines.py` | Modify | Add pyworld scope test + G2P lexicon tests |
| `unit-tests/test_text_preprocessor.py` | Modify | Add stress marker tests |
| `unit-tests/test_voice_manager.py` | Modify | Add `blend_with_extrapolation` + `pure_calm` preset tests |
| `unit-tests/test_kokoro_postprocessor.py` | Modify | Add compressor threshold + reverb level tests |
| `docs/model_implementation_guides/kokoro_tts.md` | Modify | Fix humanize_voice status; update FX values |
| `docs/prompting_guides/vocal_kokoro_instructions.md` | Modify | Fix paragraph pause 2.5s → 6.5s |
| `CLAUDE.md` | Modify | Add Stage 2c; update FX table; add constants |
| `docs/optimization_and_processing/post-processing-pipeline.md` | Modify | Add humanize stage; update compressor/reverb |

---

## Task 1: Fix pyworld humanize_voice scope

**Files:**
- Modify: `core/kokoro_tts/engine.py`
- Test: `unit-tests/test_tts_engines.py`

`humanize_voice` is currently imported inline at the end of `synthesize()` and called on the full assembled audio (including room-tone pauses). It must move to: (1) a top-level import, (2) a per-speech-chunk call immediately after `apply_segment_fades()`.

- [ ] **Step 1: Write the failing test**

  Add to `unit-tests/test_tts_engines.py`:

  ```python
  import unittest
  from unittest.mock import MagicMock, patch
  import numpy as np
  import torch

  from core.kokoro_tts.engine import KokoroEngine, SAMPLE_RATE


  def _fake_voice() -> torch.Tensor:
      """Deterministic fake voice tensor matching Kokoro's (511, 1, 256) shape."""
      torch.manual_seed(0)
      return torch.randn(511, 1, 256)


  class TestHumanizeVoiceScope(unittest.TestCase):
      """humanize_voice must be called per speech sentence, not on full assembly."""

      def _make_engine_with_mock_pipeline(self):
          """Return a KokoroEngine with a mocked KPipeline that yields 0.3s audio."""
          engine = KokoroEngine()
          fake_audio = np.ones(int(0.3 * SAMPLE_RATE), dtype=np.float32) * 0.1
          mock_pipe = MagicMock()
          mock_pipe.return_value = [("graphemes", "phonemes", fake_audio)]
          engine.pipeline = mock_pipe
          engine.pipeline_en_gb = None
          return engine

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      @patch('core.kokoro_tts.voice_manager.get_voice', return_value='af_heart')
      @patch('core.kokoro_tts.engine.humanize_voice')
      def test_humanize_called_per_speech_sentence_not_on_full_audio(
          self, mock_humanize, mock_get_voice, mock_load_tensor
      ):
          mock_load_tensor.return_value = _fake_voice()
          mock_humanize.side_effect = lambda audio, sr: audio  # passthrough

          engine = self._make_engine_with_mock_pipeline()
          segments = [
              {"type": "speech", "text": "Breathe in deeply."},
              {"type": "pause", "duration_sec": 3.0},
              {"type": "speech", "text": "Now exhale slowly."},
          ]
          voice_audio, _ = engine.synthesize(segments, voice="af_heart")

          # Must be called once per speech sentence (2 sentences → 2 calls)
          self.assertEqual(
              mock_humanize.call_count, 2,
              f"Expected 2 calls (one per speech sentence), got {mock_humanize.call_count}"
          )
          # Each call must receive a chunk shorter than the full assembled audio
          total_len = len(voice_audio)
          for i, call in enumerate(mock_humanize.call_args_list):
              chunk_len = len(call[0][0])  # first positional argument
              self.assertLess(
                  chunk_len, total_len,
                  f"Call {i}: humanize_voice received full-length audio ({chunk_len} >= {total_len})"
              )
  ```

- [ ] **Step 2: Run the test — verify it fails**

  ```bash
  cd /Users/rahul/Downloads/moodscape-guided-meditations
  .venv/bin/python -m pytest unit-tests/test_tts_engines.py::TestHumanizeVoiceScope -v
  ```

  Expected: `FAILED` — `AttributeError: module 'core.kokoro_tts.engine' has no attribute 'humanize_voice'` (because `humanize_voice` is not yet a top-level name in the engine module).

- [ ] **Step 3: Update the import block in engine.py**

  In `core/kokoro_tts/engine.py`, find the existing postprocessor import block:

  ```python
  from core.kokoro_tts.postprocessor import (
      CROSSFADE_SAMPLES,
      process_chunk,
      crossfade_chunks,
      apply_segment_fades,
      reduce_synthesis_noise,
      generate_room_tone,
  )
  ```

  Replace with:

  ```python
  from core.kokoro_tts.postprocessor import (
      CROSSFADE_SAMPLES,
      process_chunk,
      crossfade_chunks,
      apply_segment_fades,
      reduce_synthesis_noise,
      generate_room_tone,
      humanize_voice,
  )
  ```

- [ ] **Step 4: Move humanize_voice call to per-chunk scope**

  In `core/kokoro_tts/engine.py`, inside `synthesize()`, find the speech assembly block:

  ```python
                  if speech_parts:
                      speech_audio = crossfade_chunks(speech_parts)
                      speech_audio = apply_segment_fades(speech_audio)
                  else:
                      speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                  audio_chunks.append(speech_audio)
                  activity_chunks.append(np.ones(len(speech_audio), dtype=bool))
  ```

  Replace with:

  ```python
                  if speech_parts:
                      speech_audio = crossfade_chunks(speech_parts)
                      speech_audio = apply_segment_fades(speech_audio)
                      # Stage 2c: Pitch humanization per speech chunk.
                      # Applied here (not on full assembly) so pyworld never
                      # processes room-tone silence — avoids pitch-tracker artifacts
                      # at speech→pause boundaries.
                      speech_audio = humanize_voice(speech_audio, sr=SAMPLE_RATE)
                  else:
                      speech_audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

                  audio_chunks.append(speech_audio)
                  activity_chunks.append(np.ones(len(speech_audio), dtype=bool))
  ```

- [ ] **Step 5: Remove the old end-of-synthesize humanize_voice call**

  Find and delete these lines near the end of `synthesize()`:

  ```python
          # Stage 2c: Pitch humanization + formant warmth via pyworld.
          # Adds micro-pitch drift (~0.5 Hz / ±6 cents), subtle vibrato (5 Hz / ±3 cents),
          # random jitter (±2 cents), and 3% formant shift — all invisible to the ear
          # but transforming the robotic flatness of TTS into perceived expressiveness.
          from core.kokoro_tts.postprocessor import humanize_voice
          voice_audio = humanize_voice(voice_audio, sr=SAMPLE_RATE)
  ```

  The block should now end with:

  ```python
          # Stage 2b: Spectral gating — remove low-level ISTFTNet synthesis hiss.
          # Runs on the full assembled audio so the noise profile estimate is stable.
          voice_audio = reduce_synthesis_noise(voice_audio, sr=SAMPLE_RATE)

          return voice_audio, voice_activity
  ```

- [ ] **Step 6: Run the test — verify it passes**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_tts_engines.py::TestHumanizeVoiceScope -v
  ```

  Expected: `PASSED`

- [ ] **Step 7: Run the full unit-test suite — verify no regressions**

  ```bash
  .venv/bin/python -m pytest unit-tests/ -v
  ```

  Expected: all previously-passing tests still pass.

- [ ] **Step 8: Commit**

  ```bash
  git add core/kokoro_tts/engine.py unit-tests/test_tts_engines.py
  git commit -m "fix(kokoro): apply humanize_voice per speech chunk, not on full assembly

  Moving humanize_voice from end-of-synthesize (full audio) to per-sentence
  scope prevents pyworld from processing room-tone silence segments. Avoids
  pitch-tracker artifacts at speech→pause boundaries. Adds module-level import.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Task 2: Add global G2P lexicon overrides

**Files:**
- Modify: `core/kokoro_tts/engine.py`
- Test: `unit-tests/test_tts_engines.py`

Add `_apply_meditation_lexicon()` to `KokoroEngine`. Called once after the American English `KPipeline` is created. Injects elongated IPA for 12 core English meditation words directly into `pipeline.g2p.lexicon.golds`.

- [ ] **Step 1: Write the failing tests**

  Append to `unit-tests/test_tts_engines.py` (after `TestHumanizeVoiceScope`):

  ```python
  class TestMeditationLexicon(unittest.TestCase):
      """_apply_meditation_lexicon must write IPA overrides into pipeline.g2p.lexicon.golds."""

      def _make_engine(self):
          return KokoroEngine()

      def test_core_words_injected(self):
          engine = self._make_engine()
          mock_pipe = MagicMock()
          mock_pipe.g2p.lexicon.golds = {}

          engine._apply_meditation_lexicon(mock_pipe)

          for word in ("breathe", "exhale", "inhale", "relax", "release",
                       "soften", "surrender", "dissolve", "melt", "sink", "drift"):
              self.assertIn(
                  word, mock_pipe.g2p.lexicon.golds,
                  f"Expected '{word}' in g2p.lexicon.golds after _apply_meditation_lexicon"
              )

      def test_ipa_values_are_strings(self):
          engine = self._make_engine()
          mock_pipe = MagicMock()
          mock_pipe.g2p.lexicon.golds = {}

          engine._apply_meditation_lexicon(mock_pipe)

          for word, ipa in mock_pipe.g2p.lexicon.golds.items():
              self.assertIsInstance(ipa, str, f"IPA for '{word}' must be a str, got {type(ipa)}")
              self.assertGreater(len(ipa), 0, f"IPA for '{word}' must be non-empty")

      def test_missing_golds_attr_does_not_raise(self):
          """If g2p.lexicon.golds is not accessible, method silently logs a warning."""
          engine = self._make_engine()

          class _NoPipeline:
              pass  # no g2p attribute at all

          # Must not raise — AttributeError is caught internally
          try:
              engine._apply_meditation_lexicon(_NoPipeline())
          except AttributeError:
              self.fail("_apply_meditation_lexicon raised AttributeError instead of catching it")
  ```

- [ ] **Step 2: Run the tests — verify they fail**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_tts_engines.py::TestMeditationLexicon -v
  ```

  Expected: `FAILED` — `AttributeError: 'KokoroEngine' object has no attribute '_apply_meditation_lexicon'`

- [ ] **Step 3: Add _apply_meditation_lexicon to engine.py**

  In `core/kokoro_tts/engine.py`, add the method to `KokoroEngine` (between `_get_pipeline` and `unload_model`):

  ```python
      # ── Meditation lexicon constants ──────────────────────────────────────
      # Maps English meditation words → elongated IPA variants using American
      # English phonemes. Written once into pipeline.g2p.lexicon.golds at
      # load_model() time — overrides default conversational pronunciations
      # globally for all subsequent synthesis calls.
      # Only applied to the American English pipeline (lang_code='a');
      # the British pipeline uses non-rhotic IPA which differs.
      _MEDITATION_LEXICON: dict[str, str] = {
          # Core breath words — elongated vowels for resonance
          "breathe":    "bɹiːːð",     # hyper-elongated 'ee'
          "breath":     "bɹɛːθ",      # elongated open vowel
          "exhale":     "ɛksˈheɪːl",  # elongated 'ay' on release
          "inhale":     "ɪnˈheɪːl",   # elongated 'ay' on intake
          # Relaxation verbs — flattened stress, elongated vowels
          "relax":      "ɹɪˈlæːks",   # elongated 'a'
          "release":    "ɹɪˈliːːs",   # hyper-elongated 'ee'
          "soften":     "ˈsɔːfən",    # elongated 'o' for warmth
          "surrender":  "səˈɹɛndə",   # reduced stress, trailing schwa
          "dissolve":   "dɪˈzɒːlv",   # elongated 'o'
          "melt":       "mɛːlt",      # elongated vowel
          "sink":       "sɪːŋk",      # elongated vowel
          "drift":      "dɹɪːft",     # elongated vowel
      }

      def _apply_meditation_lexicon(self, pipe) -> None:
          """Inject elongated IPA overrides for core English meditation words.

          Writes directly into the misaki G2P lexicon's 'golds' dict, which
          takes priority over all dictionary lookups. Applied once at
          model-load time — no per-sentence text manipulation needed.

          Only called on the American English pipeline (lang_code='a').
          The British pipeline is intentionally excluded because the IPA
          uses rhotic American phonemes (e.g., bɹiːːð with ɹ) that would
          produce incorrect non-rhotic pronunciation in British English.

          Args:
              pipe: A KPipeline instance. If its g2p.lexicon.golds attribute
                    is inaccessible (e.g., future misaki API change), an
                    AttributeError is caught and a warning is logged so
                    synthesis continues with default pronunciations.
          """
          try:
              lexicon = pipe.g2p.lexicon.golds
              for word, ipa in self._MEDITATION_LEXICON.items():
                  lexicon[word] = ipa
              logger.info(
                  "[Kokoro] Meditation G2P lexicon applied: %d word overrides",
                  len(self._MEDITATION_LEXICON),
              )
          except AttributeError as exc:
              logger.warning(
                  "[Kokoro] G2P lexicon override skipped — attribute path "
                  "g2p.lexicon.golds not accessible (%s). "
                  "Default pronunciations will be used.", exc,
              )
  ```

- [ ] **Step 4: Call _apply_meditation_lexicon after pipeline creation in load_model()**

  In `load_model()`, after `self.pipeline.model.to(device)`:

  ```python
          self.pipeline = KPipeline(
              lang_code="a",
              repo_id="hexgrad/Kokoro-82M",
              trf=True,
              device=device,
          )
          if hasattr(self.pipeline, "model") and self.pipeline.model is not None:
              self.pipeline.model.to(device)
          # Apply meditation G2P lexicon overrides (American English only)
          self._apply_meditation_lexicon(self.pipeline)
  ```

- [ ] **Step 5: Run the tests — verify they pass**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_tts_engines.py::TestMeditationLexicon -v
  ```

  Expected: `PASSED`

- [ ] **Step 6: Run full suite — verify no regressions**

  ```bash
  .venv/bin/python -m pytest unit-tests/ -v
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add core/kokoro_tts/engine.py unit-tests/test_tts_engines.py
  git commit -m "feat(kokoro): add global G2P lexicon overrides for meditation words

  Injects elongated IPA pronunciations for 12 core English meditation words
  (breathe, exhale, relax, release, etc.) directly into misaki's golds dict
  at pipeline load time. Gracefully falls back to default pronunciations if
  the g2p.lexicon.golds path changes in a future misaki version.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Task 3: Add stress reduction / boost markers

**Files:**
- Modify: `core/kokoro_tts/preprocessor.py`
- Test: `unit-tests/test_text_preprocessor.py`

Add `_apply_stress_markers()` to reduce lexical stress on tension words (`[word](-1)`) and boost affirmation words (`[word](+1)`) using misaki's stress annotation syntax. Insert as step 4b in `preprocess_for_meditation()`.

- [ ] **Step 1: Write the failing tests**

  Append to `unit-tests/test_text_preprocessor.py`:

  ```python
  from core.kokoro_tts.preprocessor import _apply_stress_markers


  class TestApplyStressMarkers(unittest.TestCase):

      def test_tension_word_reduced(self):
          result = _apply_stress_markers("Release the tension in your shoulders.")
          self.assertIn("[tension](-1)", result)

      def test_stress_word_reduced(self):
          result = _apply_stress_markers("Let go of all stress and worry.")
          self.assertIn("[stress](-1)", result)
          self.assertIn("[worry](-1)", result)

      def test_peace_word_boosted(self):
          result = _apply_stress_markers("You are filled with peace.")
          self.assertIn("[peace](+1)", result)

      def test_calm_word_boosted(self):
          result = _apply_stress_markers("Feel completely calm and still.")
          self.assertIn("[calm](+1)", result)
          self.assertIn("[still](+1)", result)

      def test_no_double_wrap_on_ipa_block(self):
          """Words already inside [word](/IPA/) blocks must not be wrapped again."""
          text = "[calm](/kˈɑːm/) is what you seek."
          result = _apply_stress_markers(text)
          # Should NOT produce [[calm]
          self.assertNotIn("[[calm]", result)

      def test_case_insensitive_matching(self):
          result = _apply_stress_markers("TENSION in your body.")
          self.assertIn("(-1)", result)

      def test_stress_markers_appear_in_full_preprocess_pipeline(self):
          """_apply_stress_markers is called inside preprocess_for_meditation."""
          text = "Release all tension. Find your peace."
          result = preprocess_for_meditation(text)
          self.assertIn("(-1)", result)  # tension → (-1)
          self.assertIn("(+1)", result)  # peace → (+1)
  ```

- [ ] **Step 2: Run the tests — verify they fail**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_text_preprocessor.py::TestApplyStressMarkers -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name '_apply_stress_markers'`

- [ ] **Step 3: Add constants and _apply_stress_markers to preprocessor.py**

  In `core/kokoro_tts/preprocessor.py`, after the `_SENSORY_WORDS` frozenset definition, add:

  ```python
  # ── Stress annotation targets ─────────────────────────────────────────────
  # Words where reducing lexical stress (-1) creates a calmer, less charged
  # delivery — the word sounds "already released" rather than emphasized.
  _STRESS_REDUCE_WORDS: list[str] = [
      "tension", "tense", "worry", "worries",
      "stress", "fear", "pain", "hurry", "thoughts", "distraction",
  ]

  # Words that serve as gentle affirmations — (+1) makes them land with
  # warmth and intentional calm rather than flat declarative delivery.
  _STRESS_BOOST_WORDS: list[str] = [
      "peace", "calm", "still", "stillness",
      "safe", "whole", "free", "light", "gently", "softly",
  ]


  def _apply_stress_markers(text: str) -> str:
      """Wrap targeted words with misaki stress annotation syntax.

      Uses misaki's built-in stress control:
        [word](-1) — reduces lexical stress by one level (softer, less charged)
        [word](+1) — boosts lexical stress by one level (gentle affirmation)

      The (?<!\\[) lookbehind prevents double-wrapping words already inside
      an IPA injection block [word](/IPA/) produced by inject_phonemes().

      Args:
          text: Text that has already been through inject_phonemes().

      Returns:
          Text with stress annotations inserted around targeted words.
      """
      for word in _STRESS_REDUCE_WORDS:
          text = re.sub(
              rf'(?<!\[)\b({word})\b',
              r'[\1](-1)',
              text,
              flags=re.IGNORECASE,
          )
      for word in _STRESS_BOOST_WORDS:
          text = re.sub(
              rf'(?<!\[)\b({word})\b',
              r'[\1](+1)',
              text,
              flags=re.IGNORECASE,
          )
      return text
  ```

  > **Note on the regex:** Each word is matched with `\b` word-boundary anchors so `still` doesn't match `stillness`, and vice versa (both are in the list separately). The `(?<!\[)` lookbehind ensures we don't touch words already preceded by `[` (inside an IPA block). The replacement `[\1](-1)` wraps the matched word: `tension` → `[tension](-1)`.

- [ ] **Step 4: Insert _apply_stress_markers into preprocess_for_meditation()**

  Find `preprocess_for_meditation()` in `preprocessor.py`. The current pipeline is:

  ```python
  def preprocess_for_meditation(text: str) -> str:
      text = expand_for_tts(text)
      text = _convert_to_contractions(text)
      text = inject_phonemes(text)
      text = enhance_prosody_punctuation(text)
      text = _inject_sensory_ellipses(text)
      text = _vary_sentence_lengths(text)
      text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)
      text = re.sub(r'\.{2,}', '...', text)
      text = re.sub(r'  +', ' ', text)
      return text.strip()
  ```

  Replace with:

  ```python
  def preprocess_for_meditation(text: str) -> str:
      """Optimise text for calm, meditation-style Kokoro TTS delivery.

      Pipeline:
      1. Expand digits/abbreviations to spoken forms.
      2. Convert formal phrasing to contractions for warmth.
      3. Inject IPA phonemes for Sanskrit/yoga terms.
      4. Apply stress reduction/boost markers (tension words softer,
         affirmation words gently emphasised).
      5. Insert commas at natural phrasing boundaries.
      6. Inject contemplative ellipses before sensory/somatic words.
      7. Vary sentence lengths for prosodic diversity.
      8. Add ellipsis transition before any inline [pause:] markers.
      9. Normalise ellipsis to exactly three dots.
      """
      text = expand_for_tts(text)
      text = _convert_to_contractions(text)
      text = inject_phonemes(text)
      text = _apply_stress_markers(text)       # step 4 — after IPA so collision guard works
      text = enhance_prosody_punctuation(text)
      text = _inject_sensory_ellipses(text)
      text = _vary_sentence_lengths(text)
      text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)
      text = re.sub(r'\.{2,}', '...', text)
      text = re.sub(r'  +', ' ', text)
      return text.strip()
  ```

- [ ] **Step 5: Run the tests — verify they pass**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_text_preprocessor.py::TestApplyStressMarkers -v
  ```

  Expected: `PASSED`

- [ ] **Step 6: Run full suite — verify no regressions**

  ```bash
  .venv/bin/python -m pytest unit-tests/ -v
  ```

  All previously-passing tests must still pass. Pay attention to `TestTextPreprocessor::test_preprocess_for_meditation` — the pipeline order change may interact with the existing ellipsis test.

- [ ] **Step 7: Commit**

  ```bash
  git add core/kokoro_tts/preprocessor.py unit-tests/test_text_preprocessor.py
  git commit -m "feat(kokoro): add stress reduction/boost markers to preprocessor

  Wraps tension-related words with misaki (-1) stress reduction and
  affirmation words with (+1) stress boost. Applied as step 4b in
  preprocess_for_meditation(), after IPA injection so the collision
  guard prevents double-wrapping Sanskrit IPA blocks.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Task 4: Add pure_calm preset and blend_with_extrapolation

**Files:**
- Modify: `core/kokoro_tts/voice_manager.py`
- Modify: `app.py`
- Test: `unit-tests/test_voice_manager.py`

Add `blend_with_extrapolation()` (supports negative weights, L2-renormalizes result) and the `pure_calm` preset. Update `get_voice()` to route `"method": "extrapolation"` presets to the new function.

- [ ] **Step 1: Write the failing tests**

  Append to `unit-tests/test_voice_manager.py`:

  ```python
  from core.kokoro_tts.voice_manager import blend_with_extrapolation


  class TestBlendWithExtrapolation(unittest.TestCase):

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      def test_returns_correct_shape(self, mock_load):
          mock_load.side_effect = lambda vid: _make_voice(
              {"af_heart": 0, "af_sarah": 1, "af_aoede": 2, "af_bella": 3}.get(vid, 0)
          )
          result = blend_with_extrapolation(
              {"af_heart": 0.60, "af_sarah": 0.30, "af_aoede": 0.10, "af_bella": -0.05}
          )
          self.assertEqual(result.shape, (511, 1, 256))

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      def test_no_nan_values(self, mock_load):
          mock_load.side_effect = lambda vid: _make_voice(
              {"af_heart": 0, "af_sarah": 1, "af_bella": 2}.get(vid, 0)
          )
          result = blend_with_extrapolation(
              {"af_heart": 0.60, "af_sarah": 0.30, "af_bella": -0.05}
          )
          self.assertFalse(
              torch.any(torch.isnan(result)).item(),
              "blend_with_extrapolation produced NaN values"
          )

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      def test_norm_renormalized_near_primary(self, mock_load):
          """After subtractive blending, result norm should stay near primary voice norm."""
          primary = _make_voice(0)

          def _side(vid):
              return primary if vid == "af_heart" else _make_voice(1)

          mock_load.side_effect = _side
          result = blend_with_extrapolation({"af_heart": 0.60, "af_bella": -0.05})
          primary_norm = float(torch.norm(primary.flatten().float()))
          result_norm = float(torch.norm(result.flatten().float()))
          self.assertAlmostEqual(
              result_norm, primary_norm, delta=primary_norm * 0.20,
              msg=f"Result norm {result_norm:.3f} too far from primary norm {primary_norm:.3f}"
          )

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      def test_pure_calm_preset_returns_tensor(self, mock_load):
          mock_load.side_effect = lambda vid: _make_voice(0)
          result = get_voice("pure_calm")
          self.assertIsInstance(result, torch.Tensor)
          self.assertEqual(result.shape, (511, 1, 256))

      @patch('core.kokoro_tts.voice_manager.load_voice_tensor')
      def test_pure_calm_preset_no_nan(self, mock_load):
          mock_load.side_effect = lambda vid: _make_voice(0)
          result = get_voice("pure_calm")
          self.assertFalse(torch.any(torch.isnan(result)).item())
  ```

- [ ] **Step 2: Run the tests — verify they fail**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_voice_manager.py::TestBlendWithExtrapolation -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'blend_with_extrapolation'`

- [ ] **Step 3: Add blend_with_extrapolation to voice_manager.py**

  In `core/kokoro_tts/voice_manager.py`, add after `slerp_blend()`:

  ```python
  def blend_with_extrapolation(voice_weights: dict[str, float]) -> torch.Tensor:
      """Blend voice tensors with support for negative (subtractive) weights.

      Negative weights subtract that voice's characteristics from the blend,
      useful for removing tension, energy, or specific tonal qualities that
      exist across multiple training voices.

      After blending, the result is L2-renormalized to the primary (first
      positive-weight) voice's norm, preventing amplitude drift caused by
      the subtraction operation.

      Args:
          voice_weights: e.g. {"af_heart": 0.6, "af_sarah": 0.3,
                               "af_aoede": 0.1, "af_bella": -0.05}
                         Positive weights add traits; negative weights subtract.
                         The first positive-weight voice is the primary voice
                         whose norm is used for renormalization.

      Returns:
          Blended voice tensor of the same shape as the input tensors.
      """
      result: torch.Tensor | None = None
      primary_norm: float | None = None

      for voice_id, weight in voice_weights.items():
          tensor = load_voice_tensor(voice_id).float()
          if result is None:
              result = tensor * weight
              # Capture norm of primary (first) voice for renormalization
              primary_norm = float(torch.norm(tensor.flatten()))
          else:
              result = result + tensor * weight

      if result is None:
          raise ValueError("voice_weights must contain at least one entry")

      # Renormalize to primary voice norm — prevents amplitude drift from subtraction
      if primary_norm and primary_norm > 1e-6:
          current_norm = float(torch.norm(result.flatten()))
          if current_norm > 1e-6:
              result = result * (primary_norm / current_norm)

      return result
  ```

- [ ] **Step 4: Add pure_calm preset to MEDITATION_PRESETS**

  In `core/kokoro_tts/voice_manager.py`, add `"pure_calm"` to `MEDITATION_PRESETS` after `"earth_root"`:

  ```python
      "earth_root": {
          "description": "Male/female grounding blend",
          "blend": {"af_heart": 0.7, "am_adam": 0.3},
      },
      "pure_calm": {
          "description": "Ultra-low tension — warmth with conversational energy subtracted",
          "blend": {
              "af_heart":  0.60,   # primary warmth and stability
              "af_sarah":  0.30,   # soft, natural breathiness
              "af_aoede":  0.10,   # musical prosody quality
              "af_bella": -0.05,   # subtract 5% of energetic/tension traits
          },
          "method": "extrapolation",
      },
  ```

- [ ] **Step 5: Update get_voice() to route extrapolation presets**

  Find `get_voice()` in `voice_manager.py`:

  ```python
  def get_voice(voice_spec: str):
      # Check if it is a preset
      if voice_spec in MEDITATION_PRESETS:
          return slerp_blend(MEDITATION_PRESETS[voice_spec]["blend"])
      ...
  ```

  Replace the preset branch with:

  ```python
  def get_voice(voice_spec: str):
      """Resolve a voice specification to a blended tensor or string ID.

      Accepts:
        - Single voice ID: "af_heart" → returns string
        - Comma-separated blend: "af_heart,af_nicole" → returns SLERP tensor
        - Preset name: "golden_hour" → returns blended tensor from preset
        - Extrapolation preset: "pure_calm" → uses blend_with_extrapolation()

      Returns:
          str | torch.Tensor
      """
      # Check if it is a preset
      if voice_spec in MEDITATION_PRESETS:
          preset = MEDITATION_PRESETS[voice_spec]
          if preset.get("method") == "extrapolation":
              return blend_with_extrapolation(preset["blend"])
          return slerp_blend(preset["blend"])

      # Check if it is a comma-separated blend
      if "," in voice_spec:
          voices = [v.strip() for v in voice_spec.split(",")]
          weight = 1.0 / len(voices)
          return slerp_blend({v: weight for v in voices})

      # Single voice ID — return as string (Kokoro handles it)
      return voice_spec
  ```

- [ ] **Step 6: Add pure_calm to KOKORO_VOICE_CHOICES in app.py**

  In `app.py`, find the `KOKORO_VOICE_CHOICES` list. Locate the `"earth_root"` entry and add `"pure_calm"` immediately after it:

  ```python
      ("Earth Root — grounding blend",               "earth_root"),
      ("Pure Calm — tension-free ultra-soft",        "pure_calm"),   # ← add this line
  ```

- [ ] **Step 7: Run the tests — verify they pass**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_voice_manager.py::TestBlendWithExtrapolation -v
  ```

  Expected: `PASSED`

- [ ] **Step 8: Run full suite — verify no regressions**

  ```bash
  .venv/bin/python -m pytest unit-tests/ -v
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add core/kokoro_tts/voice_manager.py app.py unit-tests/test_voice_manager.py
  git commit -m "feat(kokoro): add pure_calm preset with negative-weight extrapolation

  New blend_with_extrapolation() supports negative weights to subtract
  vocal characteristics. pure_calm preset: af_heart 0.6 + af_sarah 0.3
  + af_aoede 0.1 − af_bella 0.05. Result L2-renormalized to primary voice
  norm to prevent amplitude drift. Added to UI voice choices.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Task 5: Tighten FX chain — compressor -28 dB, reverb 18%

**Files:**
- Modify: `core/kokoro_tts/postprocessor.py`
- Test: `unit-tests/test_kokoro_postprocessor.py`

Two targeted parameter changes in `build_voice_chain()`: compressor threshold -22 → -28 dB (catches whisper-level meditation delivery), reverb wet default 0.15 → 0.18 (more enveloping acoustic space).

- [ ] **Step 1: Write the failing tests**

  Append to `unit-tests/test_kokoro_postprocessor.py`:

  ```python
  from core.kokoro_tts.postprocessor import build_voice_chain


  class TestBuildVoiceChainParameters(unittest.TestCase):
      """Verify the tuned FX chain parameters match the spec."""

      def test_compressor_threshold_is_minus_28(self):
          from pedalboard import Compressor
          chain = build_voice_chain()
          compressor = next((p for p in chain if isinstance(p, Compressor)), None)
          self.assertIsNotNone(compressor, "Expected a Compressor plugin in build_voice_chain()")
          self.assertAlmostEqual(
              compressor.threshold_db, -28.0, places=1,
              msg=f"Compressor threshold should be -28 dB, got {compressor.threshold_db}"
          )

      def test_compressor_ratio_unchanged(self):
          from pedalboard import Compressor
          chain = build_voice_chain()
          compressor = next((p for p in chain if isinstance(p, Compressor)), None)
          self.assertAlmostEqual(compressor.ratio, 2.5, places=1)

      def test_default_reverb_mix_is_0_18(self):
          from pedalboard import Convolution
          chain = build_voice_chain()
          convolution = next((p for p in chain if isinstance(p, Convolution)), None)
          self.assertIsNotNone(convolution, "Expected a Convolution plugin in build_voice_chain()")
          self.assertAlmostEqual(
              convolution.mix, 0.18, places=2,
              msg=f"Default reverb mix should be 0.18, got {convolution.mix}"
          )

      def test_custom_reverb_amount_respected(self):
          from pedalboard import Convolution
          chain = build_voice_chain(reverb_amount=0.10)
          convolution = next((p for p in chain if isinstance(p, Convolution)), None)
          self.assertAlmostEqual(convolution.mix, 0.10, places=2)
  ```

- [ ] **Step 2: Run the tests — verify they fail**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_kokoro_postprocessor.py::TestBuildVoiceChainParameters -v
  ```

  Expected: `FAILED` — `AssertionError: Compressor threshold should be -28 dB, got -22.0` and `reverb mix should be 0.18, got 0.15`

- [ ] **Step 3: Update build_voice_chain() in postprocessor.py**

  In `core/kokoro_tts/postprocessor.py`, find `build_voice_chain()`:

  Change the function signature default:
  ```python
  # Before:
  def build_voice_chain(reverb_amount: float = 0.15, ir_name: str = "warm_studio") -> Pedalboard:
  # After:
  def build_voice_chain(reverb_amount: float = 0.18, ir_name: str = "warm_studio") -> Pedalboard:
  ```

  Change the Compressor line inside the return statement:
  ```python
  # Before:
          Compressor(threshold_db=-22, ratio=2.5, attack_ms=15.0, release_ms=150.0),
  # After:
          Compressor(threshold_db=-28, ratio=2.5, attack_ms=15.0, release_ms=150.0),
  ```

  Update the docstring comment for the Compressor to reflect the new threshold:
  ```python
          # ── Dynamics ──
          # 2.5:1 @ -28 dB: catches whisper-level meditation delivery that sits
          # below the previous -22 dB threshold. Produces ~2-3 dB additional
          # gain reduction on soft phrases — transparent but consistent.
          Compressor(threshold_db=-28, ratio=2.5, attack_ms=15.0, release_ms=150.0),
  ```

- [ ] **Step 4: Run the tests — verify they pass**

  ```bash
  .venv/bin/python -m pytest unit-tests/test_kokoro_postprocessor.py::TestBuildVoiceChainParameters -v
  ```

  Expected: `PASSED`

- [ ] **Step 5: Run full suite — verify no regressions**

  ```bash
  .venv/bin/python -m pytest unit-tests/ -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add core/kokoro_tts/postprocessor.py unit-tests/test_kokoro_postprocessor.py
  git commit -m "feat(kokoro): tighten voice FX chain — compressor -28dB, reverb 18%

  Conservative tuning from research blueprint:
  - Compressor threshold: -22 → -28 dB (catches whisper-level delivery)
  - Reverb wet default: 15% → 18% (more enveloping acoustic space)
  Ratio, attack, release unchanged. Air shelf (+1dB@10kHz) unchanged.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Task 6: Reconcile all documentation

**Files:**
- Modify: `docs/model_implementation_guides/kokoro_tts.md`
- Modify: `docs/prompting_guides/vocal_kokoro_instructions.md`
- Modify: `CLAUDE.md`
- Modify: `docs/optimization_and_processing/post-processing-pipeline.md`

No code changes. Fix three known inconsistencies and update parameter tables to reflect Tasks 1–5.

- [ ] **Step 1: Fix kokoro_tts.md — humanize_voice status (§13)**

  In `docs/model_implementation_guides/kokoro_tts.md`, find section 13 "Expressiveness Enhancements":

  Find the subsection:
  ```
  ### Pitch Humanization & Formant Warmth (`postprocessor.py` — DISABLED)

  `humanize_voice(audio, sr)` exists in the codebase but is **not called** in the active pipeline. pyworld's WORLD vocoder resynthesis degrades Kokoro's neural vocoder output (ISTFTNet), causing both flatness and harshness. The function is retained for potential future use with a higher-quality resynthesis approach.
  ```

  Replace with:
  ```
  ### Pitch Humanization & Formant Warmth (`postprocessor.py` — ACTIVE, per-speech-chunk)

  `humanize_voice(audio, sr)` is called **per speech sentence** inside `KokoroEngine.synthesize()`, immediately after `apply_segment_fades()`. It is applied only to speech chunks — room-tone pause segments are excluded to avoid pyworld pitch-tracker artifacts at speech→silence boundaries.

  Parameters (all conservative — total modulation stays under ±15 cents):
  | Parameter | Value | Effect |
  |-----------|-------|--------|
  | `drift_hz` | 0.5 | Slow drift frequency (vocal fold tension simulation) |
  | `drift_cents` | 6.0 | Slow drift amplitude |
  | `vibrato_hz` | 5.0 | Subtle vibrato rate |
  | `vibrato_cents` | 3.0 | Vibrato amplitude |
  | `jitter_cents` | 2.0 | Random micro-jitter |
  | `formant_shift` | 0.97 | 3% lower formants (perceived warmth) |

  Academic basis: small pyworld perturbations (±6–15 cents) on neural vocoder output are validated for prosodic parameter manipulation without audible degradation (arXiv 2409.12176). The key constraint is keeping total modulation under ±15 cents.
  ```

- [ ] **Step 2: Fix kokoro_tts.md — FX chain values (§10)**

  In `docs/model_implementation_guides/kokoro_tts.md`, section 10 "Post-Processing Pipeline", Stage 3 FX chain table, update:

  Find:
  ```
  5. Compressor (2:1, -18 dB threshold, 10ms/100ms) — dynamics control
  ```
  Replace with:
  ```
  5. Compressor (2.5:1, **-28 dB** threshold, 15ms/150ms) — dynamics control; catches whisper-level delivery
  ```

  Find the reverb entry and update the wet percentage from `15%` to `18%`.

- [ ] **Step 3: Fix vocal_kokoro_instructions.md — paragraph pause duration**

  In `docs/prompting_guides/vocal_kokoro_instructions.md`, find:

  ```
  Any double newline (`\n\n`) in the script is automatically converted to a `[pause:2.5s]` marker before parsing.
  ```

  Replace with:

  ```
  Any double newline (`\n\n`) in the script is automatically converted to a `[pause:6.5s]` pause before parsing. The 6.5-second duration was chosen for spacious meditation pacing — it provides a generous breath between major script sections. Use explicit `[pause:Xs]` markers if you need a shorter or longer inter-paragraph pause.
  ```

  Also update the inline example if it shows "auto 2.5s pause":
  ```
  # Find:
  This is a new paragraph.                               ← blank line above → auto 2.5s pause
  # Replace with:
  This is a new paragraph.                               ← blank line above → auto 6.5s pause
  ```

- [ ] **Step 4: Fix CLAUDE.md — add Stage 2c to pipeline flow**

  In `CLAUDE.md`, find the pipeline flow block showing the Kokoro synthesis steps. Find the step ending `voice norm` and add Stage 2c between segment assembly and voice FX:

  ```
     # Find this region in the pipeline flow:
     7. voice FX         kokoro_tts/postprocessor :: build_voice_chain() + apply_fx()
    7b. voice norm       mixer.normalize_loudness() → −18 LUFS pre-mix

  # Add Stage 2c between Stage 6 (TTS upsample) and Stage 7 (voice FX):
     6. TTS upsample     audio_processor.upsample_audio() 24 kHz → 48 kHz
    6b. voice humanize   kokoro_tts/postprocessor :: humanize_voice() per speech chunk
                         (pitch drift ±6¢, vibrato ±3¢, jitter ±2¢, formant 0.97)
     7. voice FX         kokoro_tts/postprocessor :: build_voice_chain() + apply_fx()
  ```

- [ ] **Step 5: Fix CLAUDE.md — update FX Chain Summary table**

  In `CLAUDE.md`, FX Chain Summary table, find the `build_voice_chain()` row:

  ```
  | `build_voice_chain()` | Kokoro voice | NoiseGate(−40dB) → HPF(80Hz) → Peak(−2.5dB@400Hz) → LowShelf(+1.5dB@200Hz) → Compressor(2.5:1@−22dB,15ms/150ms) → HiShelf(+1dB@10kHz) → ConvReverb(warm_studio,15%wet) → Limiter(−1dBTP) |
  ```

  Replace with:

  ```
  | `build_voice_chain()` | Kokoro voice | NoiseGate(−40dB) → HPF(80Hz) → Peak(−2.5dB@400Hz) → LowShelf(+1.5dB@200Hz) → Compressor(2.5:1@**−28dB**,15ms/150ms) → HiShelf(+1dB@10kHz) → ConvReverb(warm_studio,**18%**wet) → Limiter(−1dBTP) |
  ```

- [ ] **Step 6: Fix CLAUDE.md — add humanize_voice constants to Key Constants table**

  In `CLAUDE.md`, Key Constants table, add after the `Kokoro crossfade` row:

  ```
  | Kokoro humanize drift | 6.0 cents (0.5 Hz) | `core/kokoro_tts/postprocessor.py` | Slow pitch drift; vocal fold tension simulation |
  | Kokoro humanize vibrato | 3.0 cents (5 Hz) | `core/kokoro_tts/postprocessor.py` | Subtle vibrato; ±15 cents total headroom |
  | Kokoro humanize jitter | 2.0 cents | `core/kokoro_tts/postprocessor.py` | Random micro-jitter |
  | Kokoro formant shift | 0.97 | `core/kokoro_tts/postprocessor.py` | 3% lower formants; perceived warmth |
  ```

- [ ] **Step 7: Fix post-processing-pipeline.md — add humanize stage and update values**

  In `docs/optimization_and_processing/post-processing-pipeline.md`, section 3 "Kokoro TTS Path" diagram, add the humanize stage:

  ```
  # After the spectral gating block, before "Upsample → mix sample rate":
  ┌──────────────────────────────────────┐
  │ Pitch humanization (per speech chunk)│
  │   • pyworld: drift ±6¢ @ 0.5 Hz     │
  │   • vibrato ±3¢ @ 5 Hz              │
  │   • jitter ±2¢ (Gaussian-smoothed)  │
  │   • formant shift 0.97 (3% lower)   │
  │   • skip: clips < 500ms, pauses     │
  └──────────────────────────────────────┘
  ```

  Then update the unified voice FX block parameters:
  - `Compressor 2:1 @ -18 dB` → `Compressor 2.5:1 @ -28 dB`
  - `ConvReverb 15% wet` → `ConvReverb 18% wet`

- [ ] **Step 8: Commit all documentation changes**

  ```bash
  git add docs/model_implementation_guides/kokoro_tts.md \
          docs/prompting_guides/vocal_kokoro_instructions.md \
          CLAUDE.md \
          docs/optimization_and_processing/post-processing-pipeline.md
  git commit -m "docs: reconcile Kokoro TTS documentation with implementation

  - kokoro_tts.md §13: humanize_voice is ACTIVE (per-chunk), not disabled
  - kokoro_tts.md §10: compressor -28dB, reverb 18% wet
  - vocal_kokoro_instructions.md: paragraph pause is 6.5s not 2.5s
  - CLAUDE.md: add Stage 2c, update FX table, add humanize constants
  - post-processing-pipeline.md: add humanize stage, update FX values

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  ```

---

## Final Verification

- [ ] **Run the complete test suite**

  ```bash
  cd /Users/rahul/Downloads/moodscape-guided-meditations
  .venv/bin/python -m pytest unit-tests/ -v
  ```

  Expected: All tests pass. No new failures.

- [ ] **Verify test counts increased**

  ```bash
  .venv/bin/python -m pytest unit-tests/ --collect-only -q 2>/dev/null | tail -5
  ```

  The new tests from Tasks 1–5 should appear in the collected count.

- [ ] **Spot-check the stress marker output on a sample script**

  ```bash
  cd /Users/rahul/Downloads/moodscape-guided-meditations
  .venv/bin/python -c "
  from core.kokoro_tts.preprocessor import preprocess_for_meditation
  text = 'Release all tension in your body. You are filled with peace and calm.'
  print(preprocess_for_meditation(text))
  "
  ```

  Expected output contains `[tension](-1)`, `[peace](+1)`, `[calm](+1)`.

- [ ] **Spot-check pure_calm preset resolves without errors**

  ```bash
  .venv/bin/python -c "
  from unittest.mock import patch, MagicMock
  import torch
  # Mock HF download to avoid network call
  fake_tensor = torch.randn(511, 1, 256)
  with patch('core.kokoro_tts.voice_manager.hf_hub_download', return_value='/dev/null'), \
       patch('torch.load', return_value=fake_tensor):
      from core.kokoro_tts.voice_manager import get_voice
      result = get_voice('pure_calm')
      print(f'pure_calm shape: {result.shape}, NaN: {torch.any(torch.isnan(result)).item()}')
  "
  ```

  Expected: `pure_calm shape: torch.Size([511, 1, 256]), NaN: False`
