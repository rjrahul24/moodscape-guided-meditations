import json
import os
import unittest
from unittest.mock import patch

import httpx

from core.script_gen.adapters.openai_compat import OpenAICompatEngine
from core.script_gen.engine import FakeScriptEngine
from core.script_gen.sampling import SamplingParams, sampling_for


def _clear_env():
    return patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                   if not k.startswith("MOODSCAPE_SAMPLING_")}, clear=True)


class TestSamplingFor(unittest.TestCase):
    def test_qwen_writer_uses_card_non_thinking_values(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 1.5))
            self.assertEqual(sampling_for("planner", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 1.5))

    def test_qwen_as_judge_has_no_presence_penalty(self):
        with _clear_env():
            self.assertEqual(sampling_for("judge", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 0.0))
            self.assertEqual(sampling_for("repair", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 0.0))

    def test_gemma_any_stage(self):
        with _clear_env():
            for stage in ("planner", "writer", "judge", "repair"):
                self.assertEqual(sampling_for(stage, "ollama:gemma4:31b"), SamplingParams(1.0, 0.95, None))

    def test_unknown_model_keeps_todays_behaviour(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "ollama:llama3.2:3b"), SamplingParams(1.0, None, None))

    def test_family_from_path_style_name(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "openrouter:qwen/qwen3-32b"), SamplingParams(0.7, 0.80, 1.5))
            self.assertEqual(sampling_for("writer", "weird"), SamplingParams(1.0, None, None))

    def test_env_override_and_none(self):
        env = {"MOODSCAPE_SAMPLING_WRITER_PRESENCE_PENALTY": "0.5",
               "MOODSCAPE_SAMPLING_WRITER_TOP_P": "none"}
        with _clear_env(), patch.dict(os.environ, env):
            self.assertEqual(sampling_for("writer", "ollama:qwen3.8:27b"), SamplingParams(0.7, None, 0.5))

    def test_bad_env_names_variable(self):
        with _clear_env(), patch.dict(os.environ, {"MOODSCAPE_SAMPLING_JUDGE_TEMPERATURE": "hot"}):
            with self.assertRaisesRegex(ValueError, "MOODSCAPE_SAMPLING_JUDGE_TEMPERATURE"):
                sampling_for("judge", "ollama:gemma4:31b")

    def test_temperature_none_is_rejected(self):
        var = "MOODSCAPE_SAMPLING_JUDGE_TEMPERATURE"
        for raw in ("none", "null", ""):
            with _clear_env(), patch.dict(os.environ, {var: raw}):
                with self.assertRaisesRegex(ValueError, var + ".*cannot be unset"):
                    sampling_for("judge", "ollama:gemma4:31b")

    def test_non_finite_values_rejected(self):
        for field in ("TEMPERATURE", "TOP_P", "PRESENCE_PENALTY"):
            for raw in ("nan", "inf", "-inf"):
                var = f"MOODSCAPE_SAMPLING_WRITER_{field}"
                with _clear_env(), patch.dict(os.environ, {var: raw}):
                    with self.assertRaisesRegex(ValueError, var):
                        sampling_for("writer", "ollama:gemma4:31b")

    def test_unknown_stage(self):
        with self.assertRaisesRegex(ValueError, "stage"):
            sampling_for("polish", "ollama:gemma4:31b")

    def test_as_dict(self):
        self.assertEqual(SamplingParams(0.7, 0.8, None).as_dict(),
                         {"temperature": 0.7, "top_p": 0.8, "presence_penalty": None})


class TestAdapters(unittest.TestCase):
    def _engine(self, captured):
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return OpenAICompatEngine("ollama", "qwen3.8:27b", "http://x/v1", transport=httpx.MockTransport(handler))

    def test_openai_compat_sends_sampling_fields(self):
        captured = []
        self._engine(captured).complete("s", "u", sampling=SamplingParams(0.7, 0.8, 1.5))
        body = captured[0]
        self.assertEqual((body["temperature"], body["top_p"], body["presence_penalty"]), (0.7, 0.8, 1.5))

    def test_openai_compat_omits_none_fields(self):
        captured = []
        self._engine(captured).complete("s", "u", sampling=SamplingParams(1.0))
        self.assertNotIn("top_p", captured[0])
        self.assertNotIn("presence_penalty", captured[0])

    def test_openai_compat_without_sampling_unchanged(self):
        captured = []
        self._engine(captured).complete("s", "u", temperature=1.0)
        self.assertEqual(captured[0]["temperature"], 1.0)
        self.assertNotIn("top_p", captured[0])

    def test_fake_engine_records_sampling(self):
        fake = FakeScriptEngine(["x"])
        fake.complete("s", "u", sampling=SamplingParams(0.7))
        self.assertEqual(fake.calls[0]["sampling"], SamplingParams(0.7))


if __name__ == "__main__":
    unittest.main()
