"""Tests for the ScriptEngine interface and provider registry.

No network calls: build_engine is exercised for construction and error paths
only, and behaviour is tested through FakeScriptEngine.
"""

import unittest

from core.script_gen.engine import (
    PROVIDER_BASE_URLS,
    PROVIDER_KEY_ENV,
    FakeScriptEngine,
    ScriptEngine,
    build_engine,
    parse_engine_spec,
)


class TestSpecParsing(unittest.TestCase):
    def test_splits_provider_from_model(self):
        self.assertEqual(parse_engine_spec("ollama:llama3.2"), ("ollama", "llama3.2"))

    def test_splits_on_first_colon_only(self):
        # Ollama tags contain colons; they belong to the model name.
        self.assertEqual(
            parse_engine_spec("ollama:qwen3:30b"), ("ollama", "qwen3:30b")
        )

    def test_slashes_in_model_names_survive(self):
        self.assertEqual(
            parse_engine_spec("openrouter:meta-llama/llama-3.3-70b"),
            ("openrouter", "meta-llama/llama-3.3-70b"),
        )

    def test_missing_colon_raises(self):
        with self.assertRaises(ValueError):
            parse_engine_spec("ollama")

    def test_empty_model_raises(self):
        with self.assertRaises(ValueError):
            parse_engine_spec("ollama:")

    def test_unknown_provider_raises_listing_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            build_engine("nosuchprovider:model")
        self.assertIn("ollama", str(ctx.exception))


class TestRegistry(unittest.TestCase):
    def test_openai_compatible_providers_registered(self):
        for provider in ("ollama", "openrouter", "together", "fireworks", "groq"):
            self.assertIn(provider, PROVIDER_BASE_URLS)

    def test_ollama_points_at_localhost(self):
        self.assertIn("localhost", PROVIDER_BASE_URLS["ollama"])

    def test_every_base_url_ends_with_v1(self):
        for url in PROVIDER_BASE_URLS.values():
            self.assertTrue(url.endswith("/v1"), url)

    def test_build_engine_forwards_the_anthropic_key_env(self):
        # PROVIDER_KEY_ENV["anthropic"] must actually reach AnthropicEngine —
        # otherwise the table is decorative and changing that entry would
        # silently do nothing (it previously coincided with the adapter's
        # own hardcoded default, masking the bug).
        engine = build_engine("anthropic:claude-opus-5")
        self.assertEqual(engine._api_key_env, PROVIDER_KEY_ENV["anthropic"])


class TestFakeEngine(unittest.TestCase):
    def test_is_a_script_engine(self):
        self.assertIsInstance(FakeScriptEngine(["out"]), ScriptEngine)

    def test_returns_queued_responses_in_order(self):
        engine = FakeScriptEngine(["first", "second"])
        self.assertEqual(engine.complete("sys", "usr"), "first")
        self.assertEqual(engine.complete("sys", "usr"), "second")

    def test_repeats_the_last_response_when_exhausted(self):
        engine = FakeScriptEngine(["only"])
        engine.complete("sys", "usr")
        self.assertEqual(engine.complete("sys", "usr"), "only")

    def test_records_calls_for_assertions(self):
        engine = FakeScriptEngine(["out"])
        engine.complete("SYSTEM", "USER")
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")
        self.assertEqual(engine.calls[0]["user"], "USER")

    def test_has_a_name(self):
        self.assertTrue(FakeScriptEngine(["out"]).name)

    def test_empty_response_list_raises(self):
        with self.assertRaises(ValueError):
            FakeScriptEngine([])


class UnloadAndPreflightTest(unittest.TestCase):
    def test_fake_engine_counts_unloads(self):
        engine = FakeScriptEngine(["x"])
        self.assertEqual(engine.unload_calls, 0)
        engine.unload()
        engine.unload()
        self.assertEqual(engine.unload_calls, 2)

    def test_fake_engine_counts_preflights(self):
        engine = FakeScriptEngine(["x"])
        engine.preflight()
        self.assertEqual(engine.preflight_calls, 1)

    def test_anthropic_engine_unload_is_a_harmless_no_op(self):
        """Hosted providers have nothing to unload; the call must not fail."""
        from core.script_gen.adapters.anthropic_api import AnthropicEngine

        engine = AnthropicEngine("claude-opus-5", api_key_env="ANTHROPIC_API_KEY")
        engine.unload()
        engine.preflight()


if __name__ == "__main__":
    unittest.main()
