"""Tests for the OpenAI-compatible adapter (Ollama + hosted open-weight providers).

Uses httpx.MockTransport so the real request-building and response-parsing
paths run with no network, and injects the adapter's `sleep` callable so
retry-with-backoff tests run with no real sleeping.
"""

import json
import os
import unittest
from unittest.mock import patch

import httpx

from core.script_gen.adapters.openai_compat import (
    BACKOFF_CAP_SEC,
    DEFAULT_MAX_RETRIES,
    OpenAICompatEngine,
)


def ok_transport(captured: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "GENERATED SCRIPT"}}]},
        )

    return httpx.MockTransport(handler)


def status_transport(
    code: int, body: str = "boom", captured: list | None = None
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(code, text=body)

    return httpx.MockTransport(handler)


def ok_step(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "GENERATED SCRIPT"}}]},
    )


def status_step(code: int, body: str = "boom", headers: dict | None = None):
    def _step(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, text=body, headers=headers or {})

    return _step


def connect_error_step(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def timeout_step(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def sequence_transport(steps: list, captured: list | None = None) -> httpx.MockTransport:
    """Returns responses from `steps` in order; the last step repeats once
    exhausted, so a transport that should "always fail" is just `[step]`."""
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        idx = min(state["n"], len(steps) - 1)
        state["n"] += 1
        return steps[idx](request)

    return httpx.MockTransport(handler)


class TestOpenAICompatEngine(unittest.TestCase):
    def build(self, transport, **kwargs):
        kwargs.setdefault("sleep", lambda seconds: None)
        return OpenAICompatEngine(
            provider="ollama",
            model="qwen3:30b",
            base_url="http://localhost:11434/v1",
            transport=transport,
            **kwargs,
        )

    def test_returns_the_message_content(self):
        engine = self.build(ok_transport([]))
        self.assertEqual(engine.complete("sys", "usr"), "GENERATED SCRIPT")

    def test_posts_to_chat_completions(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertEqual(
            str(captured[0].url), "http://localhost:11434/v1/chat/completions"
        )

    def test_sends_system_and_user_messages(self):
        captured = []
        self.build(ok_transport(captured)).complete("SYSTEM", "USER")
        payload = json.loads(captured[0].content)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "USER"})

    def test_sends_the_model_name(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertEqual(json.loads(captured[0].content)["model"], "qwen3:30b")

    def test_forwards_max_tokens_and_temperature(self):
        captured = []
        self.build(ok_transport(captured)).complete(
            "sys", "usr", max_tokens=1234, temperature=0.4
        )
        payload = json.loads(captured[0].content)
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["temperature"], 0.4)

    def test_no_auth_header_without_a_key_env(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertNotIn("authorization", captured[0].headers)

    def test_name_is_provider_and_model(self):
        self.assertEqual(self.build(ok_transport([])).name, "ollama:qwen3:30b")

    def test_connection_error_names_the_provider_and_the_fix(self):
        # Always fails: exhausts retries and raises RuntimeError. No real
        # sleeping (self.build injects a no-op sleep by default).
        engine = self.build(sequence_transport([connect_error_step]))
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        message = str(ctx.exception)
        self.assertIn("ollama", message.lower())
        self.assertIn("11434", message)

    def test_http_error_is_wrapped_with_the_status(self):
        engine = self.build(status_transport(500))
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIn("500", str(ctx.exception))

    def test_missing_api_key_raises_naming_the_env_var(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenAICompatEngine(
                provider="openrouter",
                model="some/model",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="MOODSCAPE_TEST_ABSENT_KEY",
                transport=ok_transport([]),
            ).complete("sys", "usr")
        self.assertIn("MOODSCAPE_TEST_ABSENT_KEY", str(ctx.exception))

    def test_malformed_response_raises_and_is_not_retried(self):
        captured = []
        transport = sequence_transport(
            [lambda request: httpx.Response(200, json={"unexpected": True})],
            captured,
        )
        with self.assertRaises(RuntimeError):
            self.build(transport).complete("sys", "usr")
        # The request succeeded; the content is wrong. Retrying would just
        # reproduce the same malformed body, so exactly one request is made.
        self.assertEqual(len(captured), 1)

    def test_null_content_raises_runtime_error(self):
        # {"content": null} in a 200 body is standard for reasoning models
        # on providers like OpenRouter/Groq. Left unguarded this returns
        # None, and generator.py later does a string op on it, raising a
        # bare TypeError that escapes every RuntimeError handler in the
        # auto-generation pipeline.
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": None}}]}
            )
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.build(transport).complete("sys", "usr")
        self.assertIn("ollama", str(ctx.exception).lower())

    def test_empty_string_content_raises_runtime_error(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}]}
            )
        )
        with self.assertRaises(RuntimeError):
            self.build(transport).complete("sys", "usr")

    def test_non_json_response_raises_runtime_error(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="not json{{")
        )
        with self.assertRaises(RuntimeError):
            self.build(transport).complete("sys", "usr")

    def test_read_error_raises_runtime_error(self):
        def reset(request):
            raise httpx.ReadError("connection reset", request=request)

        engine = self.build(sequence_transport([reset]))
        with self.assertRaises(RuntimeError):
            engine.complete("sys", "usr")

    def test_authorization_header_sent_when_key_present(self):
        captured = []
        with patch.dict(os.environ, {"MOODSCAPE_TEST_PRESENT_KEY": "secret123"}):
            OpenAICompatEngine(
                provider="openrouter",
                model="some/model",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="MOODSCAPE_TEST_PRESENT_KEY",
                transport=ok_transport(captured),
            ).complete("sys", "usr")
        self.assertEqual(captured[0].headers["authorization"], "Bearer secret123")

    # -- Retry-with-backoff -------------------------------------------------

    def test_500_then_200_succeeds_after_one_retry(self):
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(500), ok_step], captured)
        engine = self.build(transport, sleep=sleeps.append)
        result = engine.complete("sys", "usr")
        self.assertEqual(result, "GENERATED SCRIPT")
        self.assertEqual(len(captured), 2)
        self.assertEqual(len(sleeps), 1)

    def test_500_every_time_fails_after_exactly_configured_attempts(self):
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(500)], captured)
        engine = self.build(transport, sleep=sleeps.append)
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertEqual(len(captured), DEFAULT_MAX_RETRIES)
        self.assertEqual(len(sleeps), DEFAULT_MAX_RETRIES - 1)
        self.assertIn(f"{DEFAULT_MAX_RETRIES} attempt", str(ctx.exception))

    def test_400_is_never_retried(self):
        captured = []
        engine = self.build(status_transport(400, captured=captured))
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        # This is the regression guard: a bad model name or malformed
        # request fails identically every time, so retrying would only
        # triple the latency of a failure that was never going to succeed.
        self.assertEqual(len(captured), 1)
        self.assertIn("400", str(ctx.exception))

    def test_408_then_200_succeeds_after_one_retry(self):
        # Request Timeout is transient by definition -- and the anthropic
        # SDK already retries it, so this is the fix that makes the two
        # adapters' documented retry guarantee actually true.
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(408), ok_step], captured)
        engine = self.build(transport, sleep=sleeps.append)
        result = engine.complete("sys", "usr")
        self.assertEqual(result, "GENERATED SCRIPT")
        self.assertEqual(len(captured), 2)
        self.assertEqual(len(sleeps), 1)

    def test_408_every_time_fails_after_exactly_configured_attempts(self):
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(408)], captured)
        engine = self.build(transport, sleep=sleeps.append)
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertEqual(len(captured), DEFAULT_MAX_RETRIES)
        self.assertIn(f"{DEFAULT_MAX_RETRIES} attempt", str(ctx.exception))

    def test_409_then_200_succeeds_after_one_retry(self):
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(409), ok_step], captured)
        engine = self.build(transport, sleep=sleeps.append)
        result = engine.complete("sys", "usr")
        self.assertEqual(result, "GENERATED SCRIPT")
        self.assertEqual(len(captured), 2)
        self.assertEqual(len(sleeps), 1)

    def test_409_every_time_fails_after_exactly_configured_attempts(self):
        captured = []
        sleeps = []
        transport = sequence_transport([status_step(409)], captured)
        engine = self.build(transport, sleep=sleeps.append)
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertEqual(len(captured), DEFAULT_MAX_RETRIES)
        self.assertIn(f"{DEFAULT_MAX_RETRIES} attempt", str(ctx.exception))

    def test_429_retry_after_header_is_honoured_but_clamped_to_cap(self):
        transport = sequence_transport(
            [status_step(429, "slow down", {"Retry-After": "9999"}), ok_step]
        )
        sleeps = []
        engine = self.build(transport, sleep=sleeps.append)
        result = engine.complete("sys", "usr")
        self.assertEqual(result, "GENERATED SCRIPT")
        self.assertEqual(sleeps, [BACKOFF_CAP_SEC])

    def test_connection_error_then_success_succeeds(self):
        captured = []
        transport = sequence_transport([connect_error_step, ok_step], captured)
        engine = self.build(transport, sleep=lambda seconds: None)
        self.assertEqual(engine.complete("sys", "usr"), "GENERATED SCRIPT")
        self.assertEqual(len(captured), 2)

    def test_timeout_then_success_succeeds(self):
        captured = []
        transport = sequence_transport([timeout_step, ok_step], captured)
        engine = self.build(transport, sleep=lambda seconds: None)
        self.assertEqual(engine.complete("sys", "usr"), "GENERATED SCRIPT")
        self.assertEqual(len(captured), 2)

    def test_backoff_grows_between_attempts(self):
        sleeps = []
        transport = sequence_transport([status_step(500)])
        engine = self.build(transport, sleep=sleeps.append)
        with self.assertRaises(RuntimeError):
            engine.complete("sys", "usr")
        self.assertEqual(len(sleeps), DEFAULT_MAX_RETRIES - 1)
        for earlier, later in zip(sleeps, sleeps[1:]):
            self.assertLess(earlier, later)

    def test_max_retries_env_var_overrides_default(self):
        captured = []
        transport = sequence_transport([status_step(500)], captured)
        with patch.dict(os.environ, {"MOODSCAPE_SCRIPT_MAX_RETRIES": "1"}):
            engine = self.build(transport, sleep=lambda seconds: None)
            with self.assertRaises(RuntimeError) as ctx:
                engine.complete("sys", "usr")
        self.assertEqual(len(captured), 1)
        self.assertIn("1 attempt", str(ctx.exception))


class OllamaUnloadTest(unittest.TestCase):
    def _engine(self, handler, provider="ollama"):
        return OpenAICompatEngine(
            provider=provider,
            model="qwen3.8:27b",
            base_url="http://localhost:11434/v1",
            api_key_env=None,
            transport=httpx.MockTransport(handler),
        )

    def test_unload_posts_keep_alive_zero_to_the_native_endpoint(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok"})

        self._engine(handler).unload()
        self.assertEqual(seen["url"], "http://localhost:11434/api/generate")
        self.assertEqual(seen["body"]["keep_alive"], 0)
        self.assertEqual(seen["body"]["model"], "qwen3.8:27b")

    def test_unload_is_a_no_op_for_hosted_providers(self):
        def handler(request):
            raise AssertionError("hosted providers must not be called on unload")

        self._engine(handler, provider="groq").unload()

    def test_a_failing_unload_never_raises(self):
        """A model that will not unload is a memory problem, not a job failure."""

        def handler(request):
            return httpx.Response(500, text="boom")

        self._engine(handler).unload()

    def test_preflight_passes_when_the_model_is_present(self):
        def handler(request):
            return httpx.Response(
                200, json={"models": [{"name": "qwen3.8:27b"}, {"name": "gemma4:31b"}]}
            )

        self._engine(handler).preflight()

    def test_preflight_names_the_pull_command_when_the_model_is_missing(self):
        def handler(request):
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

        with self.assertRaises(RuntimeError) as ctx:
            self._engine(handler).preflight()
        self.assertIn("ollama pull qwen3.8:27b", str(ctx.exception))

    def test_preflight_is_silent_when_ollama_is_unreachable(self):
        """Preflight is an early warning, not a second connectivity check.

        complete() already reports an unreachable Ollama with a good message;
        failing here too would just replace it with a worse one.
        """

        def handler(request):
            raise httpx.ConnectError("refused")

        self._engine(handler).preflight()


if __name__ == "__main__":
    unittest.main()
