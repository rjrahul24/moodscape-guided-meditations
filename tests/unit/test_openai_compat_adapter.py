"""Tests for the OpenAI-compatible adapter (Ollama + hosted open-weight providers).

Uses httpx.MockTransport so the real request-building and response-parsing
paths run with no network.
"""

import json
import os
import unittest
from unittest.mock import patch

import httpx

from core.script_gen.adapters.openai_compat import OpenAICompatEngine


def ok_transport(captured: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "GENERATED SCRIPT"}}]},
        )

    return httpx.MockTransport(handler)


def status_transport(code: int, body: str = "boom") -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(code, text=body))


class TestOpenAICompatEngine(unittest.TestCase):
    def build(self, transport, **kwargs):
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
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)

        engine = self.build(httpx.MockTransport(refuse))
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

    def test_malformed_response_raises(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"unexpected": True})
        )
        with self.assertRaises(RuntimeError):
            self.build(transport).complete("sys", "usr")

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

        engine = self.build(httpx.MockTransport(reset))
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


if __name__ == "__main__":
    unittest.main()
