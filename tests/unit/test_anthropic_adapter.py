"""Tests for the Anthropic adapter, using a stub client (no network)."""

import os
import sys
import unittest
from unittest.mock import patch

from core.script_gen.adapters.anthropic_api import AnthropicEngine


class StubBlock:
    def __init__(self, text, type_="text"):
        self.text = text
        self.type = type_


class StubMessage:
    def __init__(self, blocks):
        self.content = blocks


class StubStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class StubMessages:
    def __init__(self, message, recorder):
        self._message = message
        self._recorder = recorder

    def stream(self, **kwargs):
        self._recorder.append(kwargs)
        return StubStream(self._message)


class StubClient:
    def __init__(self, blocks):
        self.calls: list[dict] = []
        self.messages = StubMessages(StubMessage(blocks), self.calls)


class TestAnthropicEngine(unittest.TestCase):
    def test_returns_concatenated_text_blocks(self):
        client = StubClient([StubBlock("Hello "), StubBlock("world")])
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.complete("sys", "usr"), "Hello world")

    def test_ignores_thinking_blocks(self):
        client = StubClient(
            [StubBlock("reasoning", type_="thinking"), StubBlock("answer")]
        )
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.complete("sys", "usr"), "answer")

    def test_passes_system_as_top_level_param(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("SYSTEM", "USER")
        self.assertEqual(client.calls[0]["system"], "SYSTEM")

    def test_passes_user_message(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("SYSTEM", "USER")
        self.assertEqual(
            client.calls[0]["messages"], [{"role": "user", "content": "USER"}]
        )

    def test_does_not_send_budget_tokens(self):
        # budget_tokens is rejected with a 400 on current models.
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("s", "u")
        thinking = client.calls[0].get("thinking", {})
        self.assertNotIn("budget_tokens", thinking)

    def test_forwards_max_tokens(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete(
            "s", "u", max_tokens=9000
        )
        self.assertEqual(client.calls[0]["max_tokens"], 9000)

    def test_name_is_prefixed(self):
        client = StubClient([StubBlock("out")])
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.name, "anthropic:claude-opus-5")

    def test_missing_key_raises_naming_the_env_var(self):
        engine = AnthropicEngine(
            "claude-opus-5", api_key_env="MOODSCAPE_TEST_ABSENT_KEY"
        )
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIn("MOODSCAPE_TEST_ABSENT_KEY", str(ctx.exception))

    def test_empty_response_raises(self):
        engine = AnthropicEngine("claude-opus-5", client=StubClient([]))
        with self.assertRaises(RuntimeError):
            engine.complete("sys", "usr")

    def test_package_not_installed_raises(self):
        # Forcing sys.modules["anthropic"] = None makes `import anthropic`
        # raise the real ImportError, without needing the package absent.
        with patch.dict(
            os.environ, {"MOODSCAPE_TEST_KEY_NOPKG": "sk-test"}
        ), patch.dict(sys.modules, {"anthropic": None}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_NOPKG"
            )
            with self.assertRaises(RuntimeError) as ctx:
                engine.complete("sys", "usr")
        self.assertIn("not installed", str(ctx.exception))

    def test_client_construction_failure_raises_runtime_error(self):
        class FakeAnthropicModule:
            class Anthropic:
                def __init__(self, **kwargs):
                    raise TypeError("bad proxy configuration")

        with patch.dict(
            os.environ, {"MOODSCAPE_TEST_KEY_CTORFAIL": "sk-test"}
        ), patch.dict(sys.modules, {"anthropic": FakeAnthropicModule}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_CTORFAIL"
            )
            with self.assertRaises(RuntimeError):
                engine.complete("sys", "usr")

    def test_custom_api_key_env_value_is_forwarded_to_client(self):
        captured = {}

        class FakeAnthropicModule:
            class Anthropic:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.messages = StubMessages(
                        StubMessage([StubBlock("out")]), []
                    )

        with patch.dict(
            os.environ, {"MOODSCAPE_TEST_KEY_CUSTOM": "sk-custom-value"}
        ), patch.dict(sys.modules, {"anthropic": FakeAnthropicModule}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_CUSTOM"
            )
            engine.complete("sys", "usr")

        self.assertEqual(captured.get("api_key"), "sk-custom-value")


if __name__ == "__main__":
    unittest.main()
