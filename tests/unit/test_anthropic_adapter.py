"""Tests for the Anthropic adapter, using a stub client (no network)."""

import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core.script_gen.adapters.anthropic_api import AnthropicEngine


@contextmanager
def _env_without(*names):
    """Temporarily unset env vars without clearing the rest of os.environ."""
    saved = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


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

    # -- Malformed response shapes ---------------------------------------
    #
    # The text-extraction expression used to sit outside the try/except
    # wrapping the stream call, so a malformed response escaped as a bare
    # TypeError/AttributeError instead of the RuntimeError every caller
    # (generator.py's orchestrator) catches. These pin all three observed
    # escapes now raising RuntimeError, chained from the original exception.

    def test_message_content_none_raises_runtime_error(self):
        # Iterating `for block in message.content` over None raises
        # TypeError -- must not escape as a bare TypeError.
        client = StubClient(None)
        engine = AnthropicEngine("claude-opus-5", client=client)
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIsInstance(ctx.exception.__cause__, TypeError)

    def test_message_none_raises_runtime_error(self):
        # get_final_message() returning None makes `message.content` raise
        # AttributeError -- must not escape as a bare AttributeError.
        class NoneMessageStream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return None

        class NoneMessageMessages:
            def stream(self, **kwargs):
                return NoneMessageStream()

        class NoneMessageClient:
            def __init__(self):
                self.messages = NoneMessageMessages()

        engine = AnthropicEngine("claude-opus-5", client=NoneMessageClient())
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIsInstance(ctx.exception.__cause__, AttributeError)

    def test_text_block_missing_text_attribute_raises_runtime_error(self):
        # A block reporting type="text" but with no .text attribute makes
        # `block.text` raise AttributeError -- must not escape as a bare
        # AttributeError.
        class TextTypeNoTextAttr:
            type = "text"

        client = StubClient([TextTypeNoTextAttr()])
        engine = AnthropicEngine("claude-opus-5", client=client)
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIsInstance(ctx.exception.__cause__, AttributeError)

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

    # -- Retry configuration --------------------------------------------
    #
    # The anthropic SDK already retries connection errors, 408/409/429 and
    # 5xx internally with its own exponential backoff (see
    # adapters/anthropic_api.py's module docstring). We never hand-roll a
    # second retry loop around it -- these tests only verify the client is
    # constructed with an explicit, intentional `max_retries`, not that a
    # loop here retries anything.

    def test_client_is_constructed_with_an_explicit_max_retries(self):
        captured = {}

        class FakeAnthropicModule:
            class Anthropic:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.messages = StubMessages(
                        StubMessage([StubBlock("out")]), []
                    )

        with patch.dict(
            os.environ, {"MOODSCAPE_TEST_KEY_RETRIES": "sk-test"}
        ), patch.dict(sys.modules, {"anthropic": FakeAnthropicModule}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_RETRIES"
            )
            engine.complete("sys", "usr")

        self.assertIn("max_retries", captured)
        self.assertIsInstance(captured["max_retries"], int)

    def test_max_retries_env_var_is_respected(self):
        captured = {}

        class FakeAnthropicModule:
            class Anthropic:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.messages = StubMessages(
                        StubMessage([StubBlock("out")]), []
                    )

        with patch.dict(
            os.environ,
            {
                "MOODSCAPE_TEST_KEY_RETRIES_ENV": "sk-test",
                "MOODSCAPE_SCRIPT_MAX_RETRIES": "5",
            },
        ), patch.dict(sys.modules, {"anthropic": FakeAnthropicModule}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_RETRIES_ENV"
            )
            engine.complete("sys", "usr")

        # 5 total attempts == 1 initial + 4 SDK-owned retries.
        self.assertEqual(captured["max_retries"], 4)

    def test_default_max_retries_is_two_when_env_var_unset(self):
        captured = {}

        class FakeAnthropicModule:
            class Anthropic:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.messages = StubMessages(
                        StubMessage([StubBlock("out")]), []
                    )

        with _env_without("MOODSCAPE_SCRIPT_MAX_RETRIES"), patch.dict(
            os.environ, {"MOODSCAPE_TEST_KEY_RETRIES_DEFAULT": "sk-test"}
        ), patch.dict(sys.modules, {"anthropic": FakeAnthropicModule}):
            engine = AnthropicEngine(
                "claude-opus-5", api_key_env="MOODSCAPE_TEST_KEY_RETRIES_DEFAULT"
            )
            engine.complete("sys", "usr")

        # Default DEFAULT_MAX_RETRIES = 3 total attempts == 2 SDK retries.
        self.assertEqual(captured["max_retries"], 2)


if __name__ == "__main__":
    unittest.main()
