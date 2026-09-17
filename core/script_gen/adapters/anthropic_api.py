"""Adapter for the Anthropic Messages API.

Separate from openai_compat because the wire protocol differs. Uses the
official anthropic SDK.

Current-model notes: adaptive thinking is the default and budget_tokens is
rejected with a 400, so it is never sent. Streaming is used because large
max_tokens values risk an HTTP timeout on a non-streaming request.

Retry note: unlike openai_compat.py (raw httpx, no built-in retry -- see
that module for the hand-rolled loop), the `anthropic` SDK already retries
connection errors, 408, 409, 429 and 5xx with exponential backoff
internally, governed by the client's `max_retries` (SDK default: 2). We do
NOT wrap another retry loop around it -- that would multiply attempts
(max_retries^2) and double the backoff for no benefit. Instead we pass
`max_retries` explicitly when constructing the client, so the retry budget
is an intentional, visible choice rather than an unexamined SDK default,
and stays consistent (in attempt count) with the openai_compat adapter.
"""

import os

from core.script_gen.engine import ScriptEngine

# Total attempts (1 initial + retries) the SDK will make on a retryable
# error. Overridable with the same env var openai_compat.py uses, so the
# two adapters' retry budgets can be tuned in one place. The anthropic SDK's
# `max_retries` counts *retries*, not total attempts, hence the "- 1".
DEFAULT_MAX_RETRIES = 3


def _sdk_max_retries() -> int:
    raw = os.environ.get("MOODSCAPE_SCRIPT_MAX_RETRIES")
    if not raw:
        total_attempts = DEFAULT_MAX_RETRIES
    else:
        try:
            total_attempts = max(1, int(raw))
        except ValueError:
            total_attempts = DEFAULT_MAX_RETRIES
    return max(0, total_attempts - 1)


class AnthropicEngine(ScriptEngine):
    """Talk to Claude via the Messages API."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        client: object | None = None,
    ):
        self._model = model
        self._api_key_env = api_key_env
        self._client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not os.environ.get(self._api_key_env):
            raise RuntimeError(
                f"Claude needs an API key but {self._api_key_env} is not set. "
                f"Add {self._api_key_env}=... to your .env."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is not installed. "
                "Run: pip install anthropic"
            ) from exc

        api_key = os.environ.get(self._api_key_env)
        try:
            self._client = anthropic.Anthropic(
                api_key=api_key, max_retries=_sdk_max_retries()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct the Anthropic client for model "
                f"{self._model!r}: {exc}"
            ) from exc
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        client = self._get_client()

        # temperature is deliberately not forwarded: sampling parameters are
        # rejected on current thinking-enabled models.
        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                message = stream.get_final_message()

            text = "".join(
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic request failed for model {self._model!r}: {exc}"
            ) from exc

        if not text:
            raise RuntimeError(
                f"Anthropic returned no text content for model {self._model!r}."
            )
        return text
