"""Adapter for every OpenAI-compatible /v1/chat/completions endpoint.

One adapter covers local Ollama and the hosted open-weight providers
(OpenRouter, Together, Fireworks, Groq) because they all speak the same
protocol — only the base URL and the API-key env var differ.

Uses httpx directly rather than adding another SDK dependency. Because there
is no SDK here, this module hand-rolls its own retry-with-backoff loop for
transient failures (connection errors, timeouts, 408, 409, 429, 5xx).
Compare with adapters/anthropic_api.py, where the official SDK already
retries and no second loop is added.
"""

import logging
import os
import random
import time

import httpx

from core.script_gen.engine import ScriptEngine

logger = logging.getLogger(__name__)

# Generous default: a local 32B model on Apple Silicon can take minutes for a
# few thousand tokens.
DEFAULT_TIMEOUT_SEC = 300.0

# Total attempts (1 initial + retries) for a transient failure. Overridable
# so a slow/flaky provider can be tuned without a code change.
DEFAULT_MAX_RETRIES = 3

# Backoff shape: base * 2**(attempt-1), jittered DOWN into [0.8, 1.0] of that
# value so consecutive delays never overlap (deterministic growth for
# tests), then capped so a long backoff -- or a hostile/mistaken
# Retry-After header -- cannot stall a fire-and-forget job indefinitely.
BACKOFF_BASE_SEC = 1.0
BACKOFF_CAP_SEC = 30.0
_JITTER_LOW = 0.8
_JITTER_HIGH = 1.0


def _max_retries() -> int:
    raw = os.environ.get("MOODSCAPE_SCRIPT_MAX_RETRIES")
    if not raw:
        return DEFAULT_MAX_RETRIES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_RETRIES


def _backoff_delay(attempt: int) -> float:
    """Delay before retrying after `attempt` (1-based, already-failed attempts)."""
    base = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
    jittered = base * random.uniform(_JITTER_LOW, _JITTER_HIGH)
    return min(jittered, BACKOFF_CAP_SEC)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header's delta-seconds form. HTTP-date form and
    anything unparseable falls back to ordinary exponential backoff."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


class _Transient(Exception):
    """Internal signal: this attempt failed but is worth retrying."""

    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class OpenAICompatEngine(ScriptEngine):
    """Talk to any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self._provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._transport = transport
        self._sleep = sleep

    def _native_base(self) -> str:
        """Ollama's native API root.

        base_url points at the OpenAI-compatible surface
        (http://host:11434/v1), but keep_alive and the model list are only
        available on the native API one level up.
        """
        return self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url

    def unload(self) -> None:
        """Ask Ollama to evict this model immediately.

        No-op for every other provider. Failures are logged and swallowed: a
        model that will not unload is a memory-pressure problem for the next
        stage, not a reason to fail a job that has already produced a script.
        """
        if self._provider != "ollama":
            return
        url = f"{self._native_base()}/api/generate"
        try:
            with httpx.Client(timeout=30.0, transport=self._transport) as client:
                response = client.post(url, json={"model": self._model, "keep_alive": 0})
            if response.status_code >= 400:
                logger.warning(
                    "Could not unload %s from Ollama (HTTP %d); the next stage may be "
                    "memory-constrained.",
                    self._model,
                    response.status_code,
                )
        except Exception:
            logger.warning(
                "Could not unload %s from Ollama; the next stage may be "
                "memory-constrained.",
                self._model,
                exc_info=True,
            )

    def preflight(self) -> None:
        """Check the model is pulled before a long run begins.

        No-op for every provider but Ollama. An unreachable server is NOT
        reported here: complete() already produces a good message for that,
        and duplicating it would only make the error worse.

        Raises:
            RuntimeError: If Ollama is reachable and the model is absent.
        """
        if self._provider != "ollama":
            return
        url = f"{self._native_base()}/api/tags"
        try:
            with httpx.Client(timeout=10.0, transport=self._transport) as client:
                response = client.get(url)
            # Treat any non-200 as "cannot determine model list" (connectivity
            # problem, routing issue, etc.). complete() will report the real
            # problem with a better message.
            if response.status_code != 200:
                logger.debug(
                    "Preflight got HTTP %d from %s; skipping.",
                    response.status_code,
                    url,
                )
                return
            names = {
                entry.get("name", "")
                for entry in response.json().get("models", [])
            }
        except Exception:
            logger.debug("Preflight could not reach %s; skipping.", url, exc_info=True)
            return

        if self._model not in names:
            raise RuntimeError(
                f"Ollama does not have model {self._model!r}. Pull it first:\n"
                f"    ollama pull {self._model}"
            )

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env is None:
            return headers
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"{self._provider} needs an API key but {self._api_key_env} is "
                f"not set. Add {self._api_key_env}=... to your .env."
            )
        headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._provider == "ollama":
            payload["reasoning_effort"] = "none"
        # Raises immediately (not retried) if the key is missing -- this is
        # a config error, not a transient failure, and no request is sent.
        headers = self._headers()

        max_attempts = _max_retries()
        last_error: _Transient | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._send_once(url, payload, headers)
            except _Transient as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                delay = (
                    min(exc.retry_after, BACKOFF_CAP_SEC)
                    if exc.retry_after is not None
                    else _backoff_delay(attempt)
                )
                logger.warning(
                    "%s: attempt %d/%d failed (%s); retrying in %.2fs",
                    self._provider,
                    attempt,
                    max_attempts,
                    exc.reason,
                    delay,
                )
                self._sleep(delay)

        raise RuntimeError(
            f"{self._provider} failed after {max_attempts} attempts at "
            f"{url}: {last_error.reason}"
        ) from last_error

    def _send_once(
        self, url: str, payload: dict, headers: dict[str, str]
    ) -> str:
        """One request/parse attempt.

        Raises _Transient for failures worth retrying (connection errors,
        timeouts, 429, 5xx) and RuntimeError directly for everything else
        (bad request, malformed body) so the caller's retry loop never
        retries a failure that would repeat identically.
        """
        try:
            with httpx.Client(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.ConnectError as exc:
            hint = (
                "Is `ollama serve` running?"
                if self._provider == "ollama"
                else "Check network access and the provider's status."
            )
            raise _Transient(
                f"cannot reach {self._provider} at {url}. {hint}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise _Transient(
                f"{self._provider} timed out after {self._timeout:.0f}s at "
                f"{url}. A large local model may need a longer timeout."
            ) from exc
        except httpx.RequestError as exc:
            # Catch-all for every other transport failure (ReadError,
            # WriteError, ProtocolError, RemoteProtocolError, ProxyError,
            # ...). ConnectError and TimeoutException are also RequestError
            # subclasses but are caught above with better messages, so
            # Python's first-match rule keeps those. A mid-response
            # connection reset from a local model server is ordinary, not
            # exceptional, so it must not escape as a bare httpx exception.
            raise _Transient(
                f"{self._provider} request failed at {url}: {exc}"
            ) from exc

        if response.status_code in (408, 409, 429) or response.status_code >= 500:
            # Transient: request timeouts, conflicts, rate limiting, and
            # server-side failures are worth retrying -- this mirrors the
            # anthropic SDK's own transient set (see module docstring) so
            # the two adapters' documented retry guarantee is actually true.
            # Everything else in 4xx (bad model, malformed request, auth)
            # will fail identically every time.
            raise _Transient(
                f"{self._provider} returned HTTP {response.status_code} "
                f"for model {self._model!r}: {response.text[:400]}",
                retry_after=_parse_retry_after(
                    response.headers.get("retry-after")
                ),
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{self._provider} returned HTTP {response.status_code} "
                f"for model {self._model!r}: {response.text[:400]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            # response.json() raises json.JSONDecodeError (a ValueError
            # subclass) on a non-JSON 200 body -- e.g. a proxy or load
            # balancer returning an HTML error page with status 200, or a
            # truncated response. Catching ValueError (not JSONDecodeError)
            # also covers alternative JSON backends that raise other
            # ValueError subclasses. The request succeeded; the content is
            # wrong, so this is not retried.
            raise RuntimeError(
                f"{self._provider} returned a non-JSON response body from "
                f"{url}: {response.text[:400]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self._provider} returned an unexpected response shape: "
                f"{str(data)[:400]}"
            ) from exc

        if not content:
            # A 200 body with {"content": null} is standard for reasoning
            # models on providers like OpenRouter/Groq when the reasoning
            # trace consumed the whole token budget. Left unguarded, this
            # returns None, and the caller (generator.py) later does a
            # string operation on it, raising a bare TypeError that escapes
            # every RuntimeError handler in the auto-generation pipeline.
            # Not retried: the request succeeded, the content is wrong.
            raise RuntimeError(
                f"{self._provider} returned empty content for model "
                f"{self._model!r}. Reasoning models may need a non-reasoning "
                "variant or a different endpoint."
            )
        return content
