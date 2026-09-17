"""Adapter for every OpenAI-compatible /v1/chat/completions endpoint.

One adapter covers local Ollama and the hosted open-weight providers
(OpenRouter, Together, Fireworks, Groq) because they all speak the same
protocol — only the base URL and the API-key env var differ.

Uses httpx directly rather than adding another SDK dependency.
"""

import os

import httpx

from core.script_gen.engine import ScriptEngine

# Generous default: a local 32B model on Apple Silicon can take minutes for a
# few thousand tokens.
DEFAULT_TIMEOUT_SEC = 300.0


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
    ):
        self._provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._transport = transport

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
        headers = self._headers()

        try:
            with httpx.Client(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            hint = (
                "Is `ollama serve` running?"
                if self._provider == "ollama"
                else "Check network access and the provider's status."
            )
            raise RuntimeError(
                f"Cannot reach {self._provider} at {url}. {hint}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"{self._provider} timed out after {self._timeout:.0f}s at {url}. "
                "A large local model may need a longer timeout."
            ) from exc
        except httpx.RequestError as exc:
            # Catch-all for every other transport failure (ReadError,
            # WriteError, ProtocolError, RemoteProtocolError, ProxyError,
            # ...). ConnectError and TimeoutException are also RequestError
            # subclasses but are caught above with better messages, so
            # Python's first-match rule keeps those. A mid-response
            # connection reset from a local model server is ordinary, not
            # exceptional, so it must not escape as a bare httpx exception.
            raise RuntimeError(
                f"{self._provider} request failed at {url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{self._provider} returned HTTP {exc.response.status_code} "
                f"for model {self._model!r}: {exc.response.text[:400]}"
            ) from exc
        except ValueError as exc:
            # response.json() raises json.JSONDecodeError (a ValueError
            # subclass) on a non-JSON 200 body -- e.g. a proxy or load
            # balancer returning an HTML error page with status 200, or a
            # truncated response. Catching ValueError (not JSONDecodeError)
            # also covers alternative JSON backends that raise other
            # ValueError subclasses.
            raise RuntimeError(
                f"{self._provider} returned a non-JSON response body from "
                f"{url}: {response.text[:400]}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self._provider} returned an unexpected response shape: "
                f"{str(data)[:400]}"
            ) from exc
