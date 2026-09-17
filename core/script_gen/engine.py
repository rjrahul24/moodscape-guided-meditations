"""ScriptEngine ABC and the provider registry.

Mirrors core/speech_engine.py: one narrow interface, several concrete
engines, and the rest of the system stays engine-agnostic.

Ollama, OpenRouter, Together, Fireworks and Groq all speak the
OpenAI-compatible /v1/chat/completions protocol, so one adapter with a
configurable base_url covers local *and* every hosted open-weight provider.
Only Claude needs its own adapter.
"""

from abc import ABC, abstractmethod

# Every entry here is OpenAI-compatible and served by OpenAICompatEngine.
PROVIDER_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Ollama runs locally and needs no key, so it is absent here by design.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ScriptEngine(ABC):
    """Interface every script-generation backend must implement.

    Deliberately narrow: one text-in, text-out call. Generator, judge and
    repair are all the same operation with different prompts.
    """

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """Run one completion and return the assistant's text.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Ceiling on generated tokens.
            temperature: Sampling temperature.

        Returns:
            The assistant's response as plain text.

        Raises:
            RuntimeError: On an unrecoverable backend failure, with a message
                naming the provider and how to fix it.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'ollama:qwen3:30b'."""


class FakeScriptEngine(ScriptEngine):
    """Test double returning canned responses. No network, no model.

    Lets the generator/judge/repair loop be tested exhaustively without
    spending a token.
    """

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("FakeScriptEngine needs at least one response.")
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    @property
    def name(self) -> str:
        return "fake:canned"


def parse_engine_spec(spec: str) -> tuple[str, str]:
    """Split a 'provider:model' spec.

    Splits on the FIRST colon only — Ollama model tags such as 'qwen3:30b'
    contain colons that belong to the model name.

    Raises:
        ValueError: If the spec has no colon or an empty half.
    """
    if ":" not in spec:
        raise ValueError(
            f"Malformed engine spec {spec!r}. Expected 'provider:model', "
            "e.g. 'ollama:qwen3:30b' or 'anthropic:claude-opus-5'."
        )
    provider, model = spec.split(":", 1)
    provider, model = provider.strip(), model.strip()
    if not provider or not model:
        raise ValueError(
            f"Malformed engine spec {spec!r}. Both provider and model are required."
        )
    return provider, model


def build_engine(spec: str) -> ScriptEngine:
    """Construct the engine named by a 'provider:model' spec."""
    provider, model = parse_engine_spec(spec)

    if provider == "anthropic":
        from core.script_gen.adapters.anthropic_api import AnthropicEngine

        return AnthropicEngine(model)

    if provider in PROVIDER_BASE_URLS:
        from core.script_gen.adapters.openai_compat import OpenAICompatEngine

        return OpenAICompatEngine(
            provider=provider,
            model=model,
            base_url=PROVIDER_BASE_URLS[provider],
            api_key_env=PROVIDER_KEY_ENV.get(provider),
        )

    known = ", ".join(sorted([*PROVIDER_BASE_URLS, "anthropic"]))
    raise ValueError(
        f"Unknown provider {provider!r} in spec {spec!r}. Known providers: {known}."
    )
