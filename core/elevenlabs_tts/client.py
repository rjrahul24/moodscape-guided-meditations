"""Thin wrapper over the elevenlabs SDK — the ONLY module that imports it.

Retries transient failures (network/timeout, 429, 5xx) with exponential
backoff; other 4xx fail immediately with a readable message.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("moodscape.elevenlabs")

DEFAULT_MODEL = "eleven_v4"
OUTPUT_FORMAT = "pcm_48000"
_STATUS_HINTS = {
    401: "invalid ElevenLabs API key (check ELEVENLABS_API_KEY)",
    402: "ElevenLabs account is out of credits",
    404: "voice or model not found in this ElevenLabs account",
}


class ElevenLabsConfigError(RuntimeError):
    """Missing or invalid local configuration (no network call was made)."""


class ElevenLabsAPIError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def get_api_key() -> str:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()


def has_api_key() -> bool:
    return bool(get_api_key())


def model_id() -> str:
    return os.environ.get("MOODSCAPE_ELEVENLABS_MODEL", DEFAULT_MODEL)


def max_retries() -> int:
    return max(1, int(os.environ.get("MOODSCAPE_ELEVENLABS_MAX_RETRIES", "3")))


def _status_of(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _is_transient(status: int | None) -> bool:
    return status is None or status == 429 or status >= 500


def _describe(status: int | None, exc: BaseException) -> str:
    hint = _STATUS_HINTS.get(status)
    prefix = f"ElevenLabs error {status}" if status else "ElevenLabs request failed"
    return f"{prefix}: {hint + ' — ' if hint else ''}{exc}"


def with_retries(fn, *, attempts: int, sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ElevenLabsAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            status = _status_of(exc)
            if not _is_transient(status) or attempt >= attempts:
                raise ElevenLabsAPIError(status, _describe(status, exc)) from exc
            delay = 2 ** (attempt - 1)
            logger.warning("ElevenLabs transient failure (status=%s, attempt %d/%d); retrying in %ss",
                           status, attempt, attempts, delay)
            sleep(delay)


def _request_id(headers) -> str | None:
    for key in ("request-id", "x-request-id", "x-trace-id"):
        value = headers.get(key)
        if value:
            return value
    return None


class ElevenLabsClient:
    def __init__(self, api_key: str | None = None, sdk=None, sleep=time.sleep):
        key = api_key or get_api_key()
        if not key:
            raise ElevenLabsConfigError(
                "ELEVENLABS_API_KEY is not set. Add it to .env to use the ElevenLabs provider."
            )
        if sdk is None:
            from elevenlabs.client import ElevenLabs
            sdk = ElevenLabs(api_key=key)
        self._sdk = sdk
        self._sleep = sleep
        self._attempts = max_retries()

    def convert(self, text: str, voice_id: str, *, model_id: str, stability: float,
                similarity: float, seed: int,
                previous_request_ids: list[str] | None = None) -> tuple[bytes, str | None]:
        from elevenlabs.types import VoiceSettings

        kwargs = dict(
            text=text, voice_id=voice_id, model_id=model_id, output_format=OUTPUT_FORMAT,
            voice_settings=VoiceSettings(stability=stability, similarity_boost=similarity,
                                         use_speaker_boost=True),
            seed=int(seed), apply_text_normalization="auto",
        )
        if previous_request_ids:
            kwargs["previous_request_ids"] = list(previous_request_ids)[-3:]

        def call() -> tuple[bytes, str | None]:
            with self._sdk.text_to_speech.with_raw_response.convert(**kwargs) as resp:
                rid = _request_id(resp.headers)
                data = b"".join(chunk for chunk in resp.data if chunk)
            if not data:
                raise RuntimeError("ElevenLabs returned no audio data.")
            return data, rid

        return with_retries(call, attempts=self._attempts, sleep=self._sleep)

    def voice_exists(self, voice_id: str) -> bool:
        try:
            self._sdk.voices.get(voice_id)
            return True
        except Exception as exc:  # noqa: BLE001
            if _status_of(exc) in (400, 404):
                return False
            raise ElevenLabsAPIError(_status_of(exc), _describe(_status_of(exc), exc)) from exc
