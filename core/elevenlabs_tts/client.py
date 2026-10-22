"""Thin wrapper over the elevenlabs SDK — the ONLY module that imports it.

Retries transient failures (network/timeout, 408, 429, 5xx) with exponential
backoff; other 4xx fail immediately with a readable message.
"""

from __future__ import annotations

import base64
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
    for name in ("ELEVENLABS_API_KEY", "ELEVEN_LABS_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


def has_api_key() -> bool:
    return bool(get_api_key())


def model_id() -> str:
    return os.environ.get("MOODSCAPE_ELEVENLABS_MODEL", DEFAULT_MODEL)


def max_retries() -> int:
    return max(1, int(os.environ.get("MOODSCAPE_ELEVENLABS_MAX_RETRIES", "3")))


def _status_of(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _is_transient(status: int | None) -> bool:
    return status is None or status in (408, 429) or status >= 500


def _describe(status: int | None, exc: BaseException) -> str:
    hint = _STATUS_HINTS.get(status)
    prefix = f"ElevenLabs error {status}" if status else "ElevenLabs request failed"
    return f"{prefix}: {hint + ' — ' if hint else ''}{exc}"


def with_retries(fn, *, attempts: int, sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (ElevenLabsAPIError, TypeError, AttributeError):
            # Programming errors (bad kwargs, SDK API drift) are not transient:
            # surface them unchanged instead of retrying and masking them.
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


def _alignment_dict(body) -> dict:
    al = getattr(body, "alignment", None) or getattr(body, "normalized_alignment", None)
    if al is None:
        return {"characters": [], "starts": [], "ends": []}
    return {
        "characters": list(al.characters),
        "starts": [float(t) for t in al.character_start_times_seconds],
        "ends": [float(t) for t in al.character_end_times_seconds],
    }


class ElevenLabsClient:
    def __init__(self, api_key: str | None = None, sdk=None, sleep=time.sleep):
        key = api_key or get_api_key()
        if not key:
            raise ElevenLabsConfigError(
                "ELEVENLABS_API_KEY (or ELEVEN_LABS_API_KEY) is not set. Add it to .env to use the ElevenLabs provider."
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

    def convert_with_alignment(self, text: str, voice_id: str, *, model_id: str,
                               stability: float, similarity: float, seed: int,
                               previous_request_ids: list[str] | None = None,
                               ) -> tuple[bytes, str | None, dict]:
        """Like ``convert`` but via the timestamps endpoint (same cost): returns
        (pcm, request_id, {"characters", "starts", "ends"}). The alignment
        includes the leading [direction] tag characters."""
        from elevenlabs.types import VoiceSettings

        kwargs = dict(
            text=text, voice_id=voice_id, model_id=model_id, output_format=OUTPUT_FORMAT,
            voice_settings=VoiceSettings(stability=stability, similarity_boost=similarity,
                                         use_speaker_boost=True),
            seed=int(seed), apply_text_normalization="auto",
        )
        if previous_request_ids:
            kwargs["previous_request_ids"] = list(previous_request_ids)[-3:]

        def call() -> tuple[bytes, str | None, dict]:
            # Non-streaming endpoint: the raw response is a plain HttpResponse
            # (.headers, .data), not a context manager.
            resp = self._sdk.text_to_speech.with_raw_response.convert_with_timestamps(**kwargs)
            rid = _request_id(resp.headers)
            body = resp.data
            data = base64.b64decode(getattr(body, "audio_base_64", "") or "")
            if not data:
                raise RuntimeError("ElevenLabs returned no audio data.")
            return data, rid, _alignment_dict(body)

        return with_retries(call, attempts=self._attempts, sleep=self._sleep)

    def voice_exists(self, voice_id: str) -> bool:
        try:
            with_retries(lambda: self._sdk.voices.get(voice_id),
                         attempts=self._attempts, sleep=self._sleep)
            return True
        except ElevenLabsAPIError as exc:
            if exc.status in (400, 404):
                return False
            raise
