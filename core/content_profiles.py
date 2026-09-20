"""Content-type profiles — tunings that distinguish guided meditations from sleep stories.

A single ``content_type`` ("meditation" | "sleep_story") threads through the UI,
pipeline, preprocessors, and mixer. ``"meditation"`` is the default at every layer so
the long-tuned meditation path is unchanged; the sleep-story profile only diverges via
additive, gated branches.

Two kinds of values live here:

  * **Slider-backed defaults** (``speed``, ``duck_amount_db``, ``reverb_amount``,
    ``fade_in_sec``, ``fade_out_sec``) — the UI uses these to *pre-fill* the existing
    controls when the content type changes. The user can still override any of them,
    and whatever value the slider holds is what reaches ``MeditationPipeline.generate``.

  * **Behavioural values** consumed inside the pipeline that have no slider:
    ``kokoro_paragraph_pause_sec`` / ``f5_paragraph_pause_sec`` (how long a blank-line
    paragraph break pauses for) and the ``bed`` block (sleep-story ducking that stays
    softer and more constant). These are looked up by ``content_type`` directly.

Sleep stories differ from meditations in three ways: continuous narration (shorter
paragraph pauses), the same soft delivery at a slightly slower pace, and a softer,
more-constant ambient bed.
"""

DEFAULT_CONTENT_TYPE = "meditation"

CONTENT_PROFILES: dict[str, dict] = {
    # Mirrors today's pipeline/UI defaults exactly — must stay in lock-step with the
    # defaults in MeditationPipeline.generate() and app.py so meditation never regresses.
    "meditation": {
        "label": "Guided Meditation",
        "speed": 0.90,
        "duck_amount_db": -16.0,
        "music_volume_db": -16.0,
        "reverb_amount": 0.15,
        "fade_in_sec": 1.5,
        "fade_out_sec": 6.0,
        "kokoro_paragraph_pause_sec": 6.5,
        "f5_paragraph_pause_sec": 3.0,
        # None => mixer uses its existing meditation-tuned ducking/calibration.
        "bed": None,
    },
    # Sleep stories: continuous narration, slightly slower soft delivery, and a soft,
    # near-constant ambient bed. Paragraph breaks pause far less than in meditations so
    # the story flows; the bed dips little and recovers slowly.
    "sleep_story": {
        "label": "Sleep Story",
        "speed": 0.85,
        "duck_amount_db": -11.0,
        "music_volume_db": -16.0,
        "reverb_amount": 0.18,
        "fade_in_sec": 2.5,
        "fade_out_sec": 10.0,
        "kokoro_paragraph_pause_sec": 2.5,
        "f5_paragraph_pause_sec": 1.5,
        # Gentler breathing-duck + narrower calibration so the bed stays soft and
        # nearly constant instead of swinging dramatically between speech and pauses.
        # NOTE: duck *depth* is the duck_amount_db slider (-11 above), so it is not
        # repeated here. These are the breathing-duck params with no slider, plus the
        # calibration targets.
        "bed": {
            # Extra kwargs forwarded to mixer.apply_breathing_duck / compute_breathing_gain_db
            "duck_kwargs": {
                "release_ms": 2500.0,   # slow recovery (meditation 1500)
                "lift_db": 0.5,         # minimal pause lift (meditation 1.5)
            },
            # calibrate_music_bed target overrides — small speech/pause gap keeps the
            # bed from rising/falling drastically (meditation 30.5 / 14.5).
            "speech_offset_lu": 22.0,
            "pause_offset_lu": 16.0,
        },
    },
}


def get_profile(content_type: str | None) -> dict:
    """Return the profile dict for ``content_type``, defaulting to meditation.

    Unknown or ``None`` values fall back to the meditation profile so callers can pass
    raw input safely.
    """
    if content_type is None:
        return CONTENT_PROFILES[DEFAULT_CONTENT_TYPE]
    return CONTENT_PROFILES.get(content_type, CONTENT_PROFILES[DEFAULT_CONTENT_TYPE])


def normalize_content_type(value: str | None) -> str:
    """Map a UI label or raw key to a canonical content_type key.

    Accepts the canonical keys ("meditation", "sleep_story") and the UI labels
    ("Guided Meditation", "Sleep Story"), case-insensitively. Falls back to the
    default (meditation) for anything unrecognised.
    """
    if not value:
        return DEFAULT_CONTENT_TYPE
    v = value.strip().lower()
    if v in CONTENT_PROFILES:
        return v
    for key, profile in CONTENT_PROFILES.items():
        if v == profile["label"].lower():
            return key
    return DEFAULT_CONTENT_TYPE
