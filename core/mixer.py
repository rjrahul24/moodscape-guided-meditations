"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, filtfilt


SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Ducking
# ---------------------------------------------------------------------------

def compute_duck_curve(
    voice_activity: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -8.0,
    cutoff_hz: float = 3.0,
) -> np.ndarray:
    """Create a smooth linear gain curve for music ducking.

    Uses Butterworth lowpass at ~3 Hz to create natural attack/release.
    At 3 Hz cutoff, the duck-down takes ~300 ms and the swell-back takes ~800 ms.

    Returns float32 array of linear gain values in range [10^(duck_db/20), 1.0].
    """
    envelope = voice_activity.astype(np.float32)

    nyquist = sample_rate / 2.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype="low")
    smoothed = filtfilt(b, a, envelope).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    gain_db = smoothed * duck_amount_db
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)

    return gain_linear


# ---------------------------------------------------------------------------
# Track overlay / alignment
# ---------------------------------------------------------------------------

def _loop_with_crossfade(
    music: np.ndarray,
    target_length: int,
    sample_rate: int = SAMPLE_RATE,
    crossfade_sec: float = 1.0,
) -> np.ndarray:
    """Loop music array to target_length with crossfade at boundaries."""
    crossfade_samples = min(int(crossfade_sec * sample_rate), len(music) // 2)
    result = music.copy()

    while len(result) < target_length + crossfade_samples:
        fo = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fi = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        overlap = result[-crossfade_samples:] * fo + music[:crossfade_samples] * fi
        result = np.concatenate([result[:-crossfade_samples], overlap, music[crossfade_samples:]])

    return result


def overlay_tracks(
    voice: np.ndarray,
    music: np.ndarray,
    music_pre_roll_sec: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Align voice and music. Music starts first by pre_roll_sec seconds.

    Returns (aligned_voice, aligned_music) — both same length, ready to sum.
    """
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)

    # Prepend silence to voice so music plays alone briefly first
    aligned_voice = np.concatenate([
        np.zeros(pre_roll_samples, dtype=np.float32),
        voice,
    ])

    total_length = len(aligned_voice)

    # Loop music with crossfade if shorter than needed
    if len(music) < total_length:
        music = _loop_with_crossfade(music, total_length, sample_rate)

    aligned_music = music[:total_length].astype(np.float32)

    return aligned_voice, aligned_music


# ---------------------------------------------------------------------------
# Fades
# ---------------------------------------------------------------------------

def apply_fades(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
) -> np.ndarray:
    """Apply linear fade-in and fade-out to audio."""
    result = audio.copy()

    fade_in_samples = int(fade_in_sec * sample_rate)
    if 0 < fade_in_samples < len(result):
        result[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)

    fade_out_samples = int(fade_out_sec * sample_rate)
    if 0 < fade_out_samples < len(result):
        result[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Loudness normalization
# ---------------------------------------------------------------------------

def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Normalize audio to target LUFS. Returns float32, clipped to [-1, 1]."""
    # pyloudnorm requires at least 400ms of audio for BS.1770 gating blocks
    min_samples = int(0.4 * sample_rate)
    if len(audio) < min_samples:
        return audio

    meter = pyln.Meter(sample_rate)

    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return audio

    if not np.isfinite(loudness):
        return audio

    # Guard against extreme gain changes that indicate a problem
    if abs(target_lufs - loudness) > 40.0:
        return audio

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Full mix
# ---------------------------------------------------------------------------

def mix(
    voice_audio: np.ndarray,
    voice_activity: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -4.0,
    music_volume_db: float = -3.0,
    music_pre_roll_sec: float = 2.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Full mix pipeline: align → level → duck → overlay → fades → normalize.

    Args:
        duck_amount_db: Additional dB reduction during speech on top of
            music_volume_db. -4 dB is a gentle dip that keeps music present.
        music_volume_db: Baseline music level in dB (applied before ducking).
            Sets the music as a soothing background layer at all times.
            During pauses music sits at this level; during speech it drops
            by an additional duck_amount_db.

    Called after voice FX and music FX have already been applied.
    Returns mixed mono float32 array ready for master chain + export.
    """
    # 1. Align voice and music with pre-roll
    aligned_voice, aligned_music = overlay_tracks(
        voice_audio, music_audio, music_pre_roll_sec, sample_rate
    )

    # 2. Set baseline music level so it sits behind the voice
    music_gain = np.float32(10.0 ** (music_volume_db / 20.0))
    aligned_music = aligned_music * music_gain

    # 3. Extend voice_activity to match pre-roll offset
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    aligned_activity = np.concatenate([
        np.zeros(pre_roll_samples, dtype=bool),
        voice_activity,
    ])
    # Match exact length
    target_len = len(aligned_voice)
    if len(aligned_activity) < target_len:
        aligned_activity = np.concatenate([
            aligned_activity,
            np.zeros(target_len - len(aligned_activity), dtype=bool),
        ])
    aligned_activity = aligned_activity[:target_len]

    # 4. Compute and apply ducking (further reduces music during speech)
    duck_curve = compute_duck_curve(aligned_activity, sample_rate, duck_amount_db)
    ducked_music = aligned_music * duck_curve

    # 5. Sum voice + ducked music
    mixed = aligned_voice + ducked_music

    # 6. Apply fades
    mixed = apply_fades(mixed, sample_rate, fade_in_sec, fade_out_sec)

    # 7. Normalize loudness
    mixed = normalize_loudness(mixed, sample_rate, target_lufs)

    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    output_format: str = "wav",
) -> str:
    """Export audio to a temp file. Returns absolute path."""
    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    if output_format == "wav":
        sf.write(tmp_path, audio, sample_rate, subtype="PCM_24")
    elif output_format == "mp3":
        from pedalboard.io import AudioFile

        with AudioFile(tmp_path, "w", samplerate=sample_rate, num_channels=1, quality=0.2) as f:
            f.write(audio.reshape(1, -1).astype(np.float32))
    else:
        raise ValueError(f"Unsupported format: {output_format!r}")

    return tmp_path
