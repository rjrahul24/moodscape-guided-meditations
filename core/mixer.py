"""Mixing engine: ducking, overlay, fades, normalization, and export."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Ducking
# ---------------------------------------------------------------------------

def compute_duck_curve(
    voice_activity: np.ndarray,
    sample_rate: int = 24000,
    duck_amount_db: float = -8.0,
) -> np.ndarray:
    """Create a smooth gain curve that ducks music when voice is active.

    Uses a 2nd-order Butterworth lowpass at ~3 Hz to smooth transitions,
    producing natural-sounding attack/release behavior.

    Args:
        voice_activity: Bool array — True where voice is speaking.
        sample_rate: Audio sample rate.
        duck_amount_db: Negative dB amount to reduce music during speech.

    Returns:
        Linear gain curve (float32 array, same length as voice_activity).
    """
    envelope = voice_activity.astype(np.float32)

    # Smooth with a Butterworth lowpass (~3 Hz cutoff)
    nyquist = sample_rate / 2.0
    cutoff_hz = 3.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype="low")
    smoothed = filtfilt(b, a, envelope).astype(np.float32)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    # Convert to linear gain: 0 dB (no duck) where silent, duck_amount_db where speaking
    gain_db = smoothed * duck_amount_db
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)

    return gain_linear


# ---------------------------------------------------------------------------
# Track overlay / alignment
# ---------------------------------------------------------------------------

def overlay_tracks(
    voice: np.ndarray,
    music: np.ndarray,
    music_pre_roll_sec: float = 2.0,
    sample_rate: int = 24000,
) -> tuple[np.ndarray, np.ndarray]:
    """Align voice and music tracks. Music starts first (pre-roll).

    Returns:
        (aligned_voice, aligned_music) — same length, ready to sum.
    """
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)

    # Prepend silence to voice so music plays alone briefly first
    aligned_voice = np.concatenate([
        np.zeros(pre_roll_samples, dtype=np.float32),
        voice,
    ])

    total_length = len(aligned_voice)

    # Loop music if it's shorter than the total needed length
    if len(music) < total_length:
        repeats = (total_length // len(music)) + 1
        music = np.tile(music, repeats)

    aligned_music = music[:total_length]

    return aligned_voice, aligned_music


# ---------------------------------------------------------------------------
# Fades
# ---------------------------------------------------------------------------

def apply_fades(
    audio: np.ndarray,
    sample_rate: int = 24000,
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
    sample_rate: int = 24000,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Normalize audio to target LUFS using pyloudnorm."""
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)

    # Guard against near-silent audio (returns -inf)
    if not np.isfinite(loudness):
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
    sample_rate: int = 24000,
    duck_amount_db: float = -8.0,
    music_pre_roll_sec: float = 2.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Mix voice and music with ducking, fades, and normalization.

    Returns a mono float32 numpy array ready for export.
    """
    # 1. Align tracks
    aligned_voice, aligned_music = overlay_tracks(
        voice_audio, music_audio, music_pre_roll_sec, sample_rate
    )

    # 2. Extend voice_activity to match aligned length
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    aligned_activity = np.concatenate([
        np.zeros(pre_roll_samples, dtype=bool),
        voice_activity,
    ])
    # Pad if needed
    if len(aligned_activity) < len(aligned_voice):
        aligned_activity = np.concatenate([
            aligned_activity,
            np.zeros(len(aligned_voice) - len(aligned_activity), dtype=bool),
        ])
    aligned_activity = aligned_activity[: len(aligned_voice)]

    # 3. Compute and apply ducking
    duck_curve = compute_duck_curve(aligned_activity, sample_rate, duck_amount_db)
    ducked_music = aligned_music * duck_curve

    # 4. Sum tracks
    mixed = aligned_voice + ducked_music

    # 5. Fades
    mixed = apply_fades(mixed, sample_rate, fade_in_sec, fade_out_sec)

    # 6. Normalize
    mixed = normalize_loudness(mixed, sample_rate, target_lufs)

    return mixed


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    output_format: str = "wav",
) -> str:
    """Export audio to a temporary file. Returns the file path."""
    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    if output_format == "wav":
        sf.write(tmp_path, audio, sample_rate)
    elif output_format == "mp3":
        from pedalboard.io import AudioFile

        with AudioFile(tmp_path, "w", sample_rate, num_channels=1) as f:
            f.write(audio.reshape(1, -1))
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    return tmp_path
