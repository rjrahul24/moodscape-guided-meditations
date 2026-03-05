"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
import math
from pedalboard import Pedalboard, Gain

SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Ducking (Envelope Follower)
# ---------------------------------------------------------------------------

def apply_ducking_envelope_follower(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -5.0,
    chunk_ms: float = 10.0,
    attack_ms: float = 50.0,
    release_ms: float = 500.0,
) -> np.ndarray:
    """Chunk-based Envelope Follower for auto-ducking music under voice.
    
    Processes audio in short chunks (~10ms). Calculates the RMS of the voice
    in that chunk to detect presence. Drives a smoothed target gain curve
    applied to the music via a Pedalboard Gain effect.
    """
    chunk_samples = int((chunk_ms / 1000.0) * sample_rate)
    total_samples = len(music_audio)
    
    # Pre-allocate output array
    ducked_music = np.zeros_like(music_audio, dtype=np.float32)
    
    # Envelope state
    current_gain_db = 0.0
    
    # Calculate smoothing factors (alpha) for exponential moving average
    # formula: alpha = exp(-chunk_time / time_constant)
    attack_alpha = math.exp(-(chunk_ms) / attack_ms)
    release_alpha = math.exp(-(chunk_ms) / release_ms)
    
    duck_threshold_rms = 0.03
    
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        
        # 1. Measure voice RMS in this chunk
        voice_chunk = voice_audio[start:end]
        if len(voice_chunk) == 0:
            rms = 0.0
        else:
            rms = np.sqrt(np.mean(voice_chunk**2))
            
        # 2. Determine target gain based on voice presence
        target_gain_db = duck_amount_db if rms > duck_threshold_rms else 0.0
        
        # 3. Smooth the gain transition (Attack/Release logistics)
        if target_gain_db < current_gain_db:
            # Ducking in (Attack) - moving from 0dB down to e.g. -5dB
            current_gain_db = (attack_alpha * current_gain_db) + ((1.0 - attack_alpha) * target_gain_db)
        else:
            # Recovering (Release) - moving from e.g. -5dB back up to 0dB
            current_gain_db = (release_alpha * current_gain_db) + ((1.0 - release_alpha) * target_gain_db)
            
        # 4. Apply gain to music chunk via Pedalboard
        pb = Pedalboard([Gain(gain_db=current_gain_db)])
        music_chunk = music_audio[start:end]
        # reshape(1,-1) into pedalboard, squeeze back out
        music_chunk_2d = music_chunk.reshape(1, -1)
        processed_chunk = pb(music_chunk_2d, sample_rate).squeeze(0)
        
        ducked_music[start:end] = processed_chunk
        
    return ducked_music


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

def calculate_loudness_gain(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
) -> float:
    """Calculate the linear gain multiplier to hit a target LUFS."""
    min_samples = int(0.4 * sample_rate)
    if len(audio) < min_samples:
        return 1.0

    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return 1.0

    if not np.isfinite(loudness):
        return 1.0

    if abs(target_lufs - loudness) > 40.0:
        return 1.0

    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    return float(gain_linear)

def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """Normalize audio to target LUFS. Returns float32, clipped to [-1, 1]."""
    gain = calculate_loudness_gain(audio, sample_rate, target_lufs)
    normalized = audio * gain
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
    if len(aligned_music) < target_len:
        aligned_music = np.concatenate([
            aligned_music,
            np.zeros(target_len - len(aligned_music), dtype=np.float32),
        ])
    aligned_music = aligned_music[:target_len]

    # 4. Compute and apply ducking (further reduces music during speech)
    ducked_music = apply_ducking_envelope_follower(
        aligned_voice,
        aligned_music,
        sample_rate=sample_rate,
        duck_amount_db=duck_amount_db,
        chunk_ms=10.0,
        attack_ms=50.0,
        release_ms=500.0,
    )

    # 5. Sum voice + ducked music
    mixed = aligned_voice + ducked_music

    # 6. Apply fades
    mixed = apply_fades(mixed, sample_rate, fade_in_sec, fade_out_sec)

    # 7. Do not normalize loudness here, as we will chunk-stream the mix through 
    # mastering EQ and normalizing logic in export_audio to save heap memory.

    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_for_export(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int = 44100,
) -> np.ndarray:
    """Resample audio to target sample rate for final export.

    Uses torchaudio's polyphase resampler.  Clips the result to [-1, 1] to
    counter Gibbs-phenomenon overshoot introduced by the anti-alias filter.
    """
    if source_rate == target_rate:
        return audio

    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, source_rate, target_rate)
    result = resampled.squeeze(0).numpy()
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stem export
# ---------------------------------------------------------------------------

def export_stems(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Export voice and music as separate stem files.

    Saves each stem as a 24-bit WAV so the user can re-balance or remix
    later without regenerating.

    Returns:
        Dict with keys 'voice' and 'music' pointing to file paths.
    """
    import os
    import soundfile as sf

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="moodscape_stems_")
    os.makedirs(output_dir, exist_ok=True)

    voice_path = os.path.join(output_dir, "narration_stem.wav")
    music_path = os.path.join(output_dir, "music_stem.wav")

    sf.write(voice_path, voice_audio, sample_rate, subtype="PCM_24")
    sf.write(music_path, music_audio, sample_rate, subtype="PCM_24")

    return {"voice": voice_path, "music": music_path}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    output_format: str = "wav",
    target_sample_rate: int = 44100,
    master_chain: Pedalboard | None = None,
) -> str:
    """Stream audio out to temp file with Pedalboard plugins and normalization.
    
    Reads from the memory array in bounded chunks, applies a calculated linear 
    target -16.0 LUFS gain, applies the final Pedalboard master chain, and streams 
    directly to a file to prevent immense memory spikes.
    """
    from pedalboard.io import AudioFile
    
    export_rate = target_sample_rate
    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    # Pre-calculate the linear gain to bring the whole file to -16 LUFS
    mix_lufs_gain = calculate_loudness_gain(audio, sample_rate, -16.0)
    
    # 20 second chunks
    chunk_samples = int(20.0 * sample_rate)
    total_samples = len(audio)

    # Initialize Pedalboard AudioFile struct
    f = AudioFile(tmp_path, "w", samplerate=export_rate, num_channels=1)

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        
        # Resample chunk if needed
        if sample_rate != export_rate:
            chunk = resample_for_export(chunk, sample_rate, export_rate)
            
        # Apply normalization gain
        chunk = chunk * mix_lufs_gain
        
        # Apply Pedalboard mastering chain if present
        if master_chain:
            chunk_2d = chunk.reshape(1, -1)
            chunk = master_chain(chunk_2d, export_rate).squeeze(0)
            
        # write directly as float32
        f.write(chunk.reshape(1, -1).astype(np.float32))

    f.close()
    return tmp_path
