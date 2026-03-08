"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
import math
from pedalboard import Pedalboard

SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Ducking (Lookahead Sidechain)
# ---------------------------------------------------------------------------

def apply_rms_ducking(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -5.0,
    frame_ms: float = 10.0,
    attack_ms: float = 50.0,
    release_ms: float = 500.0,
    lookahead_ms: float = 75.0,
    duck_threshold_rms: float = 0.03,
) -> np.ndarray:
    """Vectorised offline lookahead sidechain ducker.

    Unlike a reactive envelope follower, this function computes the complete
    gain envelope from the *entire* voice array in advance, then shifts it
    back in time by ``lookahead_ms`` so the music begins fading *before*
    the first syllable of each phrase — matching broadcast / DAW behaviour.

    Algorithm:
      1. Compute per-frame RMS of voice across the full timeline.
      2. Build a target-gain sequence (0 dB vs duck_amount_db per frame).
      3. Shift the sequence back by lookahead_frames (frames before voice onset).
      4. Apply vectorised attack/release EMA smoothing.
      5. Upsample frame-rate envelope to sample-rate via np.interp.
      6. Convert to linear gain and multiply music directly.

    Args:
        voice_audio:       1-D float32 voice array aligned to music_audio.
        music_audio:       1-D float32 music to duck.
        sample_rate:       Sample rate of both arrays.
        duck_amount_db:    Target attenuation during speech (negative dB).
        frame_ms:          Analysis frame size in ms (default 10 ms).
        attack_ms:         Time to reach full duck once onset detected.
        release_ms:        Time to recover after voice stops.
        lookahead_ms:      How far ahead of voice onset to start ducking.
        duck_threshold_rms: Minimum voice RMS to trigger ducking.

    Returns:
        Ducked music as 1-D float32 array, same length as music_audio.
    """
    total_samples = music_audio.shape[-1]
    frame_samples = max(1, int((frame_ms / 1000.0) * sample_rate))
    lookahead_frames = max(0, int(round((lookahead_ms / 1000.0) * sample_rate / frame_samples)))

    # ── 1. Compute per-frame RMS for the voice ──────────────────────────────
    # Pad voice to a whole number of frames
    n_frames = math.ceil(total_samples / frame_samples)
    padded = np.zeros(n_frames * frame_samples, dtype=np.float32)
    v_len = min(len(voice_audio), total_samples)
    padded[:v_len] = voice_audio[:v_len]

    frames = padded.reshape(n_frames, frame_samples)       # (n_frames, frame_samples)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))      # (n_frames,)

    # ── 2. Binary duck target per frame ─────────────────────────────────────
    target_db = np.where(frame_rms > duck_threshold_rms,
                         float(duck_amount_db), 0.0).astype(np.float64)

    # ── 3. Lookahead shift — roll envelope earlier in time ──────────────────
    #   np.roll wraps; we overwrite the wrapped tail with 0.0 (no duck).
    if lookahead_frames > 0:
        target_db = np.roll(target_db, -lookahead_frames)
        target_db[-lookahead_frames:] = 0.0

    # ── 4. Vectorised attack / release EMA smoothing ────────────────────────
    attack_alpha = math.exp(-frame_ms / attack_ms)
    release_alpha = math.exp(-frame_ms / release_ms)

    smoothed = np.zeros(n_frames, dtype=np.float64)
    current = 0.0
    for i in range(n_frames):
        t = target_db[i]
        if t < current:                              # attacking (ducking deeper)
            current = attack_alpha * current + (1.0 - attack_alpha) * t
        else:                                        # releasing (recovering)
            current = release_alpha * current + (1.0 - release_alpha) * t
        smoothed[i] = current

    # ── 5. Upsample frame envelope → sample resolution via linear interp ────
    frame_centers = np.arange(n_frames) * frame_samples + frame_samples / 2.0
    sample_indices = np.arange(total_samples, dtype=np.float64)
    gain_db_samples = np.interp(sample_indices, frame_centers, smoothed)

    # ── 6. Convert dB → linear and multiply ────────────────────────────────
    gain_linear = np.power(10.0, gain_db_samples / 20.0).astype(np.float32)
    return (music_audio[..., :total_samples] * gain_linear).astype(np.float32)


def apply_mask_ducking(
    voice_activity: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -9.0,
    attack_ms: float = 150.0,
    release_ms: float = 2000.0,
    lookahead_ms: float = 100.0,
) -> np.ndarray:
    """Apply ducking using a pre-computed voice activity boolean mask.

    Constructs a dB automation curve from the voice_activity mask, applies
    lookahead shift, smooths with exponential attack/release envelopes, and
    multiplies the music array by the resulting linear gain envelope.

    This is the preferred method when using Kokoro TTS (Method A from the
    architecture guide) because the mask is sample-accurate and requires
    no additional acoustic analysis.

    Args:
        voice_activity:  bool array, True where voice is active, aligned to music.
        music_audio:     1D float32 music array to duck.
        sample_rate:     Sample rate of both arrays.
        duck_amount_db:  dB reduction during speech (negative value, e.g. -9.0).
        attack_ms:       Time to reach full duck after voice onset.
        release_ms:      Time to recover after voice stops (long for meditation).
        lookahead_ms:    Shift duck earlier than voice onset (pre-duck).

    Returns:
        Ducked music as 1D float32 array.
    """
    total_samples = music_audio.shape[-1]

    # 1. Build raw dB automation from the boolean mask
    target_db = np.where(
        voice_activity[:total_samples],
        float(duck_amount_db),
        0.0
    ).astype(np.float64)

    # 2. Lookahead: shift envelope earlier so music starts fading before
    #    the first syllable of each phrase.
    lookahead_samples = int((lookahead_ms / 1000.0) * sample_rate)
    if lookahead_samples > 0:
        target_db = np.roll(target_db, -lookahead_samples)
        target_db[-lookahead_samples:] = 0.0

    # 3. Sample-by-sample EMA smoothing for attack and release.
    #    Attack: how fast music ducks when voice starts.
    #    Release: how slowly music recovers when voice stops.
    attack_coeff = math.exp(-1.0 / max(1, attack_ms * sample_rate / 1000.0))
    release_coeff = math.exp(-1.0 / max(1, release_ms * sample_rate / 1000.0))

    smoothed = np.zeros(total_samples, dtype=np.float64)
    current = 0.0
    for i in range(total_samples):
        t = target_db[i]
        if t < current:
            current = attack_coeff * current + (1.0 - attack_coeff) * t
        else:
            current = release_coeff * current + (1.0 - release_coeff) * t
        smoothed[i] = current

    # 4. Convert dB → linear and apply
    gain_linear = np.power(10.0, smoothed / 20.0).astype(np.float32)
    return (music_audio[..., :total_samples] * gain_linear).astype(np.float32)


# ---------------------------------------------------------------------------
# Track overlay / alignment
# ---------------------------------------------------------------------------

def _loop_with_crossfade(
    music: np.ndarray,
    target_length: int,
    sample_rate: int = SAMPLE_RATE,
    crossfade_sec: float = 2.0,
) -> np.ndarray:
    """Loop music array to target_length with equal-power crossfade at boundaries."""
    crossfade_samples = min(int(crossfade_sec * sample_rate), music.shape[-1] // 2)
    result = music.copy()

    while result.shape[-1] < target_length + crossfade_samples:
        # Equal-power cosine crossfade (replaces linear np.linspace)
        t = np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32)
        fo = np.cos(t) ** 2               # fade out: 1.0 → 0.0
        fi = np.cos(np.pi / 2 - t) ** 2   # fade in:  0.0 → 1.0

        overlap = result[..., -crossfade_samples:] * fo + music[..., :crossfade_samples] * fi
        result = np.concatenate([result[..., :-crossfade_samples], overlap, music[..., crossfade_samples:]], axis=-1)

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

    if voice.ndim == 1:
        pad_shape = pre_roll_samples
    else:
        pad_shape = (voice.shape[0], pre_roll_samples)
        
    # Prepend silence to voice so music plays alone briefly first
    aligned_voice = np.concatenate([
        np.zeros(pad_shape, dtype=np.float32),
        voice,
    ], axis=-1)

    total_length = aligned_voice.shape[-1]

    # Loop music with crossfade if shorter than needed
    if music.shape[-1] < total_length:
        music = _loop_with_crossfade(music, total_length, sample_rate)

    aligned_music = music[..., :total_length].astype(np.float32)

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
    if 0 < fade_in_samples < result.shape[-1]:
        result[..., :fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)

    fade_out_samples = int(fade_out_sec * sample_rate)
    if 0 < fade_out_samples < result.shape[-1]:
        result[..., -fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Loudness normalization
# ---------------------------------------------------------------------------

def calculate_loudness_gain(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -14.0,
) -> float:
    """Calculate the linear gain multiplier to hit a target LUFS."""
    min_samples = int(0.4 * sample_rate)
    if audio.shape[-1] < min_samples:
        return 1.0

    meter = pyln.Meter(sample_rate)
    audio_for_meter = audio.T if audio.ndim == 2 else audio
    try:
        loudness = meter.integrated_loudness(audio_for_meter)
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
    target_lufs: float = -14.0,
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
    duck_amount_db: float = -3.0,
    music_volume_db: float = -20.0,
    music_pre_roll_sec: float = 2.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """Full mix pipeline: align → level → duck → overlay → fades → normalize.

    Args:
        duck_amount_db: Additional dB reduction during speech on top of
            music_volume_db. -20 dB makes music nearly inaudible during
            narration — the primary meditation requirement.
        music_volume_db: Baseline music level in dB (applied before ducking).
            -11.7 dB keeps the music subtle even during pauses (+0.3 dB vs -12 dB).
            During speech it drops by an additional duck_amount_db → ~-31.7 dB total,
            which is essentially silence behind the voice.

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
    target_len = aligned_voice.shape[-1]
    if aligned_music.shape[-1] < target_len:
        if aligned_music.ndim == 1:
            pad_shape = target_len - aligned_music.shape[-1]
        else:
            pad_shape = (aligned_music.shape[0], target_len - aligned_music.shape[-1])

        aligned_music = np.concatenate([
            aligned_music,
            np.zeros(pad_shape, dtype=np.float32),
        ], axis=-1)
    aligned_music = aligned_music[..., :target_len]

    # 4. Apply mask-based ducking (Method A — uses pre-computed voice activity)
    # attack_ms=500: music fades gradually over 500ms — feels like a natural breath before speech
    # release_ms=5000: music rises very slowly after speech ends (5 seconds to recover)
    #   — this is the key parameter that gives the "very slowly raises" meditation feel
    # lookahead_ms=350: music starts fading 350ms before voice onset so the transition
    #   is already underway when the first syllable arrives — no abrupt floor drop
    ducked_music = apply_mask_ducking(
        aligned_activity,
        aligned_music,
        sample_rate=sample_rate,
        duck_amount_db=duck_amount_db,
        attack_ms=500.0,
        release_ms=1200.0,
        lookahead_ms=350.0,
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

    tensor = torch.tensor(audio, dtype=torch.float32)
    is_1d = tensor.ndim == 1
    if is_1d:
        tensor = tensor.unsqueeze(0)
        
    resampled = torchaudio.functional.resample(
        tensor, source_rate, target_rate,
        lowpass_filter_width=64,
        rolloff=0.9475,
    )
    
    if is_1d:
        result = resampled.squeeze(0).numpy()
    else:
        result = resampled.numpy()
        
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

    # sf.write requires (samples, channels)
    sf.write(voice_path, voice_audio.T if voice_audio.ndim == 2 else voice_audio, sample_rate, subtype="PCM_24")
    sf.write(music_path, music_audio.T if music_audio.ndim == 2 else music_audio, sample_rate, subtype="PCM_24")

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
    target_lufs: float = -14.0,
) -> str:
    """Stream audio out to temp file with Pedalboard plugins and normalization.
    
    Reads from the memory array in bounded chunks, applies a calculated linear 
    gain to reach target_lufs, applies the final Pedalboard master chain, and
    streams directly to a file to prevent immense memory spikes.

    Args:
        target_lufs: Loudness target in LUFS (-14 for streaming distribution).
    """
    from pedalboard.io import AudioFile
    
    export_rate = target_sample_rate
    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    # Pre-calculate the linear gain to bring the whole file to target LUFS
    mix_lufs_gain = calculate_loudness_gain(audio, sample_rate, target_lufs)
    
    # 20 second chunks
    chunk_samples = int(20.0 * sample_rate)
    total_samples = audio.shape[-1]
    num_channels = audio.shape[0] if audio.ndim == 2 else 1

    # Initialize Pedalboard AudioFile struct
    f = AudioFile(tmp_path, "w", samplerate=export_rate, num_channels=num_channels)

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[..., start:end]
        
        # Resample chunk if needed
        if sample_rate != export_rate:
            chunk = resample_for_export(chunk, sample_rate, export_rate)
            
        # Apply normalization gain
        chunk = chunk * mix_lufs_gain
        
        # Apply Pedalboard mastering chain if present
        if master_chain:
            chunk_2d = chunk.reshape(1, -1) if chunk.ndim == 1 else chunk
            chunk = master_chain(chunk_2d, export_rate)
            if audio.ndim == 1:
                chunk = chunk.squeeze(0)
            
        # write directly as float32. Pedalboard expects (channels, samples).
        if chunk.ndim == 1:
            f.write(chunk.reshape(1, -1).astype(np.float32))
        else:
            f.write(chunk.astype(np.float32))

    f.close()
    return tmp_path
