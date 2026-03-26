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


def apply_envelope_ducking(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -10.0,
    threshold_db: float = -35.0,
    attack_ms: float = 10.0,
    release_ms: float = 800.0,
    lookahead_ms: float = 20.0,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Sidechain ducker using a smooth RMS envelope follower with asymmetric A/R.

    Args:
        voice_audio:    1D float32 voice array.
        music_audio:    1D float32 music array (must be >= voice_audio length).
        sample_rate:    Sample rate (Hz).
        duck_amount_db: Target attenuation depth (e.g. -10 dB).
        threshold_db:   Voice level (dBFS) above which ducking triggers.
        attack_ms:      Attack time constant (ms).
        release_ms:     Release time constant (ms).
        lookahead_ms:   Time (ms) to shift the envelope forward.
        window_ms:      RMS analysis window (ms).

    Returns:
        Ducked music array.
    """
    total_samples = music_audio.shape[-1]
    voice = np.zeros(total_samples, dtype=np.float32)
    v_len = min(len(voice_audio), total_samples)
    voice[:v_len] = voice_audio[:v_len]

    # 1. Calculate squared signal and smooth with rolling window for Mean Square
    window_samples = int((window_ms / 1000.0) * sample_rate)
    window = np.ones(window_samples) / window_samples
    ms_env = np.convolve(voice**2, window, mode='same')
    
    # 2. Convert to RMS and then dBFS (clip to avoid log of zero)
    rms_env = np.sqrt(np.maximum(ms_env, 1e-10))
    db_env = 20.0 * np.log10(rms_env)

    # 3. Calculate target gain reduction in dB
    # Gain reduction only applies when db_env > threshold_db
    # Proportionally scale reduction up to duck_amount_db
    # Simple knee logic: reduction = min(0, (threshold_db - db_env)) 
    # then clamped to duck_amount_db
    target_reduction_db = np.clip(threshold_db - db_env, duck_amount_db, 0.0)
    target_gain_linear = 10.0 ** (target_reduction_db / 20.0)

    # 4. Asymmetric Smoothing (EMA)
    # y[n] = alpha * y[n-1] + (1 - alpha) * x[n]
    attack_alpha = math.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_alpha = math.exp(-1.0 / (release_ms * sample_rate / 1000.0))
    
    smoothed_gain = np.ones(total_samples, dtype=np.float64)
    current = 1.0
    for i in range(total_samples):
        target = target_gain_linear[i]
        if target < current: # Attack phase (gain dropping)
            current = attack_alpha * current + (1.0 - attack_alpha) * target
        else: # Release phase (gain rising)
            current = release_alpha * current + (1.0 - release_alpha) * target
        smoothed_gain[i] = current

    # 5. Lookahead shift
    lookahead_samples = int((lookahead_ms / 1000.0) * sample_rate)
    if lookahead_samples > 0:
        smoothed_gain = np.roll(smoothed_gain, -lookahead_samples)
        smoothed_gain[-lookahead_samples:] = 1.0

    return (music_audio * smoothed_gain).astype(np.float32)


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
    music_pre_roll_sec: float = 4.0,
    music_post_roll_sec: float = 8.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Align voice and music with professional intro/outro structure.

    Music starts ``pre_roll_sec`` before the voice, and continues playing
    ``post_roll_sec`` after the voice ends — giving the meditation a graceful
    intro and outro instead of cutting off with the last word.

    Returns (aligned_voice, aligned_music) — both same length, ready to sum.
    """
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    post_roll_samples = int(music_post_roll_sec * sample_rate)

    if voice.ndim == 1:
        pre_pad = pre_roll_samples
        post_pad = post_roll_samples
    else:
        pre_pad = (voice.shape[0], pre_roll_samples)
        post_pad = (voice.shape[0], post_roll_samples)

    # Prepend silence (music plays alone) and append silence (music outro)
    aligned_voice = np.concatenate([
        np.zeros(pre_pad, dtype=np.float32),
        voice,
        np.zeros(post_pad, dtype=np.float32),
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
    target_lufs: float = -18.0,
) -> float:
    """Calculate the linear gain multiplier to hit a target LUFS.
    
    Professional meditation narration targets ~ -18.0 LUFS integrated.
    For mono archives, we target -21.0 LUFS (mono measures ~3 dB lower 
     than stereo for same perceived loudness).
    """
    meter = pyln.Meter(sample_rate)
    
    # Adjust target for mono playback architectures if the source is mono
    num_channels = audio.shape[0] if audio.ndim == 2 else 1
    if num_channels == 1 and target_lufs == -18.0:
        actual_target = -21.0
    else:
        actual_target = target_lufs

    min_samples = int(0.4 * sample_rate)
    if audio.shape[-1] < min_samples:
        return 1.0

    audio_for_meter = audio.T if audio.ndim == 2 else audio
    try:
        loudness = meter.integrated_loudness(audio_for_meter)
    except Exception:
        return 1.0

    if not np.isfinite(loudness):
        return 1.0

    if abs(actual_target - loudness) > 40.0:
        return 1.0

    gain_db = actual_target - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)
    return float(gain_linear)

def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = -18.0,
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
    music_volume_db: float = -17.0,
    music_pre_roll_sec: float = 4.0,
    music_post_roll_sec: float = 8.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 6.0,
    target_lufs: float = -19.0,
    stereo_output: bool = False,
) -> np.ndarray:
    """Full mix pipeline: align → level → duck → overlay → fades → normalize.

    Args:
        duck_amount_db: Additional dB reduction during speech on top of
            music_volume_db. -20 dB makes music nearly inaudible during
            narration — the primary meditation requirement.
        music_volume_db: Baseline music level in dB (applied before ducking).
            -17.0 dB keeps the music present and audible during pauses without
            competing with silence. During speech it drops by an additional
            duck_amount_db → -38 dB total, nearly inaudible behind the voice.
        music_pre_roll_sec: Music plays alone for this many seconds before
            the voice begins (intro). Default 4.0s.
        music_post_roll_sec: Music plays alone for this many seconds after
            the voice ends (outro), before the fade-out. Default 8.0s.

    Called after voice FX and music FX have already been applied.
    Returns mixed float32 array (mono or stereo) ready for master chain + export.
    """
    # 1. Align voice and music with pre-roll and post-roll
    aligned_voice, aligned_music = overlay_tracks(
        voice_audio, music_audio, music_pre_roll_sec, music_post_roll_sec,
        sample_rate,
    )

    # 2. Set baseline music level so it sits behind the voice
    music_gain = np.float32(10.0 ** (music_volume_db / 20.0))
    aligned_music = aligned_music * music_gain

    # 3. Extend voice_activity to match pre-roll + post-roll offsets
    pre_roll_samples = int(music_pre_roll_sec * sample_rate)
    post_roll_samples = int(music_post_roll_sec * sample_rate)
    aligned_activity = np.concatenate([
        np.zeros(pre_roll_samples, dtype=bool),
        voice_activity,
        np.zeros(post_roll_samples, dtype=bool),
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

    # 4. Apply envelope-follower sidechain ducking
    # Meditation defaults:
    # attack_ms=40.0: gradual, breath-like music fade at voice onset (10ms was
    #   too snappy — sounded mechanical and introduced click artifacts).
    # release_ms=800.0: music returns gently, feeling calming.
    # lookahead_ms=60.0: 60ms pre-duck gives a smooth lead-in before first syllable.
    # duck_amount_db: -12 dB default (configurable via duck_amount_db argument).

    ducked_music = apply_envelope_ducking(
        aligned_voice,
        aligned_music,
        sample_rate=sample_rate,
        duck_amount_db=duck_amount_db,
        threshold_db=-35.0,
        attack_ms=40.0,
        release_ms=800.0,
        lookahead_ms=60.0,
    )

    # 5. Stereo upmix (opt-in): Haas effect on music, center-pan voice
    if stereo_output:
        from core.stereo_upmix import haas_stereo, center_pan_voice
        ducked_music = haas_stereo(ducked_music, sample_rate)       # (2, N)
        aligned_voice = center_pan_voice(aligned_voice)             # (2, N)

    # 6. Sum voice + ducked music
    mixed = aligned_voice + ducked_music

    # 7. Apply fades
    mixed = apply_fades(mixed, sample_rate, fade_in_sec, fade_out_sec)

    # 8. Do not normalize loudness here, as we will chunk-stream the mix through
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
    target_lufs: float = -19.0,
) -> str:
    """Stream audio out to temp file with Pedalboard plugins and normalization.
    
    Reads from the memory array in bounded chunks, applies a calculated linear 
    gain to reach target_lufs, applies the final Pedalboard master chain, and
    streams directly to a file to prevent immense memory spikes.

    Args:
        target_lufs: Loudness target in LUFS (-18 for meditation).
    """
    from pedalboard.io import AudioFile
    
    export_rate = target_sample_rate
    suffix = f".{output_format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    # 1. Apply Pedalboard mastering chain if present to the whole array first.
    # We do this in-memory (safe for 32GB RAM target hardware).
    mastered_audio = audio
    if master_chain:
        audio_2d = audio.reshape(1, -1) if audio.ndim == 1 else audio
        mastered_audio = master_chain(audio_2d, sample_rate)
        if audio.ndim == 1:
            mastered_audio = mastered_audio.squeeze(0)

    # 2. Pre-calculate the linear gain based on the PRE-normalized mastered audio
    # to bring the whole file to target LUFS accurately.
    mix_lufs_gain = calculate_loudness_gain(mastered_audio, sample_rate, target_lufs)
    
    # 20 second chunks for streaming export
    chunk_samples = int(20.0 * sample_rate)
    total_samples = mastered_audio.shape[-1]
    num_channels = mastered_audio.shape[0] if mastered_audio.ndim == 2 else 1

    # Initialize Pedalboard AudioFile struct
    f = AudioFile(tmp_path, "w", samplerate=export_rate, num_channels=num_channels)

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = mastered_audio[..., start:end]
        
        # Resample chunk if needed
        if sample_rate != export_rate:
            chunk = resample_for_export(chunk, sample_rate, export_rate)
            
        # Apply normalization gain
        chunk = chunk * mix_lufs_gain
        
        # Ensure true peak safety -1.0 dBFS via clipping (limiter already ran during mastering)
        chunk = np.clip(chunk, -0.891, 0.891)  # -1.0 dBFS = 10^(-1/20) ≈ 0.891
            
        # write directly as float32. Pedalboard expects (channels, samples).
        if chunk.ndim == 1:
            f.write(chunk.reshape(1, -1).astype(np.float32))
        else:
            f.write(chunk.astype(np.float32))

    f.close()
    return tmp_path
