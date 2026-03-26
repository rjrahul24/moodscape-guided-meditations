"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
import math
from scipy.ndimage import maximum_filter1d
from scipy.signal import sosfiltfilt, butter
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
    attack_ms: float = 80.0,
    release_ms: float = 1000.0,
    lookahead_ms: float = 20.0,
    window_ms: float = 50.0,
    hold_ms: float = 1200.0,
) -> np.ndarray:
    """Sidechain ducker using a smooth RMS envelope follower with asymmetric A/R.

    Meditation-optimised: the hold parameter keeps music ducked across phrase
    gaps so it only rises during extended pauses (>1-2 s), preventing the
    per-sentence pumping that breaks a calm atmosphere.

    Args:
        voice_audio:    1D float32 voice array.
        music_audio:    1D float32 music array (must be >= voice_audio length).
        sample_rate:    Sample rate (Hz).
        duck_amount_db: Target attenuation depth (e.g. -10 dB).
        threshold_db:   Voice level (dBFS) above which ducking triggers.
        attack_ms:      Attack time constant (ms). 80ms = smooth, breath-like.
        release_ms:     Release time constant (ms). 1000ms = gentle recovery.
        lookahead_ms:   Time (ms) to shift the envelope forward.
        window_ms:      RMS analysis window (ms).
        hold_ms:        Hold time (ms) to bridge inter-phrase gaps. Keeps music
                        ducked for this duration after voice goes silent,
                        preventing pumping between sentences.

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

    # 3. Apply hold time — extend voice-active regions forward to bridge
    #    inter-phrase gaps.  Uses scipy maximum_filter1d for efficiency:
    #    each sample becomes the max of itself and the next hold_samples,
    #    keeping the "voice active" signal high across short pauses.
    hold_samples = int(hold_ms * sample_rate / 1000.0)
    if hold_samples > 1:
        # Voice is "active" where db_env > threshold_db.  We dilate this
        # forward in time by hold_samples so that the release phase doesn't
        # begin until hold_ms after the last voiced sample.
        voice_active = (db_env > threshold_db).astype(np.float32)
        voice_active_held = maximum_filter1d(voice_active, size=hold_samples,
                                             origin=-(hold_samples // 2))
        # Where voice_active_held is 1.0, force db_env above threshold so
        # the proportional gain logic below still triggers full ducking.
        db_env = np.where(voice_active_held > 0.5,
                          np.maximum(db_env, threshold_db + 1.0),
                          db_env)

    # 4. Calculate target gain reduction in dB
    # Gain reduction only applies when db_env > threshold_db
    # Proportionally scale reduction up to duck_amount_db
    target_reduction_db = np.clip(threshold_db - db_env, duck_amount_db, 0.0)
    target_gain_linear = 10.0 ** (target_reduction_db / 20.0)

    # 5. Asymmetric Smoothing (EMA)
    attack_alpha = math.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_alpha = math.exp(-1.0 / (release_ms * sample_rate / 1000.0))

    smoothed_gain = np.ones(total_samples, dtype=np.float64)
    current = 1.0
    for i in range(total_samples):
        target = target_gain_linear[i]
        if target < current:  # Attack phase (gain dropping)
            current = attack_alpha * current + (1.0 - attack_alpha) * target
        else:  # Release phase (gain rising)
            current = release_alpha * current + (1.0 - release_alpha) * target
        smoothed_gain[i] = current

    # 6. Lookahead shift
    lookahead_samples = int((lookahead_ms / 1000.0) * sample_rate)
    if lookahead_samples > 0:
        smoothed_gain = np.roll(smoothed_gain, -lookahead_samples)
        smoothed_gain[-lookahead_samples:] = 1.0

    return (music_audio * smoothed_gain).astype(np.float32)


def _compute_ducking_gain(
    voice_audio: np.ndarray,
    total_samples: int,
    sample_rate: int,
    duck_amount_db: float,
    threshold_db: float,
    attack_ms: float,
    release_ms: float,
    lookahead_ms: float,
    window_ms: float,
    hold_ms: float,
) -> np.ndarray:
    """Compute the sample-rate ducking gain curve (shared by fullband and multiband).

    Returns a float64 gain array of length ``total_samples`` where 1.0 means no
    ducking and values < 1.0 represent attenuation.
    """
    voice = np.zeros(total_samples, dtype=np.float32)
    v_len = min(len(voice_audio), total_samples)
    voice[:v_len] = voice_audio[:v_len]

    window_samples = int((window_ms / 1000.0) * sample_rate)
    win = np.ones(window_samples) / window_samples
    ms_env = np.convolve(voice ** 2, win, mode="same")
    rms_env = np.sqrt(np.maximum(ms_env, 1e-10))
    db_env = 20.0 * np.log10(rms_env)

    hold_samples = int(hold_ms * sample_rate / 1000.0)
    if hold_samples > 1:
        voice_active = (db_env > threshold_db).astype(np.float32)
        voice_active_held = maximum_filter1d(
            voice_active, size=hold_samples, origin=-(hold_samples // 2)
        )
        db_env = np.where(
            voice_active_held > 0.5,
            np.maximum(db_env, threshold_db + 1.0),
            db_env,
        )

    target_reduction_db = np.clip(threshold_db - db_env, duck_amount_db, 0.0)
    target_gain_linear = 10.0 ** (target_reduction_db / 20.0)

    attack_alpha = math.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_alpha = math.exp(-1.0 / (release_ms * sample_rate / 1000.0))

    smoothed = np.ones(total_samples, dtype=np.float64)
    current = 1.0
    for i in range(total_samples):
        target = target_gain_linear[i]
        if target < current:
            current = attack_alpha * current + (1.0 - attack_alpha) * target
        else:
            current = release_alpha * current + (1.0 - release_alpha) * target
        smoothed[i] = current

    lookahead_samples = int((lookahead_ms / 1000.0) * sample_rate)
    if lookahead_samples > 0:
        smoothed = np.roll(smoothed, -lookahead_samples)
        smoothed[-lookahead_samples:] = 1.0

    return smoothed


def apply_multiband_ducking(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -10.0,
    threshold_db: float = -35.0,
    attack_ms: float = 80.0,
    release_ms: float = 1000.0,
    lookahead_ms: float = 20.0,
    window_ms: float = 50.0,
    hold_ms: float = 1200.0,
    low_crossover_hz: float = 250.0,
    high_crossover_hz: float = 4000.0,
    low_duck_ratio: float = 0.25,
    high_duck_ratio: float = 0.50,
) -> np.ndarray:
    """Frequency-selective sidechain ducking preserving bass warmth and HF shimmer.

    Splits music into three bands using Linkwitz-Riley (4th-order Butterworth
    applied forward+backward) crossover filters.  Each band is ducked by a
    different amount so that only the mid-range where voice lives (250-4000 Hz)
    receives full ducking, while bass warmth and high-frequency shimmer (singing
    bowls, ambient texture) are preserved.

    Args:
        low_crossover_hz:  Low/mid crossover frequency (Hz).
        high_crossover_hz: Mid/high crossover frequency (Hz).
        low_duck_ratio:    Fraction of duck_amount_db applied to low band.
                           0.25 → -3 dB when main duck is -12 dB.
        high_duck_ratio:   Fraction of duck_amount_db applied to high band.
                           0.50 → -6 dB when main duck is -12 dB.
        (other args):      Same as apply_envelope_ducking().

    Returns:
        Ducked music array (float32).
    """
    total_samples = music_audio.shape[-1]

    # 1. Compute the fullband gain curve once from the voice signal
    gain_full = _compute_ducking_gain(
        voice_audio, total_samples, sample_rate,
        duck_amount_db, threshold_db, attack_ms, release_ms,
        lookahead_ms, window_ms, hold_ms,
    )

    # 2. Derive per-band gain curves by scaling the attenuation depth.
    #    gain_full is in [duck_linear .. 1.0].  We rescale the *reduction*
    #    portion: reduction = 1.0 - gain_full, then apply band ratios.
    reduction = 1.0 - gain_full  # 0.0 = no duck, positive = ducked
    gain_low = (1.0 - reduction * low_duck_ratio).astype(np.float32)
    gain_mid = gain_full.astype(np.float32)  # full ducking
    gain_high = (1.0 - reduction * high_duck_ratio).astype(np.float32)

    # 3. Split music into 3 bands using Linkwitz-Riley crossovers.
    #    LR4 = 2nd-order Butterworth applied twice (sosfiltfilt = forward+backward).
    nyquist = sample_rate / 2.0
    # Guard crossover frequencies against Nyquist
    low_xo = min(low_crossover_hz, nyquist * 0.9)
    high_xo = min(high_crossover_hz, nyquist * 0.9)

    sos_lp_low = butter(2, low_xo / nyquist, btype="low", output="sos")
    sos_hp_low = butter(2, low_xo / nyquist, btype="high", output="sos")
    sos_lp_high = butter(2, high_xo / nyquist, btype="low", output="sos")
    sos_hp_high = butter(2, high_xo / nyquist, btype="high", output="sos")

    music_f32 = music_audio.astype(np.float64)
    low_band = sosfiltfilt(sos_lp_low, music_f32).astype(np.float32)
    high_band = sosfiltfilt(sos_hp_high, music_f32).astype(np.float32)
    # Mid = full signal minus low and high to guarantee perfect reconstruction
    mid_band = (music_audio - low_band - high_band).astype(np.float32)

    # 4. Apply per-band ducking
    ducked_low = low_band * gain_low
    ducked_mid = mid_band * gain_mid
    ducked_high = high_band * gain_high

    return (ducked_low + ducked_mid + ducked_high).astype(np.float32)


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

def _exponential_curve(n_samples: int, rising: bool, steepness: float = 4.0) -> np.ndarray:
    """Generate an exponential fade curve.

    For fade-in (rising=True): slow start, accelerating towards unity — feels
    like music gently emerging.  For fade-out (rising=False): slow initial
    descent then faster drop to silence — the natural tail of a meditation.

    Uses ``(exp(k*t) - 1) / (exp(k) - 1)`` which maps [0,1] → [0,1] with
    adjustable steepness *k*.  Matches professional DAW exponential curves.
    """
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    if steepness == 0.0:
        curve = t
    else:
        curve = (np.exp(steepness * t) - 1.0) / (np.exp(steepness) - 1.0)
    if not rising:
        curve = curve[::-1]
    return curve.astype(np.float32)


def apply_fades(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 5.0,
    curve: str = "exponential",
) -> np.ndarray:
    """Apply fade-in and fade-out to audio.

    Args:
        curve: Fade shape — "exponential" (default, natural for meditation),
               "linear" (legacy), or "cosine" (equal-power).
    """
    result = audio.copy()

    fade_in_samples = int(fade_in_sec * sample_rate)
    if 0 < fade_in_samples < result.shape[-1]:
        if curve == "exponential":
            fade_in = _exponential_curve(fade_in_samples, rising=True)
        elif curve == "cosine":
            fade_in = ((1.0 - np.cos(np.linspace(0.0, np.pi, fade_in_samples))) / 2.0).astype(np.float32)
        else:  # linear
            fade_in = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        result[..., :fade_in_samples] *= fade_in

    fade_out_samples = int(fade_out_sec * sample_rate)
    if 0 < fade_out_samples < result.shape[-1]:
        if curve == "exponential":
            fade_out = _exponential_curve(fade_out_samples, rising=False)
        elif curve == "cosine":
            fade_out = ((1.0 + np.cos(np.linspace(0.0, np.pi, fade_out_samples))) / 2.0).astype(np.float32)
        else:  # linear
            fade_out = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        result[..., -fade_out_samples:] *= fade_out

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
    music_pre_roll_sec: float = 8.0,
    music_post_roll_sec: float = 15.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 8.0,
    target_lufs: float = -19.0,
    stereo_output: bool = False,
    multiband: bool = True,
) -> np.ndarray:
    """Full mix pipeline: align → level → duck → overlay → fades → normalize.

    Args:
        duck_amount_db: Additional dB reduction during speech on top of
            music_volume_db. -12 dB is the recommended default for meditation.
        music_volume_db: Baseline music level in dB (applied before ducking).
            -17.0 dB keeps the music present and audible during pauses without
            competing with silence.
        music_pre_roll_sec: Music plays alone for this many seconds before
            the voice begins (intro). Default 8.0s — gives listeners time to
            settle, put on headphones, and close their eyes.
        music_post_roll_sec: Music plays alone for this many seconds after
            the voice ends (outro), before the fade-out. Default 15.0s.
        fade_out_sec: Fade-out duration. Default 8.0s — longer tail for a
            natural, calming exit.
        multiband: If True (default), use frequency-selective ducking that
            preserves bass warmth and high-frequency shimmer. Only the mid-range
            (250-4000 Hz) where voice conflicts exist receives full ducking.

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

    # 4. Apply sidechain ducking
    # Meditation-tuned parameters:
    #   attack_ms=80:     smooth, breath-like music fade at voice onset
    #   release_ms=1000:  music returns gently over ~1s, feeling calming
    #   hold_ms=1200:     bridges inter-phrase gaps — prevents per-sentence pumping
    #   lookahead_ms=60:  60ms pre-duck gives a smooth lead-in before first syllable
    _duck_kwargs = dict(
        sample_rate=sample_rate,
        duck_amount_db=duck_amount_db,
        threshold_db=-35.0,
        attack_ms=80.0,
        release_ms=1000.0,
        lookahead_ms=60.0,
        hold_ms=1200.0,
    )

    if multiband:
        ducked_music = apply_multiband_ducking(
            aligned_voice, aligned_music, **_duck_kwargs,
        )
    else:
        ducked_music = apply_envelope_ducking(
            aligned_voice, aligned_music, **_duck_kwargs,
        )

    # 5. Stereo upmix (opt-in): Haas effect on music, center-pan voice
    if stereo_output:
        from core.stereo_upmix import haas_stereo, center_pan_voice
        ducked_music = haas_stereo(ducked_music, sample_rate)       # (2, N)
        aligned_voice = center_pan_voice(aligned_voice)             # (2, N)

    # 6. Sum voice + ducked music
    mixed = aligned_voice + ducked_music

    # 7. Apply fades (exponential curves for natural meditation feel)
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
        
        # Ensure true peak safety -1.5 dBTP via clipping (limiter already ran during mastering).
        # -1.5 dBTP provides safety margin for lossy codec encoding (AAC/MP3).
        chunk = np.clip(chunk, -0.841, 0.841)  # -1.5 dBFS = 10^(-1.5/20) ≈ 0.841
            
        # write directly as float32. Pedalboard expects (channels, samples).
        if chunk.ndim == 1:
            f.write(chunk.reshape(1, -1).astype(np.float32))
        else:
            f.write(chunk.astype(np.float32))

    f.close()
    return tmp_path
