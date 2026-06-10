"""Mixing engine: ducking, overlay, fades, normalization, export — MoodScape."""

import tempfile

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from scipy.ndimage import minimum_filter1d
from scipy.signal import sosfiltfilt, sosfilt, butter, lfilter, resample_poly
from pedalboard import Pedalboard

SAMPLE_RATE = 24000

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
# Breathing sidechain duck (deep, gradual, script/VAD-aware)
# ---------------------------------------------------------------------------
#
# Replaces the old multiband reactive ducker. Goal (per user spec): as speech
# starts the bed falls *gradually*; during speech it sits *very low* (but not
# inaudible); during pauses it rises *gradually* back up and "breathes". The
# curve is built deterministically from detected phrase boundaries (predictive
# S-curve descent, deep hold, S-curve release, +lift on long pauses) and is
# combined with a reactive safety net so off-script breaths still duck.
#
# Adapted from the reference meditation_mixer (moodscape-mix-lib), made fully
# vectorized (no per-sample Python loops) and applied fullband so the whole bed
# drops — the reference ducked only the mid band, which read as "flat".


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic Hermite S-curve, x in [0,1] -> [0,1] with zero slope at the ends."""
    x = np.clip(x, 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _onepole_rms_env(x: np.ndarray, sample_rate: int, ms: float = 30.0) -> np.ndarray:
    """One-pole RMS envelope (vectorized via lfilter)."""
    alpha = float(np.exp(-1.0 / (sample_rate * ms / 1000.0)))
    sq = np.asarray(x, dtype=np.float64) ** 2
    ms_env = lfilter([1.0 - alpha], [1.0, -alpha], sq)
    return np.sqrt(np.maximum(ms_env, 1e-12)).astype(np.float32)


def _zero_phase_smooth(curve: np.ndarray, sample_rate: int, hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass to round keyframe corners."""
    if hz <= 0 or curve.shape[0] <= 12:
        return curve.astype(np.float32)
    sos = butter(2, float(hz), btype="low", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, curve).astype(np.float32)


def adaptive_vad_threshold(
    voice_audio: np.ndarray,
    sample_rate: int,
    offset_db: float = -22.0,
    clamp_db: tuple[float, float] = (-55.0, -35.0),
    fallback_db: float = -40.0,
    env_ms: float = 30.0,
) -> float:
    """Derive a VAD threshold from the voice's own RMS-envelope distribution.

    The fixed -40 dB threshold under/over-detects when a TTS engine (or a
    re-balanced stem) lands hotter or quieter than expected. Instead, anchor
    the threshold to the speech level itself: the 95th percentile of the
    envelope approximates sustained speech, and ``offset_db`` below that is
    where phrase boundaries live. ``offset_db=-22`` reproduces the legacy
    -40 dB on a voice normalized to the pipeline's pre-mix level.

    Falls back to ``fallback_db`` when the distribution is degenerate
    (audio shorter than 2 s, or speech-to-floor spread under 12 dB).
    """
    if voice_audio.shape[-1] < int(2.0 * sample_rate):
        return fallback_db
    env = _onepole_rms_env(voice_audio, sample_rate, ms=env_ms)
    env_db = 20.0 * np.log10(env + 1e-9)
    speech_db = float(np.percentile(env_db, 95))
    floor_db = float(np.percentile(env_db, 10))
    if not np.isfinite(speech_db) or speech_db - floor_db < 12.0:
        return fallback_db
    threshold = max(speech_db + offset_db, floor_db + 8.0)
    return float(np.clip(threshold, clamp_db[0], clamp_db[1]))


def detect_phrases(
    voice_audio: np.ndarray,
    sample_rate: int,
    threshold_db: float | None = -40.0,
    env_ms: float = 30.0,
    min_phrase_ms: float = 150.0,
    merge_gap_ms: float = 250.0,
) -> list[tuple[float, float]]:
    """Voice-activity detection by RMS-envelope thresholding.

    Returns (start_s, end_s) phrases. Inter-word gaps shorter than
    ``merge_gap_ms`` are merged (so the bed doesn't pump between words);
    phrases shorter than ``min_phrase_ms`` are dropped as breaths/clicks.
    ``threshold_db=None`` derives the threshold from the voice itself via
    ``adaptive_vad_threshold``.
    """
    if voice_audio.size == 0:
        return []
    if threshold_db is None:
        threshold_db = adaptive_vad_threshold(voice_audio, sample_rate, env_ms=env_ms)
    env = _onepole_rms_env(voice_audio, sample_rate, ms=env_ms)
    env_db = 20.0 * np.log10(env + 1e-9)
    is_voice = env_db > threshold_db

    diffs = np.diff(is_voice.astype(np.int8))
    starts = (np.where(diffs > 0)[0] + 1).tolist()
    ends = (np.where(diffs < 0)[0] + 1).tolist()
    if is_voice[0]:
        starts.insert(0, 0)
    if is_voice[-1]:
        ends.append(int(is_voice.size))
    if not starts or not ends:
        return []

    phrases = list(zip(starts, ends))
    merge_n = int(sample_rate * merge_gap_ms / 1000.0)
    merged: list[tuple[int, int]] = [phrases[0]]
    for s, e in phrases[1:]:
        ps, pe = merged[-1]
        if s - pe < merge_n:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    min_n = int(sample_rate * min_phrase_ms / 1000.0)
    merged = [(s, e) for s, e in merged if (e - s) >= min_n]
    return [(s / sample_rate, e / sample_rate) for s, e in merged]


def _short_term_lufs_curve(
    audio: np.ndarray,
    sample_rate: int,
    win_s: float = 3.0,
    hop_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(window_center_times_s, lufs) short-term loudness curve (3 s windows)."""
    meter = pyln.Meter(sample_rate, block_size=0.400)
    n_win = int(win_s * sample_rate)
    n_hop = int(hop_s * sample_rate)
    times: list[float] = []
    vals: list[float] = []
    for start in range(0, max(0, audio.shape[-1] - n_win) + 1, n_hop):
        seg = audio[..., start:start + n_win]
        if seg.shape[-1] < n_win:
            break
        try:
            loud = meter.integrated_loudness(seg.T if seg.ndim == 2 else seg)
        except Exception:
            loud = float("-inf")
        times.append(start / sample_rate + win_s / 2.0)
        vals.append(loud)
    return np.asarray(times), np.asarray(vals)


def calibrate_music_bed(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int,
    phrases: list[tuple[float, float]] | None = None,
    speech_offset_lu: float = 30.5,
    pause_offset_lu: float = 14.5,
    volume_clamp_db: tuple[float, float] = (-24.0, -8.0),
    duck_clamp_db: tuple[float, float] = (-20.0, -10.0),
    win_s: float = 3.0,
) -> tuple[float, float]:
    """Measure post-FX stems and return ``(music_volume_db, duck_amount_db)``.

    Automates the bed level instead of trusting fixed constants: the bed gain
    is set so the raw music's short-term loudness sits ``pause_offset_lu``
    below the voice's speech-region loudness, and the duck depth covers the
    remaining ``speech_offset_lu`` separation. The default offsets were
    measured from the reference F5 + uploaded-instrumental mix at the legacy
    constants (-16 dB bed, -16 dB duck), so a session at today's nominal
    levels calibrates back to (-16, -16) and only off-nominal material
    (a hot or whisper-quiet upload, an unusual voice level) gets corrected.

    Degenerate inputs (no phrases, silent/short stems) return the legacy
    ``(-16.0, -16.0)``.
    """
    legacy = (-16.0, -16.0)
    if voice_audio.shape[-1] < int(2 * win_s * sample_rate):
        return legacy
    if music_audio.shape[-1] < int(2 * win_s * sample_rate):
        return legacy

    if phrases is None:
        phrases = detect_phrases(voice_audio, sample_rate, threshold_db=None)
    if not phrases:
        return legacy

    t_v, l_v = _short_term_lufs_curve(voice_audio, sample_rate, win_s=win_s)
    if t_v.size == 0:
        return legacy

    # Speech windows: >=50% of the window overlaps a detected phrase.
    speech_mask = np.zeros(t_v.size, dtype=bool)
    for i, t in enumerate(t_v):
        w0, w1 = t - win_s / 2.0, t + win_s / 2.0
        overlap = sum(max(0.0, min(w1, e) - max(w0, s)) for s, e in phrases)
        speech_mask[i] = overlap >= 0.5 * win_s
    finite_v = np.isfinite(l_v)
    if not (speech_mask & finite_v).any():
        return legacy
    voice_lufs = float(np.median(l_v[speech_mask & finite_v]))

    _, l_m = _short_term_lufs_curve(music_audio, sample_rate, win_s=win_s)
    finite_m = np.isfinite(l_m)
    if not finite_m.any():
        return legacy
    music_lufs = float(np.median(l_m[finite_m]))

    music_volume_db = float(np.clip(
        (voice_lufs - pause_offset_lu) - music_lufs,
        volume_clamp_db[0], volume_clamp_db[1],
    ))
    duck_amount_db = float(np.clip(
        -(speech_offset_lu - pause_offset_lu),
        duck_clamp_db[0], duck_clamp_db[1],
    ))
    return music_volume_db, duck_amount_db


def _script_gain_db(
    n: int,
    sample_rate: int,
    phrases: list[tuple[float, float]],
    pre_descent_ms: float,
    attack_ramp_ms: float,
    release_ms: float,
    duck_db: float,
    lift_db: float,
    lift_pause_s: float,
    smooth_hz: float,
) -> np.ndarray:
    """Deterministic music-gain envelope (dB) from phrase timestamps."""
    g_db = np.zeros(n, dtype=np.float32)
    if n == 0:
        return g_db
    duration_s = n / sample_rate

    # 1. Pause-lift plateaus (centre of long gaps; bounding S-curves added below).
    if lift_db > 0 and phrases and lift_pause_s > 0:
        boundaries: list[tuple[float, float]] = []
        prev_end = 0.0
        for (s, e) in phrases:
            if s - prev_end >= lift_pause_s:
                boundaries.append((prev_end, s))
            prev_end = e
        if duration_s - prev_end >= lift_pause_s:
            boundaries.append((prev_end, duration_s))
        for (g_start, g_end) in boundaries:
            plateau_start = g_start + release_ms / 1000.0
            plateau_end = g_end - pre_descent_ms / 1000.0
            i0 = max(0, int(plateau_start * sample_rate))
            i1 = min(n, int(plateau_end * sample_rate))
            if i1 > i0:
                g_db[i0:i1] = lift_db

    # 2. Per-phrase predictive descent → hold → release.
    for idx, (t_on, t_off) in enumerate(phrases):
        ramp_n = max(1, int(attack_ramp_ms / 1000.0 * sample_rate))
        desc_start = int(round((t_on - pre_descent_ms / 1000.0) * sample_rate))
        desc_end = desc_start + ramp_n
        if desc_end > 0 and desc_start < n:
            seg_start = max(0, desc_start)
            seg_end = min(n, desc_end)
            if seg_end > seg_start:
                t = (np.arange(seg_end - seg_start, dtype=np.float32)
                     + (seg_start - desc_start)) / max(1, ramp_n)
                g0 = float(g_db[seg_start])
                g_db[seg_start:seg_end] = g0 + (duck_db - g0) * _smoothstep(t)

        i_on = max(0, int(round(t_on * sample_rate)))
        i_off = min(n, int(round(t_off * sample_rate)))
        if i_off > i_on:
            g_db[i_on:i_off] = duck_db

        next_on = phrases[idx + 1][0] if idx + 1 < len(phrases) else duration_s
        gap_s = next_on - t_off
        target = lift_db if (gap_s >= lift_pause_s and lift_db > 0) else 0.0
        rel_n = max(1, int(release_ms / 1000.0 * sample_rate))
        rel_end = min(n, i_off + rel_n)
        if rel_end > i_off:
            t = np.arange(rel_end - i_off, dtype=np.float32) / max(1, rel_n)
            g_db[i_off:rel_end] = duck_db + (target - duck_db) * _smoothstep(t)

    return _zero_phase_smooth(g_db, sample_rate, smooth_hz)


def _reactive_gain_db(
    voice_audio: np.ndarray,
    sample_rate: int,
    range_db: float,
    threshold_db: float = -32.0,
    smooth_hz: float = 6.0,
) -> np.ndarray:
    """Reactive envelope-follower gain curve (dB, <= 0). Safety net for
    off-script breaths; vectorized + zero-phase smoothed (gradual)."""
    sos = butter(2, [200.0, 4000.0], btype="band", fs=sample_rate, output="sos")
    det = sosfilt(sos, voice_audio).astype(np.float32)
    env = _onepole_rms_env(det, sample_rate, ms=30.0)
    env_db = 20.0 * np.log10(env + 1e-9)
    over = np.clip(env_db - threshold_db, 0.0, None)
    target_db = np.clip(over * (range_db / 9.0), range_db, 0.0).astype(np.float32)
    smoothed = _zero_phase_smooth(target_db, sample_rate, smooth_hz)
    return np.clip(smoothed, range_db, 0.0).astype(np.float32)


def combine_script_with_reactive(g_script: np.ndarray, g_reactive: np.ndarray) -> np.ndarray:
    """Where the script lifts (>0) the script wins (preserve breathing);
    elsewhere take the more-restrictive (deeper) of the two."""
    n = min(g_script.shape[0], g_reactive.shape[0])
    gs = g_script[:n].astype(np.float32, copy=False)
    gr = g_reactive[:n].astype(np.float32, copy=False)
    return np.where(gs > 0, gs, np.minimum(gs, gr)).astype(np.float32)


def compute_breathing_gain_db(
    voice_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_depth_db: float = -15.0,
    pre_descent_ms: float = 600.0,
    attack_ramp_ms: float = 700.0,
    release_ms: float = 1500.0,
    lift_db: float = 1.5,
    lift_pause_s: float = 1.5,
    smooth_hz: float = 6.0,
    vad_threshold_db: float = -40.0,
    phrases: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build the combined breathing-duck gain envelope (dB) for ``voice_audio``.

    ``duck_depth_db`` is the (negative) reduction held during speech. When
    ``phrases`` is None they are detected from the voice via RMS-envelope VAD.
    """
    n = int(voice_audio.shape[-1])
    if phrases is None:
        phrases = detect_phrases(voice_audio, sample_rate, threshold_db=vad_threshold_db)
    g_script = _script_gain_db(
        n, sample_rate, phrases,
        pre_descent_ms=pre_descent_ms, attack_ramp_ms=attack_ramp_ms,
        release_ms=release_ms, duck_db=duck_depth_db,
        lift_db=lift_db, lift_pause_s=lift_pause_s, smooth_hz=smooth_hz,
    )
    g_react = _reactive_gain_db(voice_audio, sample_rate, range_db=duck_depth_db)
    return combine_script_with_reactive(g_script, g_react)


def apply_breathing_duck(
    voice_audio: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_depth_db: float = -15.0,
    **kwargs,
) -> np.ndarray:
    """Apply the breathing duck fullband to ``music_audio`` (mono or stereo).

    The whole bed drops together (not just the mids) so it reads as "very low"
    under speech, then rises gradually in pauses.
    """
    g_db = compute_breathing_gain_db(
        voice_audio, sample_rate, duck_depth_db=duck_depth_db, **kwargs,
    )
    n = min(g_db.shape[0], music_audio.shape[-1])
    gain_lin = (10.0 ** (g_db[:n] / 20.0)).astype(np.float32)
    out = music_audio[..., :n].astype(np.float32).copy()
    if out.ndim == 2:
        out *= gain_lin[np.newaxis, :]
    else:
        out *= gain_lin
    return out


# ---------------------------------------------------------------------------
# True-peak limiting (ITU-R BS.1770, oversampled, vectorized)
# ---------------------------------------------------------------------------


def true_peak_limit(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -1.0,
    oversample: int = 4,
    safety_margin_db: float = 0.3,
    lookahead_ms: float = 2.0,
    smooth_hz: float = 200.0,
) -> np.ndarray:
    """Oversampled true-peak brickwall limiter (replaces pedalboard Limiter).

    Upsamples ``oversample``×, computes a per-sample target gain that keeps the
    inter-sample peak under the ceiling, applies a zero-phase-smoothed release
    that is clamped to the target (so the ceiling is never exceeded), then
    downsamples. Fully vectorized — no per-sample Python loop. Transparent
    below threshold (unlike pedalboard 0.9.23's Limiter, which adds ~+4.75 dB).
    """
    thr = float(10.0 ** ((threshold_db - safety_margin_db) / 20.0))
    is_1d = audio.ndim == 1
    x = audio.reshape(1, -1) if is_1d else audio

    up = resample_poly(x, oversample, 1, axis=-1).astype(np.float32)
    up_sr = sample_rate * oversample

    detector = np.max(np.abs(up), axis=0)
    target = np.minimum(1.0, thr / np.maximum(detector, 1e-12)).astype(np.float32)

    # Look-ahead: gain at sample i already reflects the min target over [i, i+la).
    la = max(1, int(up_sr * lookahead_ms / 1000.0))
    target = minimum_filter1d(target, size=la, origin=-(la // 2), mode="nearest")

    # Smooth release (zero-phase) then clamp to target so the ceiling holds.
    smoothed = _zero_phase_smooth(target, up_sr, smooth_hz)
    gain = np.minimum(smoothed, target)

    up *= gain[np.newaxis, :]
    np.clip(up, -thr, thr, out=up)

    down = resample_poly(up, 1, oversample, axis=-1).astype(np.float32)
    np.clip(down, -thr, thr, out=down)
    return down.squeeze(0) if is_1d else down


# ---------------------------------------------------------------------------
# Full mix
# ---------------------------------------------------------------------------

def mix(
    voice_audio: np.ndarray,
    voice_activity: np.ndarray,
    music_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    duck_amount_db: float = -15.0,
    music_volume_db: float = -16.0,
    music_pre_roll_sec: float = 8.0,
    music_post_roll_sec: float = 15.0,
    fade_in_sec: float = 3.0,
    fade_out_sec: float = 8.0,
    target_lufs: float = -19.0,
    stereo_output: bool = False,
    phrases: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Full mix pipeline: align → level → duck → overlay → fades → normalize.

    Args:
        duck_amount_db: Additional dB reduction during speech on top of
            music_volume_db. -8 dB gives ~20 dB voice-music separation.
        music_volume_db: Baseline music level in dB (applied before ducking).
            -16 dB keeps music subtly present during pauses.
        phrases: Optional pre-detected (start_s, end_s) speech phrases in
            *unaligned voice* time (e.g. from bed calibration). Shifted by
            the pre-roll internally so the ducker reuses them instead of
            re-running VAD. None = detect from the aligned voice.
        music_pre_roll_sec: Music plays alone before voice begins (intro).
            8 s gives listeners time to settle with the music before narration starts.
        music_post_roll_sec: Music plays alone after voice ends (outro).
            15 s allows a graceful, unhurried close.
        fade_out_sec: Fade-out duration for natural exit.

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

    # 4. Breathing sidechain duck (deep, gradual, rises in pauses).
    #    The bed falls with a predictive S-curve starting ~600 ms before each
    #    phrase, sits very low (duck_amount_db) during speech, and rises
    #    gradually over ~1.5 s — lifting slightly during long pauses so the
    #    music "breathes". Applied fullband so the whole bed drops, not just
    #    the mids.
    aligned_phrases = None
    if phrases is not None:
        aligned_phrases = [
            (s + music_pre_roll_sec, e + music_pre_roll_sec) for s, e in phrases
        ]
    ducked_music = apply_breathing_duck(
        aligned_voice, aligned_music, sample_rate,
        duck_depth_db=duck_amount_db,
        phrases=aligned_phrases,
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

    # 1. Apply Pedalboard mastering EQ/glue (no limiter) to the whole array.
    # We do this in-memory (safe for 36 GB RAM target hardware).
    mastered_audio = audio
    if master_chain:
        audio_2d = audio.reshape(1, -1) if audio.ndim == 1 else audio
        mastered_audio = master_chain(audio_2d, sample_rate)
        if audio.ndim == 1:
            mastered_audio = mastered_audio.squeeze(0)

    # 2. LUFS-normalize FIRST, then true-peak limit (order is critical — limiting
    #    first then normalizing would re-exceed the ceiling). The true-peak
    #    limiter replaces pedalboard's Limiter, which inflated level (~+4.75 dB)
    #    and added broadband "static" distortion.
    mix_lufs_gain = calculate_loudness_gain(mastered_audio, sample_rate, target_lufs)
    mastered_audio = (mastered_audio * mix_lufs_gain).astype(np.float32)
    mastered_audio = true_peak_limit(mastered_audio, sample_rate, threshold_db=-1.0)

    # -1.0 dBTP ceiling; final safety clip catches any resample-lowpass ringing.
    ceiling = float(10.0 ** (-1.0 / 20.0))  # ≈ 0.891

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

        chunk = np.clip(chunk, -ceiling, ceiling)

        # write directly as float32. Pedalboard expects (channels, samples).
        if chunk.ndim == 1:
            f.write(chunk.reshape(1, -1).astype(np.float32))
        else:
            f.write(chunk.astype(np.float32))

    f.close()
    return tmp_path
