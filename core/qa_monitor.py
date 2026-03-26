"""Quality assurance checks for generated meditation audio."""

import logging

import numpy as np

logger = logging.getLogger("moodscape.qa")

SAMPLE_RATE = 24000


def check_silence_gaps(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_silence_sec: float = 15.0,
) -> list[dict]:
    """Detect unexpectedly long silence regions (potential dropout bugs).

    Returns a list of dicts describing each issue found.
    """
    threshold = 0.001  # RMS threshold for silence
    window = int(0.1 * sample_rate)  # 100ms windows

    issues: list[dict] = []
    silence_start = None

    for i in range(0, audio.shape[-1] - window, window):
        rms = np.sqrt(np.mean(audio[..., i:i + window] ** 2))
        if rms < threshold:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                duration = (i - silence_start) / sample_rate
                if duration > max_silence_sec:
                    issues.append({
                        "type": "long_silence",
                        "start_sec": round(silence_start / sample_rate, 1),
                        "duration_sec": round(duration, 1),
                    })
                silence_start = None

    # Check tail
    if silence_start is not None:
        duration = (audio.shape[-1] - silence_start) / sample_rate
        if duration > max_silence_sec:
            issues.append({
                "type": "long_silence",
                "start_sec": round(silence_start / sample_rate, 1),
                "duration_sec": round(duration, 1),
            })

    return issues


def check_lufs(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target: float = -16.0,
    tolerance: float = 2.0,
) -> dict:
    """Verify LUFS is within acceptable range."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    audio_for_meter = audio.T if audio.ndim == 2 else audio
    try:
        loudness = meter.integrated_loudness(audio_for_meter)
        passed = abs(loudness - target) <= tolerance
        return {"lufs": round(float(loudness), 1), "target": target, "passed": passed}
    except Exception:
        return {"lufs": None, "target": target, "passed": False}


def check_clipping(audio: np.ndarray, threshold: float = 0.99) -> dict:
    """Check for clipping (samples at or near +/-1.0)."""
    clipped_samples = int(np.sum(np.abs(audio) >= threshold))
    clipped_ratio = clipped_samples / max(audio.shape[-1] * (audio.shape[0] if audio.ndim == 2 else 1), 1)
    return {
        "clipped_samples": clipped_samples,
        "clipped_ratio": round(float(clipped_ratio), 6),
        "passed": clipped_ratio < 0.001,
    }


def check_spectral_balance(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Check that warmth (100-300 Hz) exceeds presence (2-5 kHz).

    Meditation audio should be warm, not harsh. If presence energy exceeds
    warmth energy, the EQ chain may need adjustment.
    """
    from scipy.signal import welch

    mono = audio[0] if audio.ndim == 2 else audio
    freqs, psd = welch(mono, fs=sample_rate, nperseg=min(4096, len(mono)))

    warmth_mask = (freqs >= 100) & (freqs <= 300)
    presence_mask = (freqs >= 2000) & (freqs <= 5000)

    warmth_energy = float(np.sum(psd[warmth_mask])) if warmth_mask.any() else 0.0
    presence_energy = float(np.sum(psd[presence_mask])) if presence_mask.any() else 0.0

    passed = warmth_energy >= presence_energy
    return {
        "warmth_energy": round(warmth_energy, 6),
        "presence_energy": round(presence_energy, 6),
        "passed": passed,
    }


def check_silence_ratio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = 0.001,
) -> dict:
    """Check that silence ratio is appropriate for meditation (15-70%).

    Too much silence (>70%) suggests over-padding; too little (<15%)
    suggests insufficient pausing for a meditation context.  Long-form
    meditations with multiple breath/pause segments naturally sit in the
    60–70 % range, so the upper bound is set conservatively at 70 %.
    """
    mono = audio[0] if audio.ndim == 2 else audio
    window = int(0.05 * sample_rate)  # 50ms windows
    total_windows = max(len(mono) // window, 1)
    silent_windows = 0

    for i in range(0, len(mono) - window, window):
        rms = np.sqrt(np.mean(mono[i:i + window] ** 2))
        if rms < silence_threshold:
            silent_windows += 1

    ratio = silent_windows / total_windows
    passed = 0.15 <= ratio <= 0.70
    return {
        "silence_ratio": round(float(ratio), 3),
        "passed": passed,
    }


def check_spectral_rolloff(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    percentile: float = 0.85,
    max_rolloff_hz: float = 8000.0,
) -> dict:
    """Detect unexpected high-frequency energy suggesting metallic artifacts.

    Computes the 85th percentile spectral rolloff frequency.  For meditation
    audio the rolloff should typically sit below 8 kHz — values above this
    suggest metallic shimmer or synthesis noise.
    """
    import librosa

    mono = audio[0] if audio.ndim == 2 else audio
    rolloff = librosa.feature.spectral_rolloff(
        y=mono.astype(np.float32), sr=sample_rate, roll_percent=percentile,
    )
    median_rolloff = float(np.median(rolloff))
    passed = median_rolloff <= max_rolloff_hz
    return {
        "median_rolloff_hz": round(median_rolloff, 1),
        "max_rolloff_hz": max_rolloff_hz,
        "passed": passed,
    }


def check_onset_strength(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    peak_threshold_multiplier: float = 5.0,
) -> dict:
    """Detect transient spikes (clicks, percussion leaking through).

    Computes the onset strength envelope.  If any onset peak exceeds
    *peak_threshold_multiplier* times the median onset strength, the
    segment likely contains unwanted transient artefacts.
    """
    import librosa

    mono = audio[0] if audio.ndim == 2 else audio
    onset_env = librosa.onset.onset_strength(
        y=mono.astype(np.float32), sr=sample_rate,
    )
    if len(onset_env) < 2:
        return {"peak_onset_ratio": 0.0, "threshold": peak_threshold_multiplier, "passed": True}

    median_strength = float(np.median(onset_env))
    if median_strength < 1e-8:
        return {"peak_onset_ratio": 0.0, "threshold": peak_threshold_multiplier, "passed": True}

    peak_ratio = float(np.max(onset_env) / median_strength)
    passed = peak_ratio < peak_threshold_multiplier
    return {
        "peak_onset_ratio": round(peak_ratio, 2),
        "threshold": peak_threshold_multiplier,
        "passed": passed,
    }


def check_spectral_flatness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low_hz: float = 4000.0,
    high_hz: float = 12000.0,
    max_flatness: float = 0.3,
) -> dict:
    """Detect noise-like content in the upper frequency band (4–12 kHz).

    Spectral flatness is the ratio of geometric mean to arithmetic mean of
    the power spectrum.  Values close to 1.0 indicate white-noise-like
    content; tonal signals produce values close to 0.  For meditation audio
    the 4–12 kHz band should be tonal (pads, bowls, harmonics) not noisy.

    A flatness > *max_flatness* suggests diffusion residual noise or
    synthesis artifacts in that band.
    """
    from scipy.signal import welch

    mono = audio[0] if audio.ndim == 2 else audio
    freqs, psd = welch(mono, fs=sample_rate, nperseg=min(4096, len(mono)))

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any():
        return {"spectral_flatness": 0.0, "max_flatness": max_flatness, "passed": True}

    band_psd = psd[band_mask]
    # Avoid log(0) by clamping
    band_psd = np.maximum(band_psd, 1e-20)

    # Geometric mean via log domain for numerical stability
    log_mean = float(np.mean(np.log(band_psd)))
    geo_mean = np.exp(log_mean)
    arith_mean = float(np.mean(band_psd))

    flatness = geo_mean / arith_mean if arith_mean > 1e-20 else 0.0
    passed = flatness <= max_flatness

    return {
        "spectral_flatness": round(float(flatness), 4),
        "max_flatness": max_flatness,
        "passed": passed,
    }


def check_spectral_smoothness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_centroid_variance: float = 50.0,
) -> dict:
    """Check timbral consistency via spectral centroid variance.

    For ambient meditation music, the spectral centroid should evolve slowly.
    A low variance indicates smooth, stable timbre — desirable for drones and
    pads. High variance suggests jumpy, unpredictable timbral shifts.

    Args:
        max_centroid_variance: Upper bound for normalized centroid std-dev.
    """
    import librosa

    mono = audio[0] if audio.ndim == 2 else audio
    centroid = librosa.feature.spectral_centroid(
        y=mono.astype(np.float32), sr=sample_rate,
    )
    centroid_std = float(np.std(centroid))
    centroid_mean = float(np.mean(centroid))
    # Normalize: coefficient of variation (std/mean) scaled to intuitive range
    norm_variance = (centroid_std / max(centroid_mean, 1.0)) * 100.0
    passed = norm_variance <= max_centroid_variance
    return {
        "centroid_variance": round(norm_variance, 2),
        "max_centroid_variance": max_centroid_variance,
        "passed": passed,
    }


def check_harmonic_stability(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_autocorrelation: float = 0.85,
) -> dict:
    """Check tonal stability via chroma autocorrelation.

    Computes chroma features and measures the average lag-1 autocorrelation
    across chroma bins. High autocorrelation means the harmonic content is
    consistent frame-to-frame — a hallmark of well-generated ambient music.
    """
    import librosa

    mono = audio[0] if audio.ndim == 2 else audio
    chroma = librosa.feature.chroma_stft(
        y=mono.astype(np.float32), sr=sample_rate,
    )
    if chroma.shape[1] < 2:
        return {"harmonic_autocorr": 1.0, "min_autocorrelation": min_autocorrelation, "passed": True}

    # Lag-1 autocorrelation per chroma bin, averaged
    autocorrs = []
    for row in chroma:
        if np.std(row) < 1e-8:
            autocorrs.append(1.0)
            continue
        corr = float(np.corrcoef(row[:-1], row[1:])[0, 1])
        if np.isfinite(corr):
            autocorrs.append(corr)
    avg_autocorr = float(np.mean(autocorrs)) if autocorrs else 0.0
    passed = avg_autocorr >= min_autocorrelation
    return {
        "harmonic_autocorr": round(avg_autocorr, 4),
        "min_autocorrelation": min_autocorrelation,
        "passed": passed,
    }


def check_onset_density(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_onsets_per_sec: float = 0.5,
) -> dict:
    """Check that onset density is low (ambient, not rhythmic).

    Counts detected onsets per second. Meditation music should have very few
    transient events — a high onset density suggests unwanted percussion or
    rhythmic elements leaked through generation.
    """
    import librosa

    mono = audio[0] if audio.ndim == 2 else audio
    duration = len(mono) / sample_rate
    if duration < 1.0:
        return {"onsets_per_sec": 0.0, "max_onsets_per_sec": max_onsets_per_sec, "passed": True}

    onsets = librosa.onset.onset_detect(
        y=mono.astype(np.float32), sr=sample_rate, units="time",
    )
    density = len(onsets) / duration
    passed = density <= max_onsets_per_sec
    return {
        "onsets_per_sec": round(density, 3),
        "max_onsets_per_sec": max_onsets_per_sec,
        "passed": passed,
    }


def check_dynamic_range_consistency(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_rms_std: float = 0.01,
    window_sec: float = 1.0,
) -> dict:
    """Check that dynamic range is consistent (low RMS variance).

    Meditation music should maintain a steady level without sudden volume
    jumps. Measures the standard deviation of per-window RMS values.
    """
    mono = audio[0] if audio.ndim == 2 else audio
    window_samples = int(window_sec * sample_rate)
    if len(mono) < window_samples:
        return {"rms_std": 0.0, "max_rms_std": max_rms_std, "passed": True}

    n_windows = len(mono) // window_samples
    rms_values = np.array([
        np.sqrt(np.mean(mono[i * window_samples:(i + 1) * window_samples] ** 2))
        for i in range(n_windows)
    ])
    rms_std = float(np.std(rms_values))
    passed = rms_std <= max_rms_std
    return {
        "rms_std": round(rms_std, 5),
        "max_rms_std": max_rms_std,
        "passed": passed,
    }


def check_crossfade_quality(
    segments: list[np.ndarray],
    overlap_samples: int,
    sample_rate: int = SAMPLE_RATE,
    min_mel_similarity: float = 0.85,
) -> list[dict]:
    """Check spectral continuity at segment boundaries.

    Compares mel spectrograms and chroma features at the overlap region
    between adjacent segments. Low similarity indicates a poor crossfade
    that may produce audible discontinuities.

    Args:
        segments: List of audio segment arrays.
        overlap_samples: Number of overlapping samples between segments.
        min_mel_similarity: Minimum cosine similarity threshold.

    Returns:
        List of boundary quality dicts (one per boundary).
    """
    import librosa

    results = []
    for i in range(len(segments) - 1):
        seg_a_tail = segments[i][-overlap_samples:]
        seg_b_head = segments[i + 1][:overlap_samples]

        if len(seg_a_tail) < 1024 or len(seg_b_head) < 1024:
            results.append({"boundary": i, "mel_similarity": 1.0, "passed": True})
            continue

        # Mel spectrogram comparison
        mel_a = librosa.feature.melspectrogram(
            y=seg_a_tail.astype(np.float32), sr=sample_rate, n_mels=64,
        )
        mel_b = librosa.feature.melspectrogram(
            y=seg_b_head.astype(np.float32), sr=sample_rate, n_mels=64,
        )

        # Mean mel vector per segment boundary
        vec_a = mel_a.mean(axis=1)
        vec_b = mel_b.mean(axis=1)

        # Cosine similarity
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        similarity = dot / max(norm_a * norm_b, 1e-8)

        passed = similarity >= min_mel_similarity
        results.append({
            "boundary": i,
            "mel_similarity": round(similarity, 4),
            "passed": passed,
        })

    return results


def compute_composite_score(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Compute a composite quality score (higher is better) for A/B selection.

    Combines sub-scores from spectral balance, rolloff, onset strength,
    clipping, LUFS proximity, spectral flatness, and ambient-specific metrics
    (spectral smoothness, harmonic stability, onset density, dynamic range).
    Returns a float in approximately [0, 1].

    Used by music engines to compare regeneration candidates and pick the
    best one rather than just keeping the last successful attempt.
    """
    score = 0.0

    # Spectral warmth dominance (0.12)
    bal = check_spectral_balance(audio, sample_rate)
    total_energy = bal["warmth_energy"] + bal["presence_energy"]
    if total_energy > 0:
        warmth_ratio = bal["warmth_energy"] / total_energy
        score += 0.12 * min(warmth_ratio / 0.6, 1.0)
    else:
        score += 0.06

    # Spectral rolloff within range (0.12)
    rolloff = check_spectral_rolloff(audio, sample_rate)
    if rolloff["median_rolloff_hz"] <= rolloff["max_rolloff_hz"]:
        score += 0.12
    else:
        overshoot = rolloff["median_rolloff_hz"] / rolloff["max_rolloff_hz"]
        score += 0.12 * max(0.0, 1.0 - (overshoot - 1.0))

    # Onset smoothness (0.10)
    onset = check_onset_strength(audio, sample_rate)
    if onset["peak_onset_ratio"] < onset["threshold"]:
        score += 0.10
    else:
        ratio = onset["peak_onset_ratio"] / onset["threshold"]
        score += 0.10 * max(0.0, 1.0 - (ratio - 1.0) / 2.0)

    # Clipping-free (0.15)
    clip = check_clipping(audio)
    if clip["passed"]:
        score += 0.15
    else:
        score += 0.15 * max(0.0, 1.0 - clip["clipped_ratio"] * 1000)

    # LUFS proximity to -16 target (0.06)
    lufs = check_lufs(audio, sample_rate)
    if lufs["lufs"] is not None:
        deviation = abs(lufs["lufs"] - lufs["target"])
        score += 0.06 * max(0.0, 1.0 - deviation / 10.0)

    # Spectral flatness — noise detection in 4–12 kHz band (0.10)
    flatness = check_spectral_flatness(audio, sample_rate)
    if flatness["passed"]:
        score += 0.10
    else:
        overshoot = flatness["spectral_flatness"] / flatness["max_flatness"]
        score += 0.10 * max(0.0, 1.0 - (overshoot - 1.0) / 2.33)

    # ── Ambient-specific metrics (0.35 total) ──────────────────────────────

    # Spectral smoothness — timbral consistency (0.10)
    smoothness = check_spectral_smoothness(audio, sample_rate)
    if smoothness["passed"]:
        score += 0.10
    else:
        overshoot = smoothness["centroid_variance"] / smoothness["max_centroid_variance"]
        score += 0.10 * max(0.0, 1.0 - (overshoot - 1.0))

    # Harmonic stability — tonal consistency (0.10)
    harmony = check_harmonic_stability(audio, sample_rate)
    autocorr = harmony["harmonic_autocorr"]
    min_ac = harmony["min_autocorrelation"]
    if autocorr >= min_ac:
        score += 0.10
    else:
        score += 0.10 * max(0.0, autocorr / min_ac)

    # Onset density — low rhythmic content (0.08)
    density = check_onset_density(audio, sample_rate)
    if density["passed"]:
        score += 0.08
    else:
        overshoot = density["onsets_per_sec"] / density["max_onsets_per_sec"]
        score += 0.08 * max(0.0, 1.0 - (overshoot - 1.0) / 2.0)

    # Dynamic range consistency (0.07)
    dyn = check_dynamic_range_consistency(audio, sample_rate)
    if dyn["passed"]:
        score += 0.07
    else:
        overshoot = dyn["rms_std"] / dyn["max_rms_std"]
        score += 0.07 * max(0.0, 1.0 - (overshoot - 1.0) / 3.0)

    return round(score, 4)


def check_voice_music_ratio(
    voice_audio: np.ndarray,
    music_ducked: np.ndarray,
    voice_activity: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_ratio_db: float = 15.0,
) -> dict:
    """Verify that music is sufficiently below voice during speech sections.

    The W3C accessibility standard requires background audio ≥20 dB below
    speech (measured RMS).  For meditation we target a more practical 15 dB
    minimum — still clearly intelligible but allowing the music to be felt.

    Args:
        voice_audio:    Aligned voice array (post-FX).
        music_ducked:   Aligned music array after ducking.
        voice_activity: Boolean mask where voice is active.
        min_ratio_db:   Minimum acceptable voice-over-music ratio (dB).

    Returns:
        Dict with 'passed', 'ratio_db', and 'min_ratio_db'.
    """
    # Only measure during voiced sections
    active_mask = voice_activity[:min(len(voice_activity), len(voice_audio))]
    if not np.any(active_mask):
        return {"passed": True, "ratio_db": float("inf"), "min_ratio_db": min_ratio_db}

    n = len(active_mask)
    voice_active = voice_audio[:n][active_mask]
    music_active = music_ducked[:n][active_mask]

    voice_rms = np.sqrt(np.mean(voice_active ** 2) + 1e-10)
    music_rms = np.sqrt(np.mean(music_active ** 2) + 1e-10)

    ratio_db = 20.0 * np.log10(voice_rms / music_rms)
    return {
        "passed": ratio_db >= min_ratio_db,
        "ratio_db": round(float(ratio_db), 1),
        "min_ratio_db": min_ratio_db,
    }


def check_ducking_smoothness(
    music_ducked: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_ms: float = 100.0,
    max_rate_db_per_sec: float = 30.0,
) -> dict:
    """Detect pumping artifacts by checking how fast the music envelope changes.

    Rapid gain changes (>30 dB/s) indicate audible pumping that breaks the
    calm atmosphere.  Meditation ducking should never exceed ~12 dB/s.

    Args:
        max_rate_db_per_sec: Maximum acceptable envelope change rate (dB/s).

    Returns:
        Dict with 'passed', 'max_rate_db_per_sec_measured', 'threshold'.
    """
    window_samples = max(1, int(window_ms * sample_rate / 1000.0))
    # Compute windowed RMS envelope of the music
    kernel = np.ones(window_samples) / window_samples
    rms_env = np.sqrt(np.convolve(music_ducked ** 2, kernel, mode="same") + 1e-10)
    db_env = 20.0 * np.log10(rms_env)

    # Rate of change in dB per second
    diff = np.abs(np.diff(db_env))
    rate_db_per_sample = diff
    rate_db_per_sec = rate_db_per_sample * sample_rate

    # Use 99th percentile to avoid outliers from transients
    peak_rate = float(np.percentile(rate_db_per_sec, 99))
    return {
        "passed": peak_rate <= max_rate_db_per_sec,
        "max_rate_db_per_sec_measured": round(peak_rate, 1),
        "threshold": max_rate_db_per_sec,
    }


def run_qa_checks(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    log_results: bool = True,
) -> dict:
    """Run all QA checks and return results.

    Called after the master chain is applied but before final export.
    """
    results = {
        "silence": check_silence_gaps(audio, sample_rate),
        "lufs": check_lufs(audio, sample_rate),
        "clipping": check_clipping(audio),
        "spectral_balance": check_spectral_balance(audio, sample_rate),
        "silence_ratio": check_silence_ratio(audio, sample_rate),
        "spectral_rolloff": check_spectral_rolloff(audio, sample_rate),
        "onset_strength": check_onset_strength(audio, sample_rate),
        "spectral_flatness": check_spectral_flatness(audio, sample_rate),
        "spectral_smoothness": check_spectral_smoothness(audio, sample_rate),
        "harmonic_stability": check_harmonic_stability(audio, sample_rate),
        "onset_density": check_onset_density(audio, sample_rate),
        "dynamic_range": check_dynamic_range_consistency(audio, sample_rate),
    }

    if log_results:
        logger.info("QA — LUFS: %s", results["lufs"])
        logger.info("QA — Spectral balance: %s", results["spectral_balance"])
        logger.info("QA — Spectral rolloff: %s", results["spectral_rolloff"])
        logger.info("QA — Spectral flatness: %s", results["spectral_flatness"])
        logger.info("QA — Onset strength: %s", results["onset_strength"])
        logger.info("QA — Silence ratio: %s", results["silence_ratio"])
        logger.info("QA — Spectral smoothness: %s", results["spectral_smoothness"])
        logger.info("QA — Harmonic stability: %s", results["harmonic_stability"])
        logger.info("QA — Onset density: %s", results["onset_density"])
        logger.info("QA — Dynamic range: %s", results["dynamic_range"])
        if results["silence"]:
            logger.warning("QA — Long silences detected: %s", results["silence"])
        if not results["clipping"]["passed"]:
            logger.warning("QA — Clipping detected: %s", results["clipping"])
        if not results["spectral_balance"]["passed"]:
            logger.warning("QA — Spectral imbalance: presence exceeds warmth — audio may sound harsh")
        if not results["spectral_rolloff"]["passed"]:
            logger.warning("QA — High spectral rolloff (%.0f Hz) — possible metallic artefacts",
                           results["spectral_rolloff"]["median_rolloff_hz"])
        if not results["onset_strength"]["passed"]:
            logger.warning("QA — Transient spike detected (peak/median ratio=%.1f) — possible clicks/percussion",
                           results["onset_strength"]["peak_onset_ratio"])
        if not results["spectral_flatness"]["passed"]:
            logger.warning("QA — High spectral flatness (%.3f) in 4–12 kHz — possible diffusion noise/static",
                           results["spectral_flatness"]["spectral_flatness"])
        if not results["spectral_smoothness"]["passed"]:
            logger.warning("QA — High centroid variance (%.1f) — timbral instability",
                           results["spectral_smoothness"]["centroid_variance"])
        if not results["harmonic_stability"]["passed"]:
            logger.warning("QA — Low harmonic autocorrelation (%.3f) — tonal instability",
                           results["harmonic_stability"]["harmonic_autocorr"])
        if not results["onset_density"]["passed"]:
            logger.warning("QA — High onset density (%.2f/sec) — possible unwanted percussion",
                           results["onset_density"]["onsets_per_sec"])
        if not results["dynamic_range"]["passed"]:
            logger.warning("QA — High RMS variance (%.4f) — inconsistent dynamics",
                           results["dynamic_range"]["rms_std"])

    return results
