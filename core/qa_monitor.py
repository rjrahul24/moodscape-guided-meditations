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


def compute_composite_score(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Compute a composite quality score (higher is better) for A/B selection.

    Combines sub-scores from spectral balance, rolloff, onset strength,
    clipping, LUFS proximity, and spectral flatness (noise detection).
    Returns a float in approximately [0, 1].

    Used by music engines to compare regeneration candidates and pick the
    best one rather than just keeping the last successful attempt.
    """
    score = 0.0

    # Spectral warmth dominance (0.20)
    bal = check_spectral_balance(audio, sample_rate)
    total_energy = bal["warmth_energy"] + bal["presence_energy"]
    if total_energy > 0:
        warmth_ratio = bal["warmth_energy"] / total_energy
        score += 0.20 * min(warmth_ratio / 0.6, 1.0)  # 60%+ warmth → full marks
    else:
        score += 0.10  # neutral if silent

    # Spectral rolloff within range (0.20)
    rolloff = check_spectral_rolloff(audio, sample_rate)
    if rolloff["median_rolloff_hz"] <= rolloff["max_rolloff_hz"]:
        score += 0.20
    else:
        # Partial credit — linear decay up to 2x the threshold
        overshoot = rolloff["median_rolloff_hz"] / rolloff["max_rolloff_hz"]
        score += 0.20 * max(0.0, 1.0 - (overshoot - 1.0))

    # Onset smoothness (0.15)
    onset = check_onset_strength(audio, sample_rate)
    if onset["peak_onset_ratio"] < onset["threshold"]:
        score += 0.15
    else:
        ratio = onset["peak_onset_ratio"] / onset["threshold"]
        score += 0.15 * max(0.0, 1.0 - (ratio - 1.0) / 2.0)

    # Clipping-free (0.20)
    clip = check_clipping(audio)
    if clip["passed"]:
        score += 0.20
    else:
        score += 0.20 * max(0.0, 1.0 - clip["clipped_ratio"] * 1000)

    # LUFS proximity to -16 target (0.10)
    lufs = check_lufs(audio, sample_rate)
    if lufs["lufs"] is not None:
        deviation = abs(lufs["lufs"] - lufs["target"])
        score += 0.10 * max(0.0, 1.0 - deviation / 10.0)
    else:
        score += 0.0

    # Spectral flatness — noise detection in 4–12 kHz band (0.15)
    flatness = check_spectral_flatness(audio, sample_rate)
    if flatness["passed"]:
        score += 0.15
    else:
        # Linear penalty: flatness 0.3 → full marks, flatness 1.0 → zero
        overshoot = flatness["spectral_flatness"] / flatness["max_flatness"]
        score += 0.15 * max(0.0, 1.0 - (overshoot - 1.0) / 2.33)

    return round(score, 4)


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
    }

    if log_results:
        logger.info("QA — LUFS: %s", results["lufs"])
        logger.info("QA — Spectral balance: %s", results["spectral_balance"])
        logger.info("QA — Spectral rolloff: %s", results["spectral_rolloff"])
        logger.info("QA — Spectral flatness: %s", results["spectral_flatness"])
        logger.info("QA — Onset strength: %s", results["onset_strength"])
        logger.info("QA — Silence ratio: %s", results["silence_ratio"])
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

    return results
