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
    """Check that silence ratio is appropriate for meditation (15-60%).

    Too much silence (>60%) suggests over-padding; too little (<15%)
    suggests insufficient pausing for a meditation context.
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
    passed = 0.15 <= ratio <= 0.60
    return {
        "silence_ratio": round(float(ratio), 3),
        "passed": passed,
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
    }

    if log_results:
        logger.info("QA — LUFS: %s", results["lufs"])
        logger.info("QA — Spectral balance: %s", results["spectral_balance"])
        logger.info("QA — Silence ratio: %s", results["silence_ratio"])
        if results["silence"]:
            logger.warning("QA — Long silences detected: %s", results["silence"])
        if not results["clipping"]["passed"]:
            logger.warning("QA — Clipping detected: %s", results["clipping"])
        if not results["spectral_balance"]["passed"]:
            logger.warning("QA — Spectral imbalance: presence exceeds warmth — audio may sound harsh")

    return results
