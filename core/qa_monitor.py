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

    for i in range(0, len(audio) - window, window):
        rms = np.sqrt(np.mean(audio[i:i + window] ** 2))
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
        duration = (len(audio) - silence_start) / sample_rate
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
    try:
        loudness = meter.integrated_loudness(audio)
        passed = abs(loudness - target) <= tolerance
        return {"lufs": round(float(loudness), 1), "target": target, "passed": passed}
    except Exception:
        return {"lufs": None, "target": target, "passed": False}


def check_clipping(audio: np.ndarray, threshold: float = 0.99) -> dict:
    """Check for clipping (samples at or near +/-1.0)."""
    clipped_samples = int(np.sum(np.abs(audio) >= threshold))
    clipped_ratio = clipped_samples / max(len(audio), 1)
    return {
        "clipped_samples": clipped_samples,
        "clipped_ratio": round(float(clipped_ratio), 6),
        "passed": clipped_ratio < 0.001,
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
    }

    if log_results:
        logger.info("QA — LUFS: %s", results["lufs"])
        if results["silence"]:
            logger.warning("QA — Long silences detected: %s", results["silence"])
        if not results["clipping"]["passed"]:
            logger.warning("QA — Clipping detected: %s", results["clipping"])

    return results
