"""Parse meditation scripts with [pause:Xs] markers into structured segments."""

import re


def parse_script(script: str) -> list[dict]:
    """Parse a meditation script into speech and pause segments.

    Supported markers:
        [pause:Xs]   — explicit pause of X seconds (int or float)
        \\n\\n         — paragraph break, treated as a 2.5s pause

    Returns a list of dicts:
        {"type": "speech", "text": "..."}
        {"type": "pause", "duration_sec": float}
    """
    if not script or not script.strip():
        return []

    # First, replace double newlines with a pause marker so we handle them uniformly
    script = re.sub(r'\n\n+', ' [pause:2.5s] ', script)

    # Split on pause markers, capturing the duration
    pause_pattern = r'\[pause:(\d+(?:\.\d+)?)s\]'
    parts = re.split(pause_pattern, script)

    # re.split with a capture group alternates: [text, duration, text, duration, ...]
    segments = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text chunk
            text = part.strip()
            if text:
                segments.append({"type": "speech", "text": text})
        else:
            # Captured pause duration
            duration = float(part)
            if duration > 0:
                # Collapse consecutive pauses: merge with previous if it's also a pause
                if segments and segments[-1]["type"] == "pause":
                    segments[-1]["duration_sec"] += duration
                else:
                    segments.append({"type": "pause", "duration_sec": duration})

    return segments
