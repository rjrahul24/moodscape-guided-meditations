"""Real 10-minute ACE-Step renders in both long-form modes (loop, then evolve).

Saves WAVs + report.json with composite QA, harmonic stability, and a spectral
check at the expected seam timestamps. Takes ~20 min (loop) + ~30-40 min
(evolve) on M1 Max with the MLX path. Run under caffeinate so idle-sleep
doesn't kill it:

    caffeinate -i .venv/bin/python scripts/verify_acestep_longform.py
"""

import json
import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pathlib
REPO = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = os.environ.get("ACE_VERIFY_OUT", "/tmp/ace_verify")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
os.chdir(REPO)

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO, ".env"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf

from core.acestep import AceStepEngine
from core.qa_monitor import compute_composite_score, check_harmonic_stability

PROMPT = "warm ambient pads, gentle drones, slow harmonic evolution, no drums"
DURATION = 600.0
SR = 48000

engine = AceStepEngine()
engine.load_model(model_type="sft")

report = {}
try:
    for mode in ("loop", "evolve"):
        print(f"\n===== MODE: {mode} =====", flush=True)
        t0 = time.time()
        audio = engine.generate(
            PROMPT, DURATION, lyrics=None, bpm=50, keyscale="Auto",
            long_form_mode=mode, seed=42,
        )
        elapsed = time.time() - t0
        path = os.path.join(OUT, f"ace_600s_{mode}.wav")
        sf.write(path, audio, SR, subtype="PCM_24")

        score = compute_composite_score(audio, SR)
        harm = check_harmonic_stability(audio, SR)

        # Expected seam timestamps
        if mode == "loop":
            seams = [240.0 * k for k in range(1, int(DURATION // 240) + 1)]
        else:
            seams = [90.0 + 60.0 * k for k in range(0, int((DURATION - 90) // 60) + 1)]
        seam_metrics = []
        for t in seams:
            i = int(t * SR)
            w = int(1.0 * SR)
            if i - w < 0 or i + w > len(audio):
                continue
            d = AceStepEngine._seam_discontinuity_db(audio[i - w:i], audio[i:i + w], SR)
            seam_metrics.append({"t": t, "band_delta_db": round(float(d), 2)})

        report[mode] = {
            "path": path,
            "render_minutes": round(elapsed / 60, 1),
            "duration_s": round(len(audio) / SR, 1),
            "composite_score": round(float(score), 3),
            "harmonic_stability": harm,
            "seam_checks": seam_metrics,
        }
        print(json.dumps(report[mode], indent=2, default=str), flush=True)
finally:
    engine.unload_model()

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nACE VERIFY DONE")
