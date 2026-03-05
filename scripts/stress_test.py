"""15-minute meditation stress test — measures generation time, quality, consistency."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BODY_PARTS = [
    "forehead", "jaw", "neck and shoulders", "upper back",
    "chest", "stomach", "lower back", "hips",
    "thighs", "knees", "calves", "feet",
    "entire left side", "entire right side",
    "the space behind your eyes", "the center of your chest",
]

STRESS_SCRIPT = (
    "Welcome to this extended deep relaxation session. [pause:3s]\n\n"
    "Find a position that feels completely comfortable... [pause:5s]\n\n"
    "Take a slow breath in through your nose... [pause:4s] "
    "and release gently through your mouth. [pause:6s]\n\n"
    + "\n\n".join([
        f"Now bring your awareness to your {bp}... [pause:5s]\n"
        f"Notice any sensations there... warmth, coolness, tingling... [pause:4s]\n"
        f"With each exhale, allow that area to soften and release. [pause:6s]"
        for bp in BODY_PARTS
    ])
    + "\n\n[pause:8s]\n\n"
    "Now simply rest in this complete stillness. [pause:15s]\n\n"
    "There is nothing to do. Nowhere to go. [pause:10s]\n\n"
    "When you are ready, gently begin to return. [pause:5s]\n\n"
    "Slowly open your eyes. Thank you for this practice. [pause:3s]\n"
)


def main():
    import numpy as np
    import soundfile as sf

    from core.pipeline import MeditationPipeline
    from core.qa_monitor import run_qa_checks

    pipeline = MeditationPipeline()

    print("=" * 60)
    print("MoodScape 15-Minute Stress Test")
    print("=" * 60)

    start = time.time()

    output_path, status = pipeline.generate(
        script=STRESS_SCRIPT,
        music_prompt="slow ambient pads, warm synths, no drums, peaceful",
        voice="golden_hour",
        speed=0.78,
        output_format="wav",
        progress_cb=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
        seed=42,
    )

    elapsed = time.time() - start

    audio, sr = sf.read(output_path)
    duration = len(audio) / sr

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Output duration: {duration / 60:.1f} minutes ({duration:.0f}s)")
    print(f"  Generation time: {elapsed:.1f}s")
    print(f"  Real-time factor: {duration / elapsed:.1f}x")
    print(f"  Output file: {output_path}")

    if status:
        print(f"  Status: {status}")

    qa = run_qa_checks(audio.astype(np.float32), sample_rate=sr, log_results=False)
    print(f"\nQuality Checks:")
    lufs_val = qa["lufs"].get("lufs", "N/A")
    print(f"  LUFS: {lufs_val} (target: {qa['lufs']['target']})")
    print(f"  Clipping: {'PASS' if qa['clipping']['passed'] else 'FAIL'}")
    print(f"  Long silences: {len(qa['silence'])} issues")
    print("=" * 60)


if __name__ == "__main__":
    main()
