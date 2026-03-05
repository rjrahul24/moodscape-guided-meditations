"""CLI batch runner for meditation generation."""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import MeditationPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate guided meditation audio")
    parser.add_argument("script_file", help="Path to meditation script text file")
    parser.add_argument(
        "--music-prompt",
        default="warm ambient pads, no drums",
        help="Music description",
    )
    parser.add_argument(
        "--voice",
        default="golden_hour",
        help="Voice ID, preset name, or comma-separated blend",
    )
    parser.add_argument("--speed", type=float, default=0.78, help="Speaking speed")
    parser.add_argument("--output", default="meditation.wav", help="Output file path")
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0=auto)")
    parser.add_argument("--stems", action="store_true", help="Export separate stems")
    parser.add_argument("--upsample", action="store_true", help="48 kHz output")

    args = parser.parse_args()

    with open(args.script_file, "r") as f:
        script = f.read()

    pipeline = MeditationPipeline()

    def progress(frac, msg):
        print(f"[{frac * 100:5.1f}%] {msg}")

    output_path, status = pipeline.generate(
        script=script,
        music_prompt=args.music_prompt,
        voice=args.voice,
        speed=args.speed,
        output_format=args.format,
        progress_cb=progress,
        seed=args.seed if args.seed != 0 else None,
        do_export_stems=args.stems,
        upsample_48k=args.upsample,
    )

    shutil.copy2(output_path, args.output)
    print(f"\nSaved to: {args.output}")
    if status:
        print(f"Status: {status}")


if __name__ == "__main__":
    main()
