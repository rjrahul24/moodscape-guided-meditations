"""CLI batch runner for meditation generation."""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.content_profiles import get_profile, normalize_content_type
from core.pipeline import MeditationPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate guided meditation or sleep story audio")
    parser.add_argument("script_file", help="Path to meditation/sleep-story script text file")
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
    parser.add_argument(
        "--content-type",
        choices=["meditation", "sleep_story"],
        default="meditation",
        help="meditation (default) or sleep_story — applies the sleep profile "
             "(shorter paragraph pauses, softer bed, slower speed/longer fades).",
    )
    parser.add_argument("--speed", type=float, default=None, help="Speaking speed (overrides the content-type default)")
    parser.add_argument("--output", default="meditation.wav", help="Output file path")
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0=auto)")
    parser.add_argument("--stems", action="store_true", help="Export separate stems")
    parser.add_argument("--upsample", action="store_true", help="48 kHz output")

    args = parser.parse_args()

    with open(args.script_file, "r") as f:
        script = f.read()

    content_type = normalize_content_type(args.content_type)
    profile = get_profile(content_type)

    # Resolve slider-backed values from the content profile, honouring an explicit
    # --speed override. Meditation keeps the historical CLI speed default (0.78);
    # sleep stories take their slower profile speed and longer/softer mixing.
    if args.speed is not None:
        speed = args.speed
    elif content_type == "sleep_story":
        speed = profile["speed"]
    else:
        speed = 0.78

    extra_kwargs = {}
    if content_type == "sleep_story":
        extra_kwargs = {
            "duck_amount_db": profile["duck_amount_db"],
            "reverb_amount": profile["reverb_amount"],
            "fade_in_sec": profile["fade_in_sec"],
            "fade_out_sec": profile["fade_out_sec"],
        }

    pipeline = MeditationPipeline()

    def progress(frac, msg):
        print(f"[{frac * 100:5.1f}%] {msg}")

    output_path, status = pipeline.generate(
        script=script,
        music_prompt=args.music_prompt,
        voice=args.voice,
        speed=speed,
        output_format=args.format,
        progress_cb=progress,
        seed=args.seed if args.seed != 0 else None,
        do_export_stems=args.stems,
        upsample_48k=args.upsample,
        content_type=content_type,
        **extra_kwargs,
    )

    shutil.copy2(output_path, args.output)
    print(f"\nSaved to: {args.output}")
    if status:
        print(f"Status: {status}")


if __name__ == "__main__":
    main()
