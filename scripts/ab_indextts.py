"""A/B harness for IndexTTS-2 meditation-quality tuning.

Renders a fixed meditation text across a matrix of engine variants so changes
can be compared objectively (speaker-similarity, spectral flatness) and by
blind listening. Each output filename encodes its variant.

Usage:
    .venv/bin/python scripts/ab_indextts.py --voice <slug> [--out DIR]
        [--alphas 0.65,0.55,0.45] [--paces 0.92,1.0] [--quick]

Variants are applied by patching the module-level constants in
core.index_tts.engine before synthesis — no code changes needed per run.

Objective metrics (printed per variant + summary table):
  - speaker similarity: cosine between resemblyzer d-vectors of the reference
    clip and the rendered audio (requires `pip install resemblyzer`; skipped
    gracefully when absent)
  - HF spectral flatness (4-12 kHz): phase-vocoder smearing and synthesis
    artifacts raise it — lower is cleaner
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

# indextts → audiotools → tensorboard ships stale *_pb2.py files that crash the
# protobuf C extension. Force pure-Python mode BEFORE any transitive
# google.protobuf import (same guard as app.py).
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

TEXT = (
    "Welcome to this moment of stillness. "
    "Take a slow, deep breath in, and gently release it. "
    "With each breath, feel the weight of the day beginning to soften."
)

SEGMENTS = [{"type": "speech", "text": TEXT}]


def hf_spectral_flatness(audio: np.ndarray, sr: int, lo: float = 4000.0, hi: float = 12000.0) -> float:
    """Geometric/arithmetic-mean ratio of the HF band magnitude spectrum."""
    from scipy.signal import stft
    _, _, z = stft(audio, fs=sr, nperseg=2048)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    band = (freqs >= lo) & (freqs <= min(hi, sr / 2 - 1))
    mag = np.abs(z[band]) + 1e-12
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(geo / arith)


def speaker_similarity(ref_path: str, audio: np.ndarray, sr: int) -> float | None:
    """Cosine similarity of resemblyzer d-vectors (None if unavailable)."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None
    import librosa
    encoder = speaker_similarity._encoder
    if encoder is None:
        encoder = speaker_similarity._encoder = VoiceEncoder(verbose=False)
    ref_wav = preprocess_wav(Path(ref_path))
    wav16 = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    e_ref = encoder.embed_utterance(ref_wav)
    e_out = encoder.embed_utterance(wav16)
    return float(np.dot(e_ref, e_out) / (np.linalg.norm(e_ref) * np.linalg.norm(e_out) + 1e-9))


speaker_similarity._encoder = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", default=None, help="IndexTTS voice slug (default: first registered)")
    ap.add_argument("--out", default="ab_indextts_out", help="Output directory")
    ap.add_argument("--alphas", default="0.65,0.55,0.45", help="emo_alpha values to test")
    ap.add_argument("--paces", default="0.92,1.0", help="pace rates to test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="Only test current defaults vs old defaults")
    args = ap.parse_args()

    from core.index_tts import engine as idx
    from core.index_tts.voice_registry import scan_voices

    voices = scan_voices()
    if not voices:
        sys.exit("No IndexTTS voices found in assets/speakers/index_tts_voices/")
    slug = args.voice or sorted(voices)[0]
    ref_path = str(voices[slug]["audio"])
    print(f"Voice: {slug} ({ref_path})")

    alphas = [float(x) for x in args.alphas.split(",")]
    paces = [float(x) for x in args.paces.split(",")]
    if args.quick:
        alphas, paces = [0.65, idx.INDEXTTS_EMO_ALPHA], [idx.INDEXTTS_PACE_RATE]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = idx.IndexTTSEngine(voice_slug=slug)
    engine.load_model()

    results = []
    try:
        for alpha, pace in itertools.product(alphas, paces):
            idx.INDEXTTS_EMO_ALPHA = alpha
            idx.INDEXTTS_PACE_RATE = pace
            name = f"alpha{alpha:.2f}_pace{pace:.2f}".replace(".", "p")
            print(f"\n=== rendering {name} ===")
            t0 = time.time()
            audio, _activity = engine.synthesize(SEGMENTS, speed=1.0, seed=args.seed)
            elapsed = time.time() - t0

            path = out_dir / f"{name}.wav"
            sf.write(path, audio, idx.SAMPLE_RATE, subtype="PCM_24")

            sim = speaker_similarity(ref_path, audio, idx.SAMPLE_RATE)
            flat = hf_spectral_flatness(audio, idx.SAMPLE_RATE)
            results.append({
                "variant": name, "emo_alpha": alpha, "pace_rate": pace,
                "speaker_similarity": sim, "hf_flatness": flat,
                "render_s": round(elapsed, 1), "path": str(path),
            })
            sim_s = f"{sim:.4f}" if sim is not None else "n/a (pip install resemblyzer)"
            print(f"    similarity={sim_s}  hf_flatness={flat:.4f}  ({elapsed:.0f}s)")
    finally:
        engine.unload_model()

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== summary (higher similarity better, lower flatness cleaner) ===")
    width = max(len(r["variant"]) for r in results)
    for r in sorted(results, key=lambda r: -(r["speaker_similarity"] or 0)):
        sim_s = f"{r['speaker_similarity']:.4f}" if r["speaker_similarity"] is not None else "  n/a "
        print(f"  {r['variant']:<{width}}  sim={sim_s}  flat={r['hf_flatness']:.4f}")
    print(f"\nOutputs + results.json in {out_dir}/ — listen blind and pick.")


if __name__ == "__main__":
    main()
