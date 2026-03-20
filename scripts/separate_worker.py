"""Standalone worker for AI Source Separation (Demucs).
Runs in a separate process to isolate memory usage.
Uses memory-mapped Numpy for the lowest possible RAM footprint.
"""

import sys
import os
import logging
import numpy as np

# Disable torch's memory caching allocator to reduce peak memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# Add project root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 4:
        print("Usage: python separate_worker.py <input_npy> <sample_rate> <output_npy>")
        sys.exit(1)

    input_file = sys.argv[1]
    sample_rate = int(sys.argv[2])
    output_file = sys.argv[3]

    try:
        import torch
        torch.set_num_threads(1)

        # Load with memory mapping to avoid a full copy in RAM
        audio = np.load(input_file, mmap_mode='r')
        logger.info(f"[SeparateWorker] Loaded audio: {audio.shape} @ {sample_rate}Hz")

        from core.stem_separator import StemSeparator
        separator = StemSeparator()

        result = separator._remove_drums_and_vocals_internal(audio, sample_rate)

        np.save(output_file, result)
        logger.info("[SeparateWorker] Success")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[SeparateWorker] Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
