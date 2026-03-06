import sys
import logging
from core.pipeline import MeditationPipeline

def main():
    logging.basicConfig(level=logging.INFO)
    pipeline = MeditationPipeline()

    print("\n--- Testing 'Instrumental Only' ---")
    try:
        out_inst, msg_inst = pipeline.generate(
            script="", 
            music_prompt="Calm piano", 
            generation_mode="Instrumental Only", 
            instrumental_duration_m=0.1
        )
        print(f"SUCCESS: {out_inst} | MSG: {msg_inst}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED 'Instrumental Only': {e}")

    print("\n--- Testing 'Vocals Only' ---")
    try:
        out_voc, msg_voc = pipeline.generate(
            script="Hello world.", 
            music_prompt="", 
            generation_mode="Vocals Only"
        )
        print(f"SUCCESS: {out_voc} | MSG: {msg_voc}")
    except Exception as e:
        print(f"FAILED 'Vocals Only': {e}")

if __name__ == "__main__":
    main()
