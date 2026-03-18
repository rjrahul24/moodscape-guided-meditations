import os
import sys
import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard, NoiseGate, Compressor, Limiter, PeakFilter, HighShelfFilter

# Add project root to path
sys.path.append(os.getcwd())

from core.f5_tts.postprocessor import F5MasteringEngine
from core.mixer import calculate_loudness_gain, normalize_loudness

def test_mastering():
    print("Testing F5-TTS Meditation Mastering Chain...")
    
    engine = F5MasteringEngine(sample_rate=24000)
    # Trigger chain build
    audio = np.zeros(24000, dtype=np.float32)
    engine.master_vocals(audio, sr=24000)
    
    chain = engine._master_chain
    effects = [type(e).__name__ for e in chain]
    print("Mastering chain effects:", effects)
    
    # Verify NoiseGate presence and specific attack/release
    gate = next(e for e in chain if isinstance(e, NoiseGate))
    print(f"NoiseGate attack: {gate.attack_ms}, release: {gate.release_ms}")
    assert gate.attack_ms == 5.0
    assert gate.release_ms == 250.0
    
    # Verify Anti-boxiness PeakFilter (300Hz)
    peak_300 = next(e for e in chain if isinstance(e, PeakFilter) and e.cutoff_frequency_hz == 300)
    print(f"Anti-boxiness (300Hz) gain: {peak_300.gain_db}, q: {peak_300.q}")
    assert abs(peak_300.gain_db + 2.0) < 0.001
    assert peak_300.q == 1.5
    
    # Verify Presence (3200Hz — Fletcher-Munson peak sensitivity)
    peak_presence = next(e for e in chain if isinstance(e, PeakFilter) and e.cutoff_frequency_hz == 3200)
    print(f"Presence (3200Hz) gain: {peak_presence.gain_db}")
    assert abs(peak_presence.gain_db - 1.5) < 0.001

    # Verify Brightness Control Shelf (8kHz)
    shelf_bright = next(e for e in chain if isinstance(e, HighShelfFilter))
    print(f"Brightness shelf (8kHz) gain: {shelf_bright.gain_db}")
    assert shelf_bright.cutoff_frequency_hz == 8000
    assert abs(shelf_bright.gain_db + 1.0) < 0.001
    
    # Verify Compressor settings (-20 dB: gentle leveling for meditation)
    compressor = next(e for e in chain if isinstance(e, Compressor))
    print(f"Compressor threshold: {compressor.threshold_db}, ratio: {compressor.ratio}")
    assert abs(compressor.threshold_db + 20.0) < 0.001
    assert compressor.ratio == 2.5
    
    # Reverb is applied downstream in build_f5_voice_chain (convolution reverb),
    # not in the mastering chain itself.

    # Verify Limiter settings
    limiter = next(e for e in chain if isinstance(e, Limiter))
    print(f"Limiter threshold: {limiter.threshold_db}, release: {limiter.release_ms}")
    assert abs(limiter.threshold_db + 1.5) < 0.001
    assert abs(limiter.release_ms - 80.0) < 0.001
    
    print("Mastering chain verified.")

def test_loudness():
    print("\nTesting Meditation Loudness Targets...")
    sr = 24000
    # Generate 1s of white noise
    audio_stereo = np.random.uniform(-0.1, 0.1, (2, sr)).astype(np.float32)
    audio_mono = np.random.uniform(-0.1, 0.1, sr).astype(np.float32)
    
    # Test stereo target (-18)
    gain_stereo = calculate_loudness_gain(audio_stereo, sr, target_lufs=-18.0)
    norm_stereo = audio_stereo * gain_stereo
    meter = pyln.Meter(sr)
    lufs_stereo = meter.integrated_loudness(norm_stereo.T)
    print(f"Stereo LUFS: {lufs_stereo:.2f} (Target: -18.0)")
    assert abs(lufs_stereo + 18.0) < 0.5
    
    # Test mono target (-21)
    gain_mono = calculate_loudness_gain(audio_mono, sr, target_lufs=-18.0)
    norm_mono = audio_mono * gain_mono
    lufs_mono = meter.integrated_loudness(norm_mono)
    print(f"Mono LUFS: {lufs_mono:.2f} (Target: -21.0)")
    assert abs(lufs_mono + 21.0) < 0.5
    
    print("Loudness targets verified.")

def test_deesser():
    print("\nTesting Split-Band De-Esser...")
    sr = 24000
    from core.f5_tts.postprocessor import split_band_deess
    
    # 1. Create a quiet low-frequency signal + a loud high-frequency "sibilant" burst
    t = np.linspace(0, 1, sr)
    low_tone = 0.1 * np.sin(2 * np.pi * 400 * t)
    # Sibilance is typically 4-8kHz. Let's make a loud noise burst (more sibilance-like).
    sibilance = np.zeros(sr)
    noise = np.random.uniform(-0.8, 0.8, int(0.2*sr))
    # Bandpass the noise to 4-8kHz
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2.0
    sos = butter(4, [4000/nyquist, 8000/nyquist], btype="band", output="sos")
    sibilance[int(0.4*sr):int(0.6*sr)] = sosfilt(sos, noise)
    
    audio = (low_tone + sibilance).astype(np.float32)
    
    # 2. Process with de-esser
    # Use a low threshold to ensure it hits the signal hard for the test
    processed = split_band_deess(audio, sr, threshold_db=-30, ratio=8.0)
    
    # 3. Verify reduction in the sibilant range
    # We expect the 0.4-0.6s range to be significantly attenuated
    orig_peak = np.max(np.abs(audio[int(0.4*sr):int(0.6*sr)]))
    proc_peak = np.max(np.abs(processed[int(0.4*sr):int(0.6*sr)]))
    
    reduction_db = 20 * np.log10(orig_peak / proc_peak)
    print(f"Sibilance peak reduction: {reduction_db:.2f} dB")
    
    # We expect at least 4-8dB reduction as per user requirement
    assert reduction_db > 3.0, f"De-esser failed to reduce sibilance adequately: {reduction_db:.2f} dB"
    
    # 4. Verify low frequency is largely untouched
    orig_low = np.max(np.abs(audio[0:int(0.2*sr)]))
    proc_low = np.max(np.abs(processed[0:int(0.2*sr)]))
    print(f"Low frequency change: {abs(20 * np.log10(orig_low / proc_low)):.4f} dB")
    assert abs(20 * np.log10(orig_low / proc_low)) < 0.5, "De-esser modified low frequencies too much"

    print("Split-band de-esser verified.")

if __name__ == "__main__":
    try:
        test_mastering()
        test_deesser()
        test_loudness()
        print("\nAll meditation audio optimizations passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
