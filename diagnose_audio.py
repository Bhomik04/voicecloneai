"""
Diagnostic tool to check audio quality and identify problems
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_audio(file_path: str):
    """Analyze audio file and report quality metrics"""
    print(f"\n📊 Analyzing: {Path(file_path).name}")
    print("="*70)
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    
    # Basic stats
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample Rate: {sr} Hz")
    print(f"Samples: {len(audio)}")
    
    # Amplitude analysis
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    print(f"\n📈 Amplitude:")
    print(f"  Max: {max_amp:.4f}")
    print(f"  RMS: {rms:.4f}")
    print(f"  RMS (dB): {rms_db:.2f} dB")
    print(f"  Dynamic Range: {'GOOD' if max_amp < 0.99 else 'CLIPPING!'}")
    
    # Noise floor analysis
    sorted_audio = np.sort(np.abs(audio))
    noise_floor = np.mean(sorted_audio[:int(len(sorted_audio) * 0.1)])  # Bottom 10%
    noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
    
    print(f"\n🔇 Noise Floor:")
    print(f"  Level: {noise_floor:.6f}")
    print(f"  Level (dB): {noise_floor_db:.2f} dB")
    print(f"  SNR: {rms_db - noise_floor_db:.2f} dB")
    print(f"  Quality: {'CLEAN' if noise_floor < 0.001 else 'NOISY' if noise_floor < 0.01 else 'VERY NOISY'}")
    
    # Frequency analysis
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft)
    
    # Find dominant frequencies
    top_freqs_idx = np.argsort(magnitude)[-5:][::-1]
    
    print(f"\n🎵 Frequency Content:")
    print(f"  Dominant frequencies:")
    for i, idx in enumerate(top_freqs_idx[:3], 1):
        print(f"    {i}. {freqs[idx]:.1f} Hz")
    
    # Check for voice range (85-255 Hz for fundamental)
    voice_range = (freqs >= 85) & (freqs <= 255)
    voice_energy = np.sum(magnitude[voice_range])
    total_energy = np.sum(magnitude)
    voice_ratio = voice_energy / total_energy
    
    print(f"  Voice range energy: {voice_ratio*100:.1f}%")
    print(f"  Voice presence: {'GOOD' if voice_ratio > 0.1 else 'WEAK'}")
    
    # Dynamic range analysis
    # Calculate envelope
    from scipy.ndimage import gaussian_filter1d
    envelope = np.abs(audio)
    envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.01*sr))
    
    # Check for variation
    env_std = np.std(envelope_smooth)
    env_mean = np.mean(envelope_smooth)
    variation_coeff = env_std / (env_mean + 1e-10)
    
    print(f"\n🌊 Dynamics:")
    print(f"  Variation: {variation_coeff:.4f}")
    print(f"  Dynamic quality: {'NATURAL' if variation_coeff > 0.3 else 'FLAT/SYNTHETIC'}")
    
    # TTS artifacts detection
    # Check for unnaturally consistent amplitude
    frame_length = int(0.02 * sr)  # 20ms frames
    frame_rms = []
    for i in range(0, len(audio) - frame_length, frame_length):
        frame = audio[i:i+frame_length]
        frame_rms.append(np.sqrt(np.mean(frame**2)))
    
    frame_rms = np.array(frame_rms)
    rms_std = np.std(frame_rms)
    rms_mean = np.mean(frame_rms)
    rms_variation = rms_std / (rms_mean + 1e-10)
    
    print(f"\n🤖 TTS Artifact Detection:")
    print(f"  Frame-to-frame variation: {rms_variation:.4f}")
    if rms_variation < 0.2:
        print(f"  ⚠️  WARNING: Unnaturally consistent amplitude (TTS artifact)")
    else:
        print(f"  ✓ Natural amplitude variation")
    
    # Overall assessment
    print(f"\n📋 OVERALL ASSESSMENT:")
    issues = []
    if max_amp > 0.99:
        issues.append("CLIPPING")
    if noise_floor > 0.01:
        issues.append("HIGH NOISE")
    if variation_coeff < 0.3:
        issues.append("FLAT DYNAMICS")
    if rms_variation < 0.2:
        issues.append("TTS ARTIFACTS")
    if voice_ratio < 0.1:
        issues.append("WEAK VOICE")
    
    if issues:
        print(f"  ❌ Issues found: {', '.join(issues)}")
    else:
        print(f"  ✅ Audio quality is GOOD")
    
    return {
        'duration': duration,
        'rms_db': rms_db,
        'noise_floor_db': noise_floor_db,
        'snr': rms_db - noise_floor_db,
        'variation': variation_coeff,
        'issues': issues
    }


def compare_folders(raw_folder: str, enhanced_folder: str):
    """Compare raw vs enhanced samples"""
    print("\n" + "="*70)
    print("COMPARING RAW vs ENHANCED SAMPLES")
    print("="*70)
    
    raw_path = Path(raw_folder)
    enhanced_path = Path(enhanced_folder)
    
    # Get matching files
    raw_files = sorted(raw_path.glob("*.wav"))
    enhanced_files = sorted(enhanced_path.glob("*.wav"))
    
    if not raw_files:
        print(f"❌ No WAV files in {raw_folder}")
        return
    
    if not enhanced_files:
        print(f"❌ No WAV files in {enhanced_folder}")
        print(f"⚠️  Preprocessing may have failed!")
        return
    
    print(f"\n📁 Raw samples: {len(raw_files)}")
    print(f"📁 Enhanced samples: {len(enhanced_files)}")
    
    # Compare first sample
    if raw_files and enhanced_files:
        print(f"\n{'='*70}")
        print("SAMPLE COMPARISON (First File)")
        print(f"{'='*70}")
        
        print("\n🔴 RAW:")
        raw_stats = analyze_audio(str(raw_files[0]))
        
        print("\n🟢 ENHANCED:")
        enh_stats = analyze_audio(str(enhanced_files[0]))
        
        # Show improvement
        print(f"\n{'='*70}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*70}")
        
        snr_improvement = enh_stats['snr'] - raw_stats['snr']
        print(f"SNR improvement: {snr_improvement:+.2f} dB")
        
        variation_change = enh_stats['variation'] - raw_stats['variation']
        print(f"Dynamic variation change: {variation_change:+.4f}")
        
        if snr_improvement > 3:
            print("✅ Preprocessing improved SNR significantly")
        else:
            print("⚠️  Preprocessing did not improve SNR much")
        
        if len(enh_stats['issues']) < len(raw_stats['issues']):
            print("✅ Preprocessing fixed some issues")
        else:
            print("⚠️  Preprocessing did not fix issues")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\n🔍 AUDIO DIAGNOSTICS TOOL")
        print("="*70)
        print("\nUsage:")
        print("  1. Analyze single file:")
        print("     python diagnose_audio.py audio.wav")
        print("\n  2. Compare raw vs enhanced:")
        print("     python diagnose_audio.py --compare raw_folder enhanced_folder")
        print("\nExamples:")
        print("  python diagnose_audio.py \"audio_output/generated.wav\"")
        print("  python diagnose_audio.py --compare \"voice_profiles/pritam/samples\" \"voice_profiles/pritam/samples_enhanced\"")
        return
    
    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("❌ Need both folders for comparison")
            return
        compare_folders(sys.argv[2], sys.argv[3])
    else:
        analyze_audio(sys.argv[1])


if __name__ == "__main__":
    main()
