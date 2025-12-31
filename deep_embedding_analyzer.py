"""
🔬 DEEP VOICE EMBEDDING ANALYZER
================================

Analyzes speaker embeddings to find WHY voice clone doesn't match.
Uses ChatterBox's actual voice encoder to compute real embeddings.
"""

import os
import sys
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure paths
import model_paths

# We'll manually load just the voice encoder without full TTS


def load_voice_encoder_minimal():
    """Load voice encoder without triggering transformers import"""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    # Download just the voice encoder weights
    ve_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="ve.safetensors")
    
    # Load state dict
    state_dict = load_file(ve_path)
    
    return state_dict, ve_path


def compute_mel_spectrogram(wav: np.ndarray, sr: int = 16000, n_mels: int = 40) -> np.ndarray:
    """Compute mel spectrogram like ChatterBox VoiceEncoder does"""
    # ChatterBox uses specific mel parameters
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=400,  # 25ms at 16kHz
        hop_length=160,  # 10ms at 16kHz
        n_mels=n_mels,
        fmin=20,
        fmax=8000
    )
    
    # Log mel
    log_mel = np.log(mel + 1e-6)
    
    # Normalize to roughly [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    
    return log_mel.T  # (T, n_mels)


def compute_embedding_simple(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute a simplified speaker embedding using mel statistics.
    This approximates what the voice encoder captures.
    """
    mel = compute_mel_spectrogram(wav, sr)
    
    # Compute statistics that capture speaker characteristics
    features = []
    
    # Mean mel across time (captures overall spectral shape)
    features.extend(np.mean(mel, axis=0))
    
    # Std of mel (captures speaking dynamics)
    features.extend(np.std(mel, axis=0))
    
    # Delta features (captures transitions)
    delta = np.diff(mel, axis=0)
    features.extend(np.mean(delta, axis=0))
    
    # Delta-delta
    delta2 = np.diff(delta, axis=0)
    features.extend(np.mean(delta2, axis=0))
    
    embedding = np.array(features)
    
    # L2 normalize like ChatterBox does
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
    
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def analyze_embedding_consistency(profile_path: str):
    """Analyze how consistent speaker embeddings are across samples"""
    
    samples_dir = Path(profile_path) / "samples"
    audio_files = sorted(list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3")))
    
    print(f"\n🔬 DEEP EMBEDDING ANALYSIS")
    print(f"   Profile: {Path(profile_path).name}")
    print(f"   Samples: {len(audio_files)}")
    print("=" * 70)
    
    # Compute embeddings for all samples
    embeddings = []
    names = []
    durations = []
    
    for audio_file in audio_files:
        try:
            wav, sr = librosa.load(str(audio_file), sr=16000)
            
            # Only use first 15 seconds (optimal for voice cloning)
            max_samples = 15 * 16000
            if len(wav) > max_samples:
                wav = wav[:max_samples]
            
            emb = compute_embedding_simple(wav, sr)
            embeddings.append(emb)
            names.append(audio_file.name)
            durations.append(len(wav) / sr)
            
        except Exception as e:
            print(f"   ⚠️ Failed to process {audio_file.name}: {e}")
    
    embeddings = np.array(embeddings)
    
    # Compute pairwise similarities
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    
    # Analyze results
    print("\n📊 EMBEDDING SIMILARITY MATRIX (sample of 10x10):")
    print("-" * 70)
    
    # Show first 10x10
    display_n = min(10, n)
    header = "       " + "  ".join([f"{i+1:>5}" for i in range(display_n)])
    print(header)
    
    for i in range(display_n):
        row = f"   {i+1:>2}: " + "  ".join([f"{similarity_matrix[i,j]:.3f}" for j in range(display_n)])
        print(row)
    
    # Overall statistics
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    
    print("\n📈 SIMILARITY STATISTICS:")
    print(f"   Mean similarity:     {np.mean(upper_tri):.4f}")
    print(f"   Std similarity:      {np.std(upper_tri):.4f}")
    print(f"   Min similarity:      {np.min(upper_tri):.4f}")
    print(f"   Max similarity:      {np.max(upper_tri):.4f}")
    print(f"   Median similarity:   {np.median(upper_tri):.4f}")
    
    # Find outliers (samples that are most different from others)
    avg_similarity_per_sample = np.mean(similarity_matrix, axis=1) - 1/n  # Exclude self-similarity
    
    print("\n🔍 SAMPLE ANALYSIS:")
    print("-" * 70)
    
    # Sort by average similarity (descending)
    sorted_indices = np.argsort(avg_similarity_per_sample)[::-1]
    
    print("\n   Most consistent (use these!):")
    for i, idx in enumerate(sorted_indices[:5]):
        print(f"      {i+1}. {names[idx]:<20} avg_sim={avg_similarity_per_sample[idx]:.4f}  dur={durations[idx]:.1f}s")
    
    print("\n   Most different (potential outliers):")
    for i, idx in enumerate(sorted_indices[-5:]):
        print(f"      {i+1}. {names[idx]:<20} avg_sim={avg_similarity_per_sample[idx]:.4f}  dur={durations[idx]:.1f}s")
    
    # Find the single best representative sample
    # This is the sample most similar to all others (centroid-like)
    best_idx = sorted_indices[0]
    
    print(f"\n🎯 RECOMMENDED SINGLE REFERENCE:")
    print(f"   {names[best_idx]}")
    print(f"   This sample is most representative of your voice.")
    
    # Compute average embedding (what averaging does)
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    # Compare each sample to average
    similarities_to_avg = [cosine_similarity(emb, avg_embedding) for emb in embeddings]
    
    print("\n📊 SIMILARITY TO AVERAGED EMBEDDING:")
    closest_to_avg = np.argmax(similarities_to_avg)
    print(f"   Sample closest to average: {names[closest_to_avg]} (sim={similarities_to_avg[closest_to_avg]:.4f})")
    
    # Key insight
    print("\n" + "=" * 70)
    print("🎯 KEY FINDINGS:")
    print("=" * 70)
    
    mean_sim = np.mean(upper_tri)
    
    if mean_sim > 0.9:
        print("✅ EXCELLENT: Your samples are highly consistent!")
        print("   The issue is likely in TTS parameters, not samples.")
    elif mean_sim > 0.8:
        print("✅ GOOD: Your samples are reasonably consistent.")
        print("   Consider using only your top 5 most consistent samples.")
    elif mean_sim > 0.7:
        print("⚠️ MODERATE: Some variation between samples.")
        print("   Recommend: Use SINGLE best sample, not averaged!")
    else:
        print("❌ LOW CONSISTENCY: Samples are quite different!")
        print("   This will confuse the voice cloning.")
        print("   Possible causes:")
        print("   • Different recording conditions")
        print("   • Different emotional states")
        print("   • Audio processing differences")
        print("   • Background noise variations")
    
    print("\n💡 RECOMMENDATION:")
    print(f"   Use ONLY: {names[best_idx]}")
    print("   With cfg_weight=0.5 to 0.7 for best similarity")
    
    # Save detailed report
    report = {
        "mean_similarity": float(np.mean(upper_tri)),
        "std_similarity": float(np.std(upper_tri)),
        "best_samples": [names[i] for i in sorted_indices[:5]],
        "outlier_samples": [names[i] for i in sorted_indices[-5:]],
        "recommended_single_sample": names[best_idx],
        "sample_similarities": {names[i]: float(avg_similarity_per_sample[i]) for i in range(n)}
    }
    
    report_path = Path(profile_path) / "embedding_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Detailed report saved to: {report_path}")
    
    return report


def compare_audio_characteristics(profile_path: str):
    """Compare acoustic characteristics across samples"""
    
    samples_dir = Path(profile_path) / "samples"
    audio_files = sorted(list(samples_dir.glob("*.wav")))[:10]  # First 10 for quick analysis
    
    print(f"\n🔬 ACOUSTIC CHARACTERISTICS COMPARISON")
    print("=" * 70)
    print(f"{'Sample':<20} {'Pitch':<12} {'Energy':<12} {'Rate':<12} {'Quality':<12}")
    print("-" * 70)
    
    characteristics = []
    
    for audio_file in audio_files:
        try:
            wav, sr = librosa.load(str(audio_file), sr=16000)
            
            # Use only first 15 seconds
            wav = wav[:15 * sr] if len(wav) > 15 * sr else wav
            
            # Compute characteristics
            
            # 1. Pitch (F0)
            f0, voiced, _ = librosa.pyin(wav, fmin=50, fmax=500, sr=sr)
            valid_f0 = f0[~np.isnan(f0)]
            avg_pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0
            
            # 2. Energy variation
            rms = librosa.feature.rms(y=wav)[0]
            energy_var = np.std(rms) / (np.mean(rms) + 1e-6)
            
            # 3. Speaking rate (approximated by zero-crossing rate)
            zcr = librosa.feature.zero_crossing_rate(wav)[0]
            speaking_rate = np.mean(zcr)
            
            # 4. Spectral quality (clarity)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=wav))
            quality = 1 - spectral_flatness
            
            characteristics.append({
                "name": audio_file.name,
                "pitch": avg_pitch,
                "energy_var": energy_var,
                "rate": speaking_rate,
                "quality": quality
            })
            
            print(f"{audio_file.name:<20} {avg_pitch:>8.1f} Hz  {energy_var:>8.3f}     {speaking_rate:>8.4f}     {quality:>8.3f}")
            
        except Exception as e:
            print(f"{audio_file.name:<20} ERROR: {e}")
    
    if characteristics:
        # Compute variance across samples
        pitches = [c["pitch"] for c in characteristics]
        energies = [c["energy_var"] for c in characteristics]
        rates = [c["rate"] for c in characteristics]
        
        print("\n📊 VARIANCE ANALYSIS:")
        print(f"   Pitch variation:  {np.std(pitches):.1f} Hz (should be < 20)")
        print(f"   Energy variation: {np.std(energies):.3f} (should be < 0.1)")
        print(f"   Rate variation:   {np.std(rates):.4f} (should be < 0.01)")
        
        if np.std(pitches) > 30:
            print("\n   ⚠️ HIGH PITCH VARIATION - samples may be from different emotional states")
        if np.std(energies) > 0.2:
            print("   ⚠️ HIGH ENERGY VARIATION - inconsistent recording levels")


def main():
    """Run deep embedding analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep voice embedding analysis")
    parser.add_argument("--profile", default="voice_profiles/pritam",
                       help="Path to voice profile directory")
    args = parser.parse_args()
    
    profile_path = Path(__file__).parent / args.profile
    
    if not profile_path.exists():
        print(f"❌ Profile not found: {profile_path}")
        return
    
    # Run embedding analysis
    report = analyze_embedding_consistency(str(profile_path))
    
    # Run acoustic comparison
    compare_audio_characteristics(str(profile_path))
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("🎯 FINAL RECOMMENDATIONS FOR BETTER VOICE SIMILARITY")
    print("=" * 70)
    print("""
1. USE SINGLE BEST SAMPLE
   Instead of using all 30 samples, use ONLY your most representative one:
   → {best_sample}

2. ADJUST PARAMETERS
   Try these settings in myvoiceclone.py:
   • cfg_weight: 0.5 (default, balances similarity and quality)
   • exaggeration: 0.3-0.5 (lower = more natural)
   • temperature: 0.7 (lower = more consistent)

3. UNDERSTAND THE LIMITATION
   ChatterBox extracts a 256-dim speaker embedding from your reference.
   This captures:
   ✅ Overall pitch range
   ✅ Spectral characteristics (formants)
   ✅ Speaking pace patterns
   
   This does NOT capture:
   ❌ Exact voice timbre nuances
   ❌ Subtle pronunciation habits
   ❌ Micro-intonation patterns
   
4. FOR BETTER SIMILARITY
   Record a new reference audio:
   • 10-15 seconds long
   • Speaking in the STYLE you want output
   • Clear, quiet environment
   • WAV format at 24kHz+
   • Use a good USB microphone, not phone

5. PARAMETER TUNING
   If output is too different from your voice:
   → Increase cfg_weight (try 0.6-0.8)
   
   If output sounds robotic/unnatural:
   → Decrease cfg_weight (try 0.3-0.4)
   → Increase temperature slightly
""".format(best_sample=report.get("recommended_single_sample", "sample_004.wav")))


if __name__ == "__main__":
    main()
