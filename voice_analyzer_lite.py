"""
🔬 VOICE SIMILARITY ANALYZER (Lite Version)
============================================

Analyzes voice samples WITHOUT loading full ChatterBox model.
Uses only audio analysis to diagnose quality issues.

This avoids the torchvision compatibility issue.
"""

import os
import sys
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SampleAnalysis:
    """Analysis results for a single audio sample"""
    filename: str
    duration_sec: float
    sample_rate: int
    rms_db: float  # Volume level in dB
    snr_estimate: float  # Signal-to-noise ratio
    silence_ratio: float  # Percentage of silence
    clipping_events: int  # Number of clips
    frequency_bandwidth: float  # Hz range of voice
    fundamental_freq: float  # Average pitch (Hz)
    pitch_variation: float  # Standard deviation of pitch
    spectral_clarity: float  # How clean the spectrum is
    voice_activity_ratio: float  # Ratio of speech to non-speech
    quality_score: float  # Overall 0-100 score
    issues: List[str]  # Specific issues found


class VoiceSimilarityAnalyzerLite:
    """
    Lightweight voice sample analyzer.
    Diagnoses quality issues without loading ML models.
    """
    
    # Recommended parameters for ChatterBox
    EXPECTED_SR = 16000  # Voice encoder expects 16kHz
    S3GEN_SR = 24000     # S3Gen expects 24kHz
    MIN_DURATION = 3.0   # Minimum recommended duration
    OPTIMAL_DURATION = 10.0  # Optimal duration
    MAX_DURATION = 30.0  # Beyond this, quality may decrease
    
    def __init__(self):
        print("🔬 Voice Similarity Analyzer (Lite) initialized")
    
    def analyze_sample(self, audio_path: str) -> SampleAnalysis:
        """Analyze a single audio sample for quality"""
        
        issues = []
        
        # Load audio
        try:
            wav, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            return SampleAnalysis(
                filename=os.path.basename(audio_path),
                duration_sec=0, sample_rate=0, rms_db=-100,
                snr_estimate=0, silence_ratio=1.0, clipping_events=0,
                frequency_bandwidth=0, fundamental_freq=0, pitch_variation=0,
                spectral_clarity=0, voice_activity_ratio=0, quality_score=0,
                issues=[f"Failed to load: {str(e)}"]
            )
        
        duration = len(wav) / sr
        
        # Check duration
        if duration < self.MIN_DURATION:
            issues.append(f"Too short ({duration:.1f}s < {self.MIN_DURATION}s)")
        elif duration > self.MAX_DURATION:
            issues.append(f"Very long ({duration:.1f}s > {self.MAX_DURATION}s recommended)")
        
        # Check sample rate
        if sr < 16000:
            issues.append(f"Low sample rate ({sr}Hz < 16kHz)")
        
        # 1. RMS Level (volume)
        rms = np.sqrt(np.mean(wav**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        if rms_db < -40:
            issues.append(f"Very quiet audio ({rms_db:.1f}dB)")
        elif rms_db < -30:
            issues.append(f"Quiet audio ({rms_db:.1f}dB)")
        elif rms_db > -3:
            issues.append(f"Possibly clipped/too loud ({rms_db:.1f}dB)")
        
        # 2. SNR estimation (using silence detection)
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # Use librosa's frame utility if available
        try:
            frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length)
            frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        except:
            # Fallback
            frame_rms = np.array([rms])
        
        # Estimate noise floor from quietest 10% of frames
        sorted_rms = np.sort(frame_rms)
        noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms)//10)])
        signal_level = np.mean(sorted_rms[-len(sorted_rms)//2:])  # Top 50%
        snr = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        
        if snr < 10:
            issues.append(f"Very noisy (SNR={snr:.1f}dB < 10dB)")
        elif snr < 20:
            issues.append(f"Noisy (SNR={snr:.1f}dB < 20dB)")
        
        # 3. Silence ratio
        silence_threshold = 0.02 * np.max(np.abs(wav))
        silence_ratio = np.sum(np.abs(wav) < silence_threshold) / len(wav)
        
        if silence_ratio > 0.5:
            issues.append(f"Too much silence ({silence_ratio*100:.1f}%)")
        
        # 4. Clipping detection
        clipping_threshold = 0.99
        clipping_events = np.sum(np.abs(wav) > clipping_threshold)
        
        if clipping_events > 100:
            issues.append(f"Severe clipping ({clipping_events} samples)")
        elif clipping_events > 10:
            issues.append(f"Some clipping ({clipping_events} samples)")
        
        # 5. Spectral analysis
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=wav, sr=sr))
        
        # 6. Fundamental frequency (pitch)
        try:
            f0, voiced_flag, _ = librosa.pyin(wav, fmin=50, fmax=500, sr=sr)
            valid_f0 = f0[~np.isnan(f0)]
            avg_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else 0
            pitch_variation = np.std(valid_f0) if len(valid_f0) > 1 else 0
            voice_activity_ratio = len(valid_f0) / len(f0) if len(f0) > 0 else 0
        except:
            avg_f0 = 0
            pitch_variation = 0
            voice_activity_ratio = 0
        
        if avg_f0 == 0:
            issues.append("Could not detect voice pitch")
        elif avg_f0 > 300:
            issues.append(f"Unusual high pitch ({avg_f0:.0f}Hz) - may be a child or female voice")
        
        if pitch_variation < 10 and avg_f0 > 0:
            issues.append("Monotone speech - lacks natural variation")
        
        if voice_activity_ratio < 0.3:
            issues.append(f"Low voice activity ({voice_activity_ratio*100:.1f}%)")
        
        # 7. Spectral clarity (flatness - lower = more tonal/clear)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=wav))
        spectral_clarity = 1.0 - spectral_flatness  # Invert so higher = cleaner
        
        if spectral_clarity < 0.7:
            issues.append(f"Low spectral clarity ({spectral_clarity:.2f})")
        
        # 8. Phone compression detection (common in WhatsApp)
        # Check for narrow bandwidth (phone typically cuts off above 8kHz)
        S = np.abs(librosa.stft(wav))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_energy = np.mean(S[freqs > 8000, :]) if sr > 16000 else 0
        low_freq_energy = np.mean(S[(freqs > 100) & (freqs < 4000), :])
        
        if sr > 16000 and high_freq_energy / (low_freq_energy + 1e-10) < 0.01:
            issues.append("Narrow bandwidth - likely phone-compressed audio")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            duration=duration,
            snr=snr,
            silence_ratio=silence_ratio,
            clipping_events=clipping_events,
            spectral_clarity=spectral_clarity,
            rms_db=rms_db,
            voice_activity_ratio=voice_activity_ratio,
            pitch_variation=pitch_variation
        )
        
        return SampleAnalysis(
            filename=os.path.basename(audio_path),
            duration_sec=round(duration, 2),
            sample_rate=sr,
            rms_db=round(rms_db, 1),
            snr_estimate=round(snr, 1),
            silence_ratio=round(silence_ratio, 3),
            clipping_events=int(clipping_events),
            frequency_bandwidth=round(spectral_bandwidth, 1),
            fundamental_freq=round(avg_f0, 1),
            pitch_variation=round(pitch_variation, 1),
            spectral_clarity=round(spectral_clarity, 3),
            voice_activity_ratio=round(voice_activity_ratio, 3),
            quality_score=round(quality_score, 1),
            issues=issues
        )
    
    def _calculate_quality_score(self, duration, snr, silence_ratio, 
                                  clipping_events, spectral_clarity, 
                                  rms_db, voice_activity_ratio,
                                  pitch_variation) -> float:
        """Calculate overall quality score 0-100"""
        score = 100.0
        
        # Duration penalties/bonuses
        if duration < self.MIN_DURATION:
            score -= 20 * (1 - duration / self.MIN_DURATION)
        elif duration > self.MAX_DURATION:
            score -= 10 * min(1, (duration - self.MAX_DURATION) / 30)
        elif self.MIN_DURATION <= duration <= self.OPTIMAL_DURATION:
            score += 5  # Sweet spot bonus
        
        # SNR penalty
        if snr < 10:
            score -= 30  # Very noisy
        elif snr < 20:
            score -= 15  # Noisy
        elif snr < 30:
            score -= 5   # Acceptable
        
        # Silence ratio penalty
        if silence_ratio > 0.5:
            score -= 20
        elif silence_ratio > 0.3:
            score -= 10
        
        # Clipping penalty
        if clipping_events > 100:
            score -= 20
        elif clipping_events > 10:
            score -= 10
        
        # Spectral clarity bonus
        if spectral_clarity > 0.9:
            score += 10
        elif spectral_clarity > 0.8:
            score += 5
        elif spectral_clarity < 0.6:
            score -= 15
        
        # RMS level check
        if rms_db < -40:
            score -= 15  # Too quiet
        elif rms_db < -30:
            score -= 5
        elif rms_db > -3:
            score -= 10  # Too loud/clipped
        
        # Voice activity
        if voice_activity_ratio < 0.3:
            score -= 15
        elif voice_activity_ratio > 0.7:
            score += 5
        
        # Pitch variation (naturalness)
        if pitch_variation > 20:
            score += 5  # Good natural variation
        
        return max(0, min(100, score))
    
    def analyze_profile(self, profile_path: str) -> Dict:
        """Analyze all samples in a profile"""
        
        profile_dir = Path(profile_path)
        profile_name = profile_dir.name
        samples_dir = profile_dir / "samples"
        
        if not samples_dir.exists():
            print(f"❌ Samples directory not found: {samples_dir}")
            return {}
        
        # Find all audio files
        audio_files = list(samples_dir.glob("*.wav")) + \
                     list(samples_dir.glob("*.mp3")) + \
                     list(samples_dir.glob("*.flac"))
        
        if not audio_files:
            print(f"❌ No audio files found in {samples_dir}")
            return {}
        
        print(f"\n🔬 Analyzing profile: {profile_name}")
        print(f"   Found {len(audio_files)} samples")
        print("-" * 60)
        
        # Analyze each sample
        analyses = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"   [{i+1}/{len(audio_files)}] {audio_file.name}...", end=" ")
            analysis = self.analyze_sample(str(audio_file))
            analyses.append(analysis)
            
            # Print quick status
            status = "✅" if analysis.quality_score >= 70 else "⚠️" if analysis.quality_score >= 50 else "❌"
            print(f"{status} Score: {analysis.quality_score:.0f}")
        
        # Sort by quality
        analyses.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Compile report
        avg_quality = np.mean([a.quality_score for a in analyses])
        avg_snr = np.mean([a.snr_estimate for a in analyses])
        avg_duration = np.mean([a.duration_sec for a in analyses])
        
        # Identify common issues
        all_issues = []
        for a in analyses:
            all_issues.extend(a.issues)
        
        issue_counts = {}
        for issue in all_issues:
            # Simplify issue to category
            for key in ["noisy", "quiet", "short", "silence", "clipping", "phone", "pitch", "clarity"]:
                if key in issue.lower():
                    issue_counts[key] = issue_counts.get(key, 0) + 1
                    break
            else:
                issue_counts[issue[:30]] = issue_counts.get(issue[:30], 0) + 1
        
        # Generate recommendations
        recommendations = []
        
        if avg_quality < 50:
            recommendations.append("🔴 CRITICAL: Sample quality is poor. Re-record in quiet environment with good microphone.")
        elif avg_quality < 70:
            recommendations.append("🟡 MODERATE: Sample quality needs improvement for best results.")
        
        if issue_counts.get("noisy", 0) > len(analyses) * 0.3:
            recommendations.append("🎤 Many samples are noisy. Use noise reduction or record in quieter environment.")
        
        if issue_counts.get("phone", 0) > len(analyses) * 0.5:
            recommendations.append("📱 WhatsApp compression detected! Phone audio loses quality. Record directly on computer with good mic.")
        
        if issue_counts.get("short", 0) > len(analyses) * 0.3:
            recommendations.append("⏱️ Many samples too short. Optimal length is 10-15 seconds.")
        
        if len(analyses) > 20:
            recommendations.append(f"📊 You have {len(analyses)} samples. ChatterBox uses ONE at a time. Focus on your best 3-5.")
        
        # Best samples
        best_samples = [a.filename for a in analyses[:5]]
        worst_samples = [a.filename for a in analyses[-5:]]
        
        # Calculate optimal params based on quality
        recommended_params = {
            "cfg_weight": 0.5 if avg_quality > 70 else 0.3,  # Lower if quality is poor
            "exaggeration": 0.4,  # Keep neutral
            "temperature": 0.7 if avg_quality > 70 else 0.8,  # More random if poor quality
        }
        
        report = {
            "profile_name": profile_name,
            "num_samples": len(analyses),
            "avg_quality_score": round(avg_quality, 1),
            "avg_snr_db": round(avg_snr, 1),
            "avg_duration_sec": round(avg_duration, 1),
            "best_samples": best_samples,
            "worst_samples": worst_samples,
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)),
            "recommendations": recommendations,
            "recommended_params": recommended_params,
            "all_analyses": [asdict(a) for a in analyses]
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        
        print("\n" + "=" * 70)
        print("🔬 VOICE SIMILARITY ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n📁 Profile: {report['profile_name']}")
        print(f"📊 Samples: {report['num_samples']}")
        print(f"⭐ Average Quality: {report['avg_quality_score']}/100")
        print(f"🔊 Average SNR: {report['avg_snr_db']} dB")
        print(f"⏱️ Average Duration: {report['avg_duration_sec']} sec")
        
        print("\n🏆 BEST SAMPLES (use these!):")
        for i, s in enumerate(report['best_samples'], 1):
            print(f"   {i}. {s}")
        
        print("\n📉 WORST SAMPLES (avoid these):")
        for i, s in enumerate(report['worst_samples'], 1):
            print(f"   {i}. {s}")
        
        if report['common_issues']:
            print("\n⚠️ COMMON ISSUES:")
            for issue, count in report['common_issues'].items():
                print(f"   • {issue}: {count} samples")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        print("\n⚙️ RECOMMENDED PARAMETERS:")
        for k, v in report['recommended_params'].items():
            print(f"   {k}: {v}")
        
        print("\n" + "=" * 70)
        print("🎯 KEY INSIGHT: WHY YOUR VOICE DOESN'T MATCH")
        print("=" * 70)
        print("""
ChatterBox is ZERO-SHOT voice cloning. It doesn't learn your voice -
it extracts features from ONE reference audio at a time.

YOUR PROBLEM - WhatsApp voice notes have:
❌ Phone compression (cuts frequencies above 8kHz)
❌ Lossy codec (loses voice detail)  
❌ Variable quality between recordings
❌ Background noise from phone mic
❌ Different emotional states/speaking styles

THIS CAUSES:
• Speaker embedding captures "phone quality" not "your voice"
• Inconsistent embeddings between samples
• Generated audio sounds "phone-like" not like you

SOLUTION:
1. Record 3-5 samples directly on computer with good microphone
2. Use WAV format at 24kHz or higher
3. Speak in quiet room with minimal echo
4. Each sample should be 10-15 seconds
5. Speak in the SAME STYLE you want generated
6. Use your SINGLE BEST sample, not all 30!
""")
        print("=" * 70)


def main():
    """Run voice similarity analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze voice profile quality")
    parser.add_argument("--profile", default="voice_profiles/pritam",
                       help="Path to voice profile directory")
    args = parser.parse_args()
    
    analyzer = VoiceSimilarityAnalyzerLite()
    
    profile_path = Path(__file__).parent / args.profile
    
    if not profile_path.exists():
        print(f"❌ Profile not found: {profile_path}")
        return
    
    report = analyzer.analyze_profile(str(profile_path))
    
    if report:
        analyzer.print_report(report)
        
        # Save report
        report_file = profile_path / "quality_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📝 Report saved to: {report_file}")


if __name__ == "__main__":
    main()
