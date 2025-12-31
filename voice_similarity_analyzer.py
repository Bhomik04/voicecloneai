"""
🔬 VOICE SIMILARITY ANALYZER
============================

Comprehensive analysis of why your voice clone doesn't sound like you.
This tool diagnoses:
1. Reference audio quality issues
2. Speaker embedding quality
3. Parameter optimization suggestions
4. Best sample selection
5. Specific recommendations

Based on analysis of ChatterBox TTS internals and voice cloning research.
"""

import os
import sys
import json
import numpy as np
import librosa
import torch
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add chatterbox to path
CHATTERBOX_PATH = Path(__file__).parent / "chatterbox"
sys.path.insert(0, str(CHATTERBOX_PATH / "src"))

# Import model paths first
import model_paths

from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.tts import ChatterboxTTS


@dataclass
class SampleAnalysis:
    """Analysis results for a single audio sample"""
    filename: str
    duration_sec: float
    sample_rate: int
    rms_level: float  # Volume level
    snr_estimate: float  # Signal-to-noise ratio
    silence_ratio: float  # Percentage of silence
    clipping_events: int  # Number of clips
    frequency_bandwidth: float  # Hz range of voice
    fundamental_freq: float  # Average pitch (Hz)
    spectral_clarity: float  # How clean the spectrum is
    embedding_norm: float  # L2 norm of embedding
    quality_score: float  # Overall 0-100 score


@dataclass  
class VoiceSimilarityReport:
    """Complete voice similarity analysis report"""
    profile_name: str
    num_samples: int
    best_samples: List[str]
    worst_samples: List[str]
    overall_quality: float
    embedding_consistency: float
    recommended_params: Dict
    issues: List[str]
    recommendations: List[str]


class VoiceSimilarityAnalyzer:
    """
    Analyzes voice samples and embeddings to diagnose why cloned voice
    doesn't sound similar to the original.
    """
    
    # ChatterBox's expected parameters
    EXPECTED_SR = 16000  # Voice encoder expects 16kHz
    S3GEN_SR = 24000     # S3Gen expects 24kHz
    MIN_DURATION = 3.0   # Minimum recommended duration
    OPTIMAL_DURATION = 10.0  # Optimal duration
    MAX_DURATION = 30.0  # Beyond this, quality may decrease
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🔬 Voice Similarity Analyzer initialized on {self.device}")
        
        # Load voice encoder
        print("   Loading voice encoder...")
        self.ve = self._load_voice_encoder()
        
    def _load_voice_encoder(self) -> VoiceEncoder:
        """Load the ChatterBox voice encoder"""
        try:
            model = ChatterboxTTS.from_pretrained(device=self.device)
            return model.ve
        except Exception as e:
            print(f"   ⚠️ Could not load pretrained: {e}")
            print("   Trying local load...")
            ve = VoiceEncoder()
            # Load weights from HuggingFace cache
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            ve_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="ve.safetensors")
            ve.load_state_dict(load_file(ve_path))
            ve.to(self.device).eval()
            return ve
    
    def analyze_sample(self, audio_path: str) -> SampleAnalysis:
        """Analyze a single audio sample for quality and voice characteristics"""
        
        # Load audio
        wav, sr = librosa.load(audio_path, sr=None)
        duration = len(wav) / sr
        
        # 1. Basic audio stats
        rms = np.sqrt(np.mean(wav**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # 2. SNR estimation (using silence detection)
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        
        # Estimate noise floor from quietest 10% of frames
        sorted_rms = np.sort(frame_rms)
        noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms)//10)])
        signal_level = np.mean(sorted_rms[-len(sorted_rms)//2:])  # Top 50%
        snr = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        
        # 3. Silence ratio
        silence_threshold = 0.02 * np.max(np.abs(wav))
        silence_ratio = np.sum(np.abs(wav) < silence_threshold) / len(wav)
        
        # 4. Clipping detection
        clipping_threshold = 0.99
        clipping_events = np.sum(np.abs(wav) > clipping_threshold)
        
        # 5. Frequency analysis
        S = np.abs(librosa.stft(wav))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Spectral centroid (average frequency)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr))
        
        # Bandwidth (frequency range)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=wav, sr=sr))
        
        # 6. Fundamental frequency (pitch)
        f0, voiced_flag, _ = librosa.pyin(wav, fmin=50, fmax=500, sr=sr)
        valid_f0 = f0[~np.isnan(f0)]
        avg_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else 0
        
        # 7. Spectral clarity (flatness - lower = more tonal/clear)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=wav))
        spectral_clarity = 1.0 - spectral_flatness  # Invert so higher = cleaner
        
        # 8. Get embedding
        wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=self.EXPECTED_SR)
        with torch.no_grad():
            embedding = self.ve.embeds_from_wavs([wav_16k], sample_rate=self.EXPECTED_SR, as_spk=True)
        embedding_norm = np.linalg.norm(embedding)
        
        # 9. Calculate quality score
        quality_score = self._calculate_quality_score(
            duration=duration,
            snr=snr,
            silence_ratio=silence_ratio,
            clipping_events=clipping_events,
            spectral_clarity=spectral_clarity,
            embedding_norm=embedding_norm,
            rms_db=rms_db
        )
        
        return SampleAnalysis(
            filename=os.path.basename(audio_path),
            duration_sec=duration,
            sample_rate=sr,
            rms_level=rms_db,
            snr_estimate=snr,
            silence_ratio=silence_ratio,
            clipping_events=clipping_events,
            frequency_bandwidth=spectral_bandwidth,
            fundamental_freq=avg_f0,
            spectral_clarity=spectral_clarity,
            embedding_norm=embedding_norm,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(self, duration, snr, silence_ratio, 
                                  clipping_events, spectral_clarity, 
                                  embedding_norm, rms_db) -> float:
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
        
        # Silence ratio penalty (too much silence = less voice data)
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
        
        # RMS level check (should be reasonable, not too quiet)
        if rms_db < -40:
            score -= 15  # Too quiet
        elif rms_db < -30:
            score -= 5
        
        # Embedding norm check (should be ~1.0 for L2 normalized)
        if abs(embedding_norm - 1.0) > 0.01:
            score -= 5  # Embedding might be off
        
        return max(0, min(100, score))
    
    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Get speaker embedding for an audio file"""
        wav, sr = librosa.load(audio_path, sr=self.EXPECTED_SR)
        with torch.no_grad():
            embedding = self.ve.embeds_from_wavs([wav], sample_rate=self.EXPECTED_SR, as_spk=True)
        return embedding
    
    def analyze_profile(self, profile_path: str) -> VoiceSimilarityReport:
        """
        Comprehensive analysis of a voice profile.
        
        Args:
            profile_path: Path to profile directory containing 'samples' folder
            
        Returns:
            VoiceSimilarityReport with findings and recommendations
        """
        profile_dir = Path(profile_path)
        profile_name = profile_dir.name
        samples_dir = profile_dir / "samples"
        
        if not samples_dir.exists():
            raise ValueError(f"Samples directory not found: {samples_dir}")
        
        # Find all audio files
        audio_files = list(samples_dir.glob("*.wav")) + \
                     list(samples_dir.glob("*.mp3")) + \
                     list(samples_dir.glob("*.flac"))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {samples_dir}")
        
        print(f"\n🔬 Analyzing profile: {profile_name}")
        print(f"   Found {len(audio_files)} samples")
        
        # Analyze each sample
        analyses = []
        embeddings = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"   [{i+1}/{len(audio_files)}] Analyzing {audio_file.name}...")
            try:
                analysis = self.analyze_sample(str(audio_file))
                analyses.append(analysis)
                embedding = self.get_embedding(str(audio_file))
                embeddings.append((audio_file.name, embedding))
            except Exception as e:
                print(f"      ⚠️ Failed: {e}")
        
        # Sort by quality
        analyses.sort(key=lambda x: x.quality_score, reverse=True)
        best_samples = [a.filename for a in analyses[:5]]
        worst_samples = [a.filename for a in analyses[-5:]]
        
        # Calculate embedding consistency (cosine similarity between all pairs)
        if len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i][1], embeddings[j][1])
                    similarities.append(sim)
            embedding_consistency = np.mean(similarities)
        else:
            embedding_consistency = 1.0
        
        # Identify issues
        issues = []
        recommendations = []
        
        # Check sample count
        if len(analyses) < 5:
            issues.append("❌ Too few samples (need at least 5-10)")
            recommendations.append("Record more voice samples in quiet environment")
        elif len(analyses) > 30:
            issues.append("⚠️ Many samples may dilute voice identity")
            recommendations.append("Use only your 5-10 best quality samples")
        
        # Check overall quality
        avg_quality = np.mean([a.quality_score for a in analyses])
        if avg_quality < 50:
            issues.append("❌ Low overall sample quality")
            recommendations.append("Re-record samples in quiet environment with better mic")
        elif avg_quality < 70:
            issues.append("⚠️ Moderate sample quality - could be improved")
        
        # Check embedding consistency  
        if embedding_consistency < 0.7:
            issues.append("❌ LOW EMBEDDING CONSISTENCY - samples sound too different!")
            recommendations.append("Your samples have different voice characteristics - use only similar samples")
            recommendations.append("Some samples may have different speakers, noise, or processing")
        elif embedding_consistency < 0.85:
            issues.append("⚠️ Moderate embedding variation")
            recommendations.append("Consider using only your most consistent samples")
        
        # Check for noise issues
        noisy_samples = [a for a in analyses if a.snr_estimate < 20]
        if len(noisy_samples) > len(analyses) * 0.3:
            issues.append(f"❌ {len(noisy_samples)} samples have noise issues")
            recommendations.append("Use noise-reduced samples or record in quiet environment")
        
        # Check for duration issues
        short_samples = [a for a in analyses if a.duration_sec < 5]
        if len(short_samples) > len(analyses) * 0.5:
            issues.append(f"⚠️ {len(short_samples)} samples are very short (<5 sec)")
            recommendations.append("Optimal sample length is 10-15 seconds")
        
        # Check for clipping
        clipped_samples = [a for a in analyses if a.clipping_events > 10]
        if clipped_samples:
            issues.append(f"⚠️ {len(clipped_samples)} samples have audio clipping")
            recommendations.append("Re-record with lower gain to avoid clipping")
        
        # Calculate recommended parameters based on analysis
        avg_snr = np.mean([a.snr_estimate for a in analyses])
        avg_clarity = np.mean([a.spectral_clarity for a in analyses])
        
        recommended_params = self._calculate_optimal_params(
            embedding_consistency=embedding_consistency,
            avg_quality=avg_quality,
            avg_snr=avg_snr,
            avg_clarity=avg_clarity
        )
        
        # Add key recommendations
        if not issues:
            recommendations.append("✅ Sample quality looks good!")
        
        recommendations.append(f"🎯 Use ONLY your best sample: {best_samples[0]}")
        recommendations.append("💡 For best similarity, pass only ONE high-quality sample, not averaged embedding")
        recommendations.append(f"⚙️ Try these parameters: cfg_weight={recommended_params['cfg_weight']}, exaggeration={recommended_params['exaggeration']}")
        
        return VoiceSimilarityReport(
            profile_name=profile_name,
            num_samples=len(analyses),
            best_samples=best_samples,
            worst_samples=worst_samples,
            overall_quality=avg_quality,
            embedding_consistency=embedding_consistency,
            recommended_params=recommended_params,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_optimal_params(self, embedding_consistency: float,
                                   avg_quality: float, avg_snr: float,
                                   avg_clarity: float) -> Dict:
        """
        Calculate optimal TTS parameters based on analysis.
        
        Key insight from ChatterBox research:
        - cfg_weight: Higher = more similar to reference voice
        - exaggeration: Controls emotion intensity
        - temperature: Controls randomness
        """
        
        params = {
            "cfg_weight": 0.5,      # Default
            "exaggeration": 0.5,    # Default  
            "temperature": 0.8,     # Default
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
        
        # If embeddings are inconsistent, use lower cfg to average out differences
        # If embeddings are consistent, use higher cfg for closer match
        if embedding_consistency > 0.9:
            params["cfg_weight"] = 0.7  # High consistency = can use higher guidance
        elif embedding_consistency > 0.8:
            params["cfg_weight"] = 0.5
        else:
            params["cfg_weight"] = 0.3  # Low consistency = lower guidance to avoid artifacts
        
        # High quality samples = can use higher cfg
        if avg_quality > 80:
            params["cfg_weight"] = min(0.8, params["cfg_weight"] + 0.1)
        elif avg_quality < 50:
            params["cfg_weight"] = max(0.2, params["cfg_weight"] - 0.1)
        
        # Temperature based on quality
        if avg_clarity > 0.85:
            params["temperature"] = 0.7  # Clear voice = lower temp for consistency
        else:
            params["temperature"] = 0.85  # Less clear = more variation may help
        
        # Exaggeration - keep low for natural sound
        params["exaggeration"] = 0.4  # Slightly below neutral for naturalness
        
        return params
    
    def compare_embeddings(self, sample1: str, sample2: str) -> float:
        """Compare two audio samples and return cosine similarity"""
        emb1 = self.get_embedding(sample1)
        emb2 = self.get_embedding(sample2)
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def find_best_single_sample(self, profile_path: str) -> Tuple[str, float]:
        """
        Find the single best reference sample for voice cloning.
        
        ChatterBox CRITICAL INSIGHT: 
        The model uses ONLY ONE reference audio at a time.
        Averaging multiple samples can DILUTE the voice identity!
        
        Returns:
            (best_sample_path, quality_score)
        """
        samples_dir = Path(profile_path) / "samples"
        audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        
        best_score = 0
        best_file = None
        
        print("\n🎯 Finding best single reference sample...")
        
        for audio_file in audio_files:
            try:
                analysis = self.analyze_sample(str(audio_file))
                
                # Ideal sample: 10-15 sec, high SNR, clear voice
                score = analysis.quality_score
                
                # Bonus for optimal duration
                if 8 <= analysis.duration_sec <= 15:
                    score += 10
                elif 5 <= analysis.duration_sec <= 20:
                    score += 5
                
                # Bonus for high SNR
                if analysis.snr_estimate > 30:
                    score += 10
                elif analysis.snr_estimate > 25:
                    score += 5
                
                print(f"   {audio_file.name}: score={score:.1f} (dur={analysis.duration_sec:.1f}s, SNR={analysis.snr_estimate:.1f}dB)")
                
                if score > best_score:
                    best_score = score
                    best_file = str(audio_file)
                    
            except Exception as e:
                print(f"   ⚠️ {audio_file.name}: Failed - {e}")
        
        return best_file, best_score
    
    def generate_report(self, report: VoiceSimilarityReport) -> str:
        """Generate a formatted text report"""
        lines = [
            "=" * 70,
            "🔬 VOICE SIMILARITY ANALYSIS REPORT",
            "=" * 70,
            f"Profile: {report.profile_name}",
            f"Samples analyzed: {report.num_samples}",
            "",
            "📊 SCORES:",
            f"   Overall Quality: {report.overall_quality:.1f}/100",
            f"   Embedding Consistency: {report.embedding_consistency:.2f} (1.0 = perfect)",
            "",
            "🏆 BEST SAMPLES (use these!):",
        ]
        
        for i, sample in enumerate(report.best_samples[:5], 1):
            lines.append(f"   {i}. {sample}")
        
        lines.extend([
            "",
            "📉 WORST SAMPLES (avoid these):",
        ])
        
        for i, sample in enumerate(report.worst_samples[:5], 1):
            lines.append(f"   {i}. {sample}")
        
        if report.issues:
            lines.extend(["", "⚠️ ISSUES FOUND:"])
            for issue in report.issues:
                lines.append(f"   {issue}")
        
        lines.extend(["", "💡 RECOMMENDATIONS:"])
        for rec in report.recommendations:
            lines.append(f"   {rec}")
        
        lines.extend([
            "",
            "⚙️ RECOMMENDED PARAMETERS:",
            f"   cfg_weight: {report.recommended_params['cfg_weight']}",
            f"   exaggeration: {report.recommended_params['exaggeration']}",
            f"   temperature: {report.recommended_params['temperature']}",
            f"   top_p: {report.recommended_params['top_p']}",
            "",
            "=" * 70,
            "🎯 KEY INSIGHT FOR BETTER SIMILARITY:",
            "=" * 70,
            "",
            "ChatterBox is ZERO-SHOT - it doesn't learn your voice,",
            "it extracts features from ONE reference audio at a time.",
            "",
            "For BEST similarity:",
            "1. Use your SINGLE BEST sample (not all 30!)",
            "2. That sample should be 10-15 seconds of CLEAR speech",
            "3. Should have minimal background noise",
            "4. Speaking style should match what you want to generate",
            "   (e.g., conversational reference → conversational output)",
            "",
            "CRITICAL: Your WhatsApp voice notes likely have:",
            "- Phone compression artifacts",
            "- Background noise",
            "- Variable audio quality",
            "- Different emotional states/speaking styles",
            "",
            "For podcast-quality output, you need podcast-quality input!",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def main():
    """Run voice similarity analysis on pritam profile"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze voice profile similarity")
    parser.add_argument("--profile", default="voice_profiles/pritam",
                       help="Path to voice profile directory")
    parser.add_argument("--find-best", action="store_true",
                       help="Find the single best reference sample")
    parser.add_argument("--compare", nargs=2, metavar=("SAMPLE1", "SAMPLE2"),
                       help="Compare two audio samples")
    args = parser.parse_args()
    
    analyzer = VoiceSimilarityAnalyzer()
    
    if args.compare:
        similarity = analyzer.compare_embeddings(args.compare[0], args.compare[1])
        print(f"\n📊 Similarity between samples: {similarity:.4f}")
        print(f"   (1.0 = identical, 0.0 = completely different)")
        return
    
    profile_path = Path(__file__).parent / args.profile
    
    if args.find_best:
        best_sample, score = analyzer.find_best_single_sample(str(profile_path))
        print(f"\n🏆 BEST SINGLE SAMPLE:")
        print(f"   File: {best_sample}")
        print(f"   Score: {score:.1f}")
        print(f"\n💡 Use this SINGLE file as your reference audio!")
        return
    
    # Full analysis
    report = analyzer.analyze_profile(str(profile_path))
    
    # Print report
    report_text = analyzer.generate_report(report)
    print(report_text)
    
    # Save report
    report_file = profile_path / "similarity_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"\n📝 Report saved to: {report_file}")
    
    # Also save as JSON for programmatic use
    json_report = {
        "profile_name": report.profile_name,
        "num_samples": report.num_samples,
        "best_samples": report.best_samples,
        "worst_samples": report.worst_samples,
        "overall_quality": report.overall_quality,
        "embedding_consistency": report.embedding_consistency,
        "recommended_params": report.recommended_params,
        "issues": report.issues,
        "recommendations": report.recommendations
    }
    
    json_file = profile_path / "similarity_report.json"
    with open(json_file, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"📝 JSON report saved to: {json_file}")


if __name__ == "__main__":
    main()
