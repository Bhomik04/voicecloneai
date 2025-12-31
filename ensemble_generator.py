"""
Ensemble Generator - Multi-Variant Voice Synthesis
===================================================

Generates multiple variants of the same text and intelligently
combines the best segments for optimal quality.

Key features:
- Generate N variants with different random seeds
- Quality scoring per segment
- Intelligent blending of best segments
- Consistency checking across variants
"""

import torch
import torchaudio as ta
import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
import random


@dataclass
class GenerationVariant:
    """A single generation variant"""
    audio: torch.Tensor
    sample_rate: int
    seed: int
    quality_score: float = 0.0
    segment_scores: List[float] = None
    
    def __post_init__(self):
        if self.segment_scores is None:
            self.segment_scores = []


class QualityScorer:
    """
    Scores audio quality for comparison between variants
    
    Uses multiple heuristics:
    - SNR estimation
    - Spectral flatness (naturalness)
    - Energy consistency
    - Artifact detection
    """
    
    def __init__(self):
        print("📊 Quality Scorer initialized")
        
    def score(self, audio: torch.Tensor, sr: int) -> float:
        """
        Calculate overall quality score for audio
        
        Args:
            audio: Audio tensor [channels, samples]
            sr: Sample rate
            
        Returns:
            Quality score (0-100)
        """
        scores = []
        
        # 1. Energy consistency score
        energy_score = self._score_energy_consistency(audio)
        scores.append(energy_score * 0.25)
        
        # 2. Spectral quality score
        spectral_score = self._score_spectral_quality(audio, sr)
        scores.append(spectral_score * 0.35)
        
        # 3. Artifact detection score
        artifact_score = self._score_artifacts(audio, sr)
        scores.append(artifact_score * 0.25)
        
        # 4. Dynamic range score
        dynamics_score = self._score_dynamics(audio)
        scores.append(dynamics_score * 0.15)
        
        return sum(scores)
    
    def _score_energy_consistency(self, audio: torch.Tensor) -> float:
        """Score based on energy consistency (no sudden drops/spikes)"""
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
            
        # Calculate energy in windows
        window_size = 1024
        hop_size = 512
        
        energies = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energy = (window ** 2).mean().item()
            energies.append(energy)
            
        if len(energies) < 2:
            return 50.0
            
        energies = np.array(energies)
        
        # Calculate variance of energy (lower is more consistent)
        if energies.mean() == 0:
            return 50.0
            
        cv = energies.std() / (energies.mean() + 1e-10)  # Coefficient of variation
        
        # Map CV to score (lower CV = higher score)
        # Typical CV for good speech: 0.5-1.5
        if cv < 0.3:
            return 95.0  # Very consistent
        elif cv < 1.0:
            return 85.0 - (cv - 0.3) * 20
        elif cv < 2.0:
            return 65.0 - (cv - 1.0) * 30
        else:
            return max(20.0, 35.0 - (cv - 2.0) * 10)
            
    def _score_spectral_quality(self, audio: torch.Tensor, sr: int) -> float:
        """Score based on spectral characteristics"""
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
            
        # Compute spectrogram
        n_fft = 1024
        hop_length = 256
        
        # Use torch STFT
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=audio.device),
            return_complex=True,
        )
        
        magnitude = spec.abs()
        
        # Calculate spectral flatness
        # Higher flatness = more noise-like, lower = more tonal
        geometric_mean = torch.exp(torch.log(magnitude + 1e-10).mean(dim=0))
        arithmetic_mean = magnitude.mean(dim=0) + 1e-10
        flatness = (geometric_mean / arithmetic_mean).mean().item()
        
        # Speech should have moderate flatness (not too flat, not too peaky)
        # Typical range: 0.01-0.3
        if 0.05 < flatness < 0.25:
            flatness_score = 90.0
        elif 0.02 < flatness < 0.35:
            flatness_score = 70.0
        else:
            flatness_score = 50.0
            
        # Check for frequency coverage (speech should have energy 100-8000 Hz)
        freq_bins = n_fft // 2 + 1
        freq_per_bin = sr / n_fft
        
        # Bins for speech range
        low_bin = int(100 / freq_per_bin)
        high_bin = min(int(8000 / freq_per_bin), freq_bins - 1)
        
        speech_energy = magnitude[low_bin:high_bin, :].mean().item()
        total_energy = magnitude.mean().item() + 1e-10
        
        speech_ratio = speech_energy / total_energy
        
        if speech_ratio > 0.7:
            coverage_score = 90.0
        elif speech_ratio > 0.5:
            coverage_score = 70.0
        else:
            coverage_score = 50.0
            
        return (flatness_score + coverage_score) / 2
    
    def _score_artifacts(self, audio: torch.Tensor, sr: int) -> float:
        """Score based on artifact detection (clicks, pops, buzzing)"""
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
            
        score = 100.0
        
        # 1. Check for clicks (sudden amplitude changes)
        diff = torch.diff(audio)
        large_jumps = (diff.abs() > 0.5).sum().item()
        jump_ratio = large_jumps / len(diff)
        
        if jump_ratio > 0.01:  # More than 1% sudden jumps
            score -= min(30, jump_ratio * 1000)
            
        # 2. Check for silence gaps (longer than 500ms)
        silence_threshold = 0.01
        silence_samples = int(0.5 * sr)
        
        is_silent = audio.abs() < silence_threshold
        
        # Find runs of silence
        silence_runs = 0
        current_run = 0
        for s in is_silent:
            if s:
                current_run += 1
            else:
                if current_run > silence_samples:
                    silence_runs += 1
                current_run = 0
                
        if silence_runs > 2:
            score -= min(20, silence_runs * 5)
            
        # 3. Check for clipping
        clipping = (audio.abs() > 0.99).sum().item()
        clip_ratio = clipping / len(audio)
        
        if clip_ratio > 0.001:  # More than 0.1% clipping
            score -= min(25, clip_ratio * 10000)
            
        return max(20.0, score)
    
    def _score_dynamics(self, audio: torch.Tensor) -> float:
        """Score based on dynamic range"""
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
            
        # Calculate RMS in dB
        rms = torch.sqrt((audio ** 2).mean()).item()
        
        if rms == 0:
            return 30.0
            
        rms_db = 20 * np.log10(rms)
        
        # Calculate peak
        peak = audio.abs().max().item()
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Dynamic range
        crest_factor = peak_db - rms_db
        
        # Good speech has crest factor around 10-20 dB
        if 8 < crest_factor < 20:
            return 90.0
        elif 5 < crest_factor < 25:
            return 70.0
        else:
            return 50.0
            
    def score_segment(
        self,
        audio: torch.Tensor,
        sr: int,
        start_sample: int,
        end_sample: int,
    ) -> float:
        """Score a specific segment of audio"""
        segment = audio[..., start_sample:end_sample]
        return self.score(segment, sr)


class EnsembleGenerator:
    """
    Multi-variant generation and intelligent blending
    
    Generates multiple versions of the same text and combines
    the best segments from each for optimal quality.
    """
    
    def __init__(
        self,
        n_variants: int = 3,
        segment_duration: float = 2.0,  # seconds
        crossfade_duration: float = 0.1,  # seconds
    ):
        """
        Initialize ensemble generator
        
        Args:
            n_variants: Number of variants to generate
            segment_duration: Duration of each segment for scoring
            crossfade_duration: Duration of crossfade between segments
        """
        self.n_variants = n_variants
        self.segment_duration = segment_duration
        self.crossfade_duration = crossfade_duration
        
        self.scorer = QualityScorer()
        
        print(f"🎼 Ensemble Generator initialized (n_variants={n_variants})")
        
    def generate_variants(
        self,
        generate_fn: Callable[[int], Tuple[torch.Tensor, int]],
        seeds: Optional[List[int]] = None,
    ) -> List[GenerationVariant]:
        """
        Generate multiple variants using provided generation function
        
        Args:
            generate_fn: Function that takes seed and returns (audio, sample_rate)
            seeds: Optional list of seeds (random if not provided)
            
        Returns:
            List of GenerationVariant objects
        """
        if seeds is None:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(self.n_variants)]
            
        variants = []
        
        print(f"🎲 Generating {len(seeds)} variants...")
        
        for i, seed in enumerate(seeds):
            print(f"  Variant {i+1}/{len(seeds)} (seed={seed})...")
            
            # Generate audio
            audio, sr = generate_fn(seed)
            
            # Score quality
            quality_score = self.scorer.score(audio, sr)
            
            # Score segments
            segment_scores = self._score_segments(audio, sr)
            
            variant = GenerationVariant(
                audio=audio,
                sample_rate=sr,
                seed=seed,
                quality_score=quality_score,
                segment_scores=segment_scores,
            )
            
            variants.append(variant)
            print(f"    Quality score: {quality_score:.1f}")
            
        return variants
    
    def _score_segments(self, audio: torch.Tensor, sr: int) -> List[float]:
        """Score audio in segments"""
        segment_samples = int(self.segment_duration * sr)
        
        scores = []
        
        for start in range(0, audio.shape[-1], segment_samples):
            end = min(start + segment_samples, audio.shape[-1])
            
            if end - start < segment_samples // 4:
                # Skip very short final segments
                if scores:
                    scores.append(scores[-1])  # Use previous score
                else:
                    scores.append(50.0)
            else:
                segment_score = self.scorer.score_segment(audio, sr, start, end)
                scores.append(segment_score)
                
        return scores
    
    def select_best(self, variants: List[GenerationVariant]) -> GenerationVariant:
        """
        Select the best single variant
        
        Args:
            variants: List of generated variants
            
        Returns:
            Best variant based on quality score
        """
        return max(variants, key=lambda v: v.quality_score)
    
    def blend_best_segments(
        self,
        variants: List[GenerationVariant],
    ) -> Tuple[torch.Tensor, int]:
        """
        Blend the best segments from all variants
        
        Args:
            variants: List of generated variants
            
        Returns:
            Blended audio tensor and sample rate
        """
        if not variants:
            raise ValueError("No variants provided")
            
        sr = variants[0].sample_rate
        segment_samples = int(self.segment_duration * sr)
        crossfade_samples = int(self.crossfade_duration * sr)
        
        # Get number of segments (use shortest variant)
        min_segments = min(len(v.segment_scores) for v in variants)
        
        if min_segments == 0:
            return variants[0].audio, sr
            
        print(f"🎨 Blending best segments from {len(variants)} variants...")
        
        # Select best segment from each position
        blended_segments = []
        
        for seg_idx in range(min_segments):
            # Find variant with best score for this segment
            best_variant = max(
                variants,
                key=lambda v: v.segment_scores[seg_idx] if seg_idx < len(v.segment_scores) else 0,
            )
            
            # Extract segment
            start = seg_idx * segment_samples
            end = min(start + segment_samples, best_variant.audio.shape[-1])
            
            segment = best_variant.audio[..., start:end]
            blended_segments.append(segment)
            
            print(f"  Segment {seg_idx+1}: using variant with seed {best_variant.seed} "
                  f"(score: {best_variant.segment_scores[seg_idx]:.1f})")
                  
        # Concatenate with crossfade
        if len(blended_segments) == 1:
            return blended_segments[0], sr
            
        blended = self._concatenate_with_crossfade(
            blended_segments, crossfade_samples
        )
        
        return blended, sr
    
    def _concatenate_with_crossfade(
        self,
        segments: List[torch.Tensor],
        crossfade_samples: int,
    ) -> torch.Tensor:
        """Concatenate segments with crossfade"""
        if not segments:
            return torch.zeros(1, 0)
            
        # Calculate total length
        total_length = sum(s.shape[-1] for s in segments)
        total_length -= crossfade_samples * (len(segments) - 1)
        
        # Ensure consistent shape
        n_channels = segments[0].shape[0] if segments[0].dim() == 2 else 1
        
        result = torch.zeros(n_channels, total_length)
        
        current_pos = 0
        
        for i, segment in enumerate(segments):
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)
                
            seg_len = segment.shape[-1]
            
            if i == 0:
                # First segment - no fade in
                result[:, :seg_len] = segment
                current_pos = seg_len - crossfade_samples
            else:
                # Create crossfade
                fade_in = torch.linspace(0, 1, crossfade_samples)
                fade_out = torch.linspace(1, 0, crossfade_samples)
                
                # Apply crossfade in overlap region
                overlap_start = current_pos
                
                # Fade out previous
                result[:, overlap_start:overlap_start + crossfade_samples] *= fade_out
                
                # Fade in and add current
                if crossfade_samples > 0:
                    result[:, overlap_start:overlap_start + crossfade_samples] += \
                        segment[:, :crossfade_samples] * fade_in
                        
                # Add rest of segment
                remaining = seg_len - crossfade_samples
                if remaining > 0:
                    end_pos = overlap_start + seg_len
                    result[:, overlap_start + crossfade_samples:end_pos] = \
                        segment[:, crossfade_samples:]
                        
                current_pos = overlap_start + seg_len - crossfade_samples
                
        # Trim any excess
        result = result[:, :current_pos + crossfade_samples]
        
        return result
    
    def generate_and_blend(
        self,
        generate_fn: Callable[[int], Tuple[torch.Tensor, int]],
        blend_mode: str = "best_single",
    ) -> Tuple[torch.Tensor, int]:
        """
        Convenience method to generate variants and return blended result
        
        Args:
            generate_fn: Generation function (takes seed, returns audio and sr)
            blend_mode: "best_single" or "best_segments"
            
        Returns:
            Best audio tensor and sample rate
        """
        variants = self.generate_variants(generate_fn)
        
        if blend_mode == "best_single":
            best = self.select_best(variants)
            print(f"✅ Selected variant with seed {best.seed} (score: {best.quality_score:.1f})")
            return best.audio, best.sample_rate
            
        elif blend_mode == "best_segments":
            return self.blend_best_segments(variants)
            
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")


def test_ensemble_generator():
    """Test ensemble generation with synthetic audio"""
    print("🎼 Testing Ensemble Generator\n")
    
    # Create test generation function
    def mock_generate(seed: int) -> Tuple[torch.Tensor, int]:
        """Generate synthetic audio for testing"""
        torch.manual_seed(seed)
        
        sr = 22050
        duration = 3.0
        samples = int(sr * duration)
        
        # Generate sine wave with some noise
        t = torch.linspace(0, duration, samples)
        freq = 440 + torch.randn(1).item() * 50  # Vary frequency
        
        audio = torch.sin(2 * np.pi * freq * t)
        
        # Add noise (varies by seed)
        noise_level = 0.05 + torch.rand(1).item() * 0.1
        audio += torch.randn(samples) * noise_level
        
        # Add random artifacts to some
        if seed % 3 == 0:
            # Add clicks
            click_positions = torch.randint(0, samples, (5,))
            audio[click_positions] = torch.randn(5) * 0.8
            
        audio = audio.unsqueeze(0)  # Add channel dim
        
        return audio, sr
        
    # Test ensemble generator
    generator = EnsembleGenerator(n_variants=4)
    
    # Generate variants
    print("\n--- Generating Variants ---")
    variants = generator.generate_variants(
        mock_generate,
        seeds=[42, 123, 456, 789]
    )
    
    print("\n--- Variant Quality Scores ---")
    for i, v in enumerate(variants):
        print(f"  Variant {i+1} (seed={v.seed}): {v.quality_score:.1f}")
        
    # Test best single
    print("\n--- Best Single Selection ---")
    best = generator.select_best(variants)
    print(f"  Best: seed={best.seed}, score={best.quality_score:.1f}")
    
    # Test segment blending
    print("\n--- Segment Blending ---")
    blended, sr = generator.blend_best_segments(variants)
    print(f"  Blended audio shape: {blended.shape}")
    print(f"  Blended audio duration: {blended.shape[-1] / sr:.2f}s")
    
    print("\n✅ Ensemble generator test complete!")


if __name__ == "__main__":
    test_ensemble_generator()
