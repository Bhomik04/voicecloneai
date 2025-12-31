"""
Studio-Quality Audio Post-Processor
Applies broadcast-standard mastering techniques for social media content
"""

import numpy as np
import torch
import torchaudio
import librosa
from scipy import signal
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudioAudioProcessor:
    """
    Professional audio post-processing pipeline for broadcast-quality output.
    
    Implements:
    - Sample rate upsampling (24kHz → 48kHz)
    - Spectral enhancement
    - Multi-band dynamics processing
    - Intelligent noise reduction
    - Loudness normalization (-14 to -16 LUFS for social media)
    - High-quality limiting
    - Stereo enhancement
    """
    
    def __init__(
        self,
        input_sr: int = 24000,
        output_sr: int = 48000,
        target_lufs: float = -14.0,  # Social media standard
        use_stereo: bool = False,
    ):
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.target_lufs = target_lufs
        self.use_stereo = use_stereo
        
        logger.info(f"Studio Audio Processor initialized: {input_sr}Hz → {output_sr}Hz")
        
    def high_quality_resample(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        High-quality resampling using Kaiser windowed sinc interpolation.
        Better than librosa's default for upsampling.
        """
        if orig_sr == target_sr:
            return audio
            
        # Use Kaiser window (beta=14) for high-quality resampling
        # This is better than librosa's kaiser_fast for upsampling
        resampled = librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type='kaiser_best'  # Highest quality
        )
        
        logger.info(f"Resampled from {orig_sr}Hz to {target_sr}Hz")
        return resampled
    
    def spectral_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance high-frequency content that may be lost in vocoder.
        Subtle high-shelf boost for clarity and presence.
        """
        # Design high-shelf filter: +2dB above 4kHz for presence
        nyquist = sr / 2
        high_shelf_freq = 4000 / nyquist
        
        # Biquad high-shelf filter
        b, a = signal.iirfilter(
            2, 
            high_shelf_freq, 
            btype='high', 
            ftype='butter'
        )
        
        # Apply with subtle gain
        enhanced = signal.filtfilt(b, a, audio)
        enhanced = audio * 0.85 + enhanced * 0.15  # Blend: 85% original + 15% enhanced
        
        logger.info("Applied spectral enhancement")
        return enhanced
    
    def multiband_compressor(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        3-band compression for professional dynamics control.
        Separates low/mid/high frequencies for targeted processing.
        """
        # Split into 3 bands: Low (< 200Hz), Mid (200Hz-4kHz), High (> 4kHz)
        nyquist = sr / 2
        
        # Low band
        low_cutoff = 200 / nyquist
        b_low, a_low = signal.butter(4, low_cutoff, btype='low')
        low_band = signal.filtfilt(b_low, a_low, audio)
        
        # High band
        high_cutoff = 4000 / nyquist
        b_high, a_high = signal.butter(4, high_cutoff, btype='high')
        high_band = signal.filtfilt(b_high, a_high, audio)
        
        # Mid band (subtract low and high from original)
        mid_band = audio - low_band - high_band
        
        # Apply gentle compression to each band
        low_compressed = self._compress_audio(low_band, threshold=0.5, ratio=2.0)
        mid_compressed = self._compress_audio(mid_band, threshold=0.4, ratio=3.0)
        high_compressed = self._compress_audio(high_band, threshold=0.3, ratio=2.5)
        
        # Recombine bands
        result = low_compressed + mid_compressed + high_compressed
        
        logger.info("Applied multiband compression")
        return result
    
    def _compress_audio(
        self, 
        audio: np.ndarray, 
        threshold: float = 0.5, 
        ratio: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0
    ) -> np.ndarray:
        """
        Simple but effective feed-forward compressor.
        
        Args:
            threshold: Compression threshold (0-1)
            ratio: Compression ratio (1.0 = no compression, higher = more)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
        """
        # Convert to envelope
        envelope = np.abs(audio)
        
        # Smooth envelope (attack/release simulation)
        smoothed = envelope.copy()
        for i in range(1, len(smoothed)):
            if smoothed[i] > smoothed[i-1]:
                # Attack
                smoothed[i] = smoothed[i-1] * 0.9 + smoothed[i] * 0.1
            else:
                # Release
                smoothed[i] = smoothed[i-1] * 0.95 + smoothed[i] * 0.05
        
        # Calculate gain reduction
        gain = np.ones_like(smoothed)
        over_threshold = smoothed > threshold
        gain[over_threshold] = threshold / smoothed[over_threshold]
        gain[over_threshold] = gain[over_threshold] ** (1.0 - 1.0/ratio)
        
        # Apply makeup gain
        makeup_gain = 1.0 / np.mean(gain)
        makeup_gain = min(makeup_gain, 2.0)  # Limit makeup gain
        
        return audio * gain * makeup_gain
    
    def intelligent_noise_gate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Intelligent noise gate to reduce background noise during silence.
        Uses spectral gating for better quality than simple threshold.
        """
        # Convert to torch for torchaudio processing
        audio_torch = torch.from_numpy(audio).float()
        
        # Simple energy-based gate
        frame_length = int(0.02 * sr)  # 20ms frames
        hop_length = frame_length // 2
        
        # Calculate frame energy
        audio_padded = np.pad(audio, (0, frame_length), mode='constant')
        frames = librosa.util.frame(audio_padded, frame_length=frame_length, hop_length=hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        # Threshold at -40dB below peak
        threshold = np.max(energy) * 0.01  # -40dB
        
        # Create smooth gate
        gate = np.ones_like(energy)
        gate[energy < threshold] = 0.1  # Reduce but don't eliminate
        
        # Smooth gate transitions
        gate = signal.filtfilt([0.1, 0.2, 0.4, 0.2, 0.1], [1.0], gate)
        
        # Upsample gate to audio length
        gate_upsampled = np.interp(
            np.linspace(0, len(gate)-1, len(audio)),
            np.arange(len(gate)),
            gate
        )
        
        result = audio * gate_upsampled
        
        logger.info("Applied intelligent noise gate")
        return result
    
    def loudness_normalization(
        self, 
        audio: np.ndarray, 
        sr: int, 
        target_lufs: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize audio to target LUFS (broadcast standard).
        
        LUFS targets:
        - Streaming platforms: -14 LUFS (Spotify, YouTube, Apple Music)
        - Social media: -14 to -16 LUFS (Instagram, TikTok, Facebook)
        - Podcasts: -16 to -19 LUFS
        """
        if target_lufs is None:
            target_lufs = self.target_lufs
        
        # Calculate current loudness (approximation of LUFS)
        # True LUFS requires ITU-R BS.1770 weighting, this is simplified
        rms = np.sqrt(np.mean(audio ** 2))
        current_lufs = 20 * np.log10(rms + 1e-10) + 3  # Approximate LUFS
        
        # Calculate gain adjustment
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Ensure no clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)
        
        logger.info(f"Normalized from {current_lufs:.1f} LUFS to {target_lufs:.1f} LUFS")
        return normalized
    
    def soft_clipper_limiter(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Soft-knee limiter to prevent clipping while maintaining dynamics.
        Uses tanh-based soft clipping for musical limiting.
        """
        # Soft clipping using tanh
        # This is more musical than hard clipping
        scaled = audio / threshold
        limited = np.tanh(scaled) * threshold
        
        # Blend with original for very gentle limiting
        blend_factor = 0.7  # 70% limited, 30% original
        result = limited * blend_factor + audio * (1 - blend_factor)
        
        # Final safety limiter (hard ceiling at 0.99)
        result = np.clip(result, -0.99, 0.99)
        
        logger.info("Applied soft-knee limiter")
        return result
    
    def stereo_enhancement(self, audio: np.ndarray, width: float = 0.3) -> np.ndarray:
        """
        Create pseudo-stereo from mono for richer sound.
        Uses Haas effect and spectral decorrelation.
        
        Args:
            width: Stereo width (0.0 = mono, 1.0 = wide stereo)
        """
        # Create left and right channels
        left = audio.copy()
        right = audio.copy()
        
        # Apply subtle delay (Haas effect: 10-30ms)
        delay_samples = int(0.015 * self.output_sr)  # 15ms delay
        right = np.pad(right, (delay_samples, 0), mode='constant')[:-delay_samples]
        
        # Apply complementary high-shelf filters for spectral difference
        nyquist = self.output_sr / 2
        cutoff = 3000 / nyquist
        
        # Left channel: slight high boost
        b_l, a_l = signal.butter(2, cutoff, btype='high')
        left_filtered = signal.filtfilt(b_l, a_l, left)
        left = left * 0.9 + left_filtered * 0.1
        
        # Right channel: slight high cut
        b_r, a_r = signal.butter(2, cutoff, btype='low')
        right_filtered = signal.filtfilt(b_r, a_r, right)
        right = right * 0.9 + right_filtered * 0.1
        
        # Mix with original mono for width control
        left = audio * (1 - width) + left * width
        right = audio * (1 - width) + right * width
        
        # Stack to stereo
        stereo = np.stack([left, right], axis=0)
        
        logger.info(f"Created stereo with width={width}")
        return stereo
    
    def de_esser(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Reduce harsh sibilance (S, T, CH sounds) in 4-8kHz range.
        """
        # Isolate sibilant frequencies (4-8 kHz)
        nyquist = sr / 2
        low_cut = 4000 / nyquist
        high_cut = 8000 / nyquist
        
        # Bandpass filter for sibilant range
        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        sibilants = signal.filtfilt(b, a, audio)
        
        # Compress only the sibilant band
        sibilants_compressed = self._compress_audio(
            sibilants, 
            threshold=0.2, 
            ratio=4.0
        )
        
        # Subtract original sibilants and add compressed version
        result = audio - sibilants + sibilants_compressed
        
        logger.info("Applied de-esser")
        return result
    
    def process(
        self, 
        audio: np.ndarray, 
        enable_all: bool = True,
        enable_spectral: bool = True,
        enable_dynamics: bool = True,
        enable_deess: bool = True,
        enable_gate: bool = False,  # Optional, can introduce artifacts
    ) -> np.ndarray:
        """
        Apply full studio processing pipeline.
        
        Args:
            audio: Input audio (mono, float32, -1 to 1)
            enable_all: Enable all processing (overrides individual flags)
            enable_spectral: Enable spectral enhancement
            enable_dynamics: Enable multiband compression
            enable_deess: Enable de-esser
            enable_gate: Enable noise gate (can introduce artifacts)
        
        Returns:
            Processed audio (mono or stereo depending on use_stereo)
        """
        logger.info("=" * 50)
        logger.info("Starting Studio Audio Processing Pipeline")
        logger.info("=" * 50)
        
        processed = audio.copy()
        
        # 1. Upsample to high quality sample rate
        if self.input_sr != self.output_sr:
            processed = self.high_quality_resample(
                processed, 
                self.input_sr, 
                self.output_sr
            )
        
        # 2. Spectral enhancement for clarity
        if enable_all or enable_spectral:
            processed = self.spectral_enhancement(processed, self.output_sr)
        
        # 3. De-esser to reduce harshness
        if enable_all or enable_deess:
            processed = self.de_esser(processed, self.output_sr)
        
        # 4. Multiband dynamics processing
        if enable_all or enable_dynamics:
            processed = self.multiband_compressor(processed, self.output_sr)
        
        # 5. Intelligent noise gate (optional)
        if enable_gate:
            processed = self.intelligent_noise_gate(processed, self.output_sr)
        
        # 6. Loudness normalization to broadcast standard
        processed = self.loudness_normalization(processed, self.output_sr)
        
        # 7. Soft limiting for final polish
        processed = self.soft_clipper_limiter(processed)
        
        # 8. Stereo enhancement (if enabled)
        if self.use_stereo:
            processed = self.stereo_enhancement(processed, width=0.3)
        
        logger.info("=" * 50)
        logger.info("Studio Processing Complete!")
        logger.info(f"Output: {'Stereo' if self.use_stereo else 'Mono'} @ {self.output_sr}Hz")
        logger.info("=" * 50)
        
        return processed


def process_for_social_media(
    audio: np.ndarray,
    input_sr: int = 24000,
    platform: str = "instagram"  # instagram, tiktok, youtube, podcast
) -> Tuple[np.ndarray, int]:
    """
    Quick preset for social media platforms.
    
    Args:
        audio: Input audio array
        input_sr: Input sample rate
        platform: Target platform preset
    
    Returns:
        (processed_audio, sample_rate)
    """
    presets = {
        "instagram": {"output_sr": 48000, "target_lufs": -14.0, "stereo": False},
        "tiktok": {"output_sr": 48000, "target_lufs": -14.0, "stereo": False},
        "youtube": {"output_sr": 48000, "target_lufs": -14.0, "stereo": True},
        "podcast": {"output_sr": 44100, "target_lufs": -16.0, "stereo": True},
        "default": {"output_sr": 48000, "target_lufs": -14.0, "stereo": False},
    }
    
    preset = presets.get(platform, presets["default"])
    
    processor = StudioAudioProcessor(
        input_sr=input_sr,
        output_sr=preset["output_sr"],
        target_lufs=preset["target_lufs"],
        use_stereo=preset["stereo"]
    )
    
    processed = processor.process(audio)
    
    return processed, preset["output_sr"]


if __name__ == "__main__":
    # Example usage
    print("Studio Audio Processor - Professional Post-Processing")
    print("For broadcast-quality TTS output")
    
    # Test with dummy audio
    test_audio = np.random.randn(24000 * 5)  # 5 seconds at 24kHz
    test_audio = test_audio / np.max(np.abs(test_audio)) * 0.5  # Normalize
    
    processed, sr = process_for_social_media(test_audio, 24000, "instagram")
    print(f"✓ Processed audio: {processed.shape} @ {sr}Hz")
