"""
Audio Enhancement Pipeline for Professional Quality Voice Cloning
Implements ElevenLabs-style audio post-processing

Features:
- Noise reduction
- Dynamic range compression
- Clarity enhancement
- Loudness normalization
"""

import torch
import torchaudio as ta
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
from scipy import signal


class AudioEnhancer:
    """Professional audio enhancement pipeline"""
    
    def __init__(
        self,
        target_loudness: float = -16.0,  # LUFS (ElevenLabs standard)
        noise_reduce_strength: float = 0.6,  # 0.0-1.0
        compression_ratio: float = 4.0,  # Dynamic range compression
        clarity_boost: float = 0.3,  # High-frequency enhancement
    ):
        """
        Initialize audio enhancer
        
        Args:
            target_loudness: Target loudness in LUFS (-16 is broadcast standard)
            noise_reduce_strength: Strength of noise reduction (0-1)
            compression_ratio: Compression ratio for dynamics (higher = more compression)
            clarity_boost: Amount of high-frequency boost for clarity (0-1)
        """
        self.target_loudness = target_loudness
        self.noise_reduce_strength = noise_reduce_strength
        self.compression_ratio = compression_ratio
        self.clarity_boost = clarity_boost
        
        # Initialize loudness meter
        self.meter = pyln.Meter(24000)  # 24kHz sample rate (ChatterboxTTS)
        
    def enhance(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
        apply_noise_reduction: bool = True,
        apply_compression: bool = True,
        apply_clarity: bool = True,
        apply_normalization: bool = True,
    ) -> torch.Tensor:
        """
        Apply full enhancement pipeline
        
        Args:
            audio: Input audio tensor [channels, samples] or [samples]
            sample_rate: Audio sample rate
            apply_noise_reduction: Enable noise reduction
            apply_compression: Enable dynamic range compression
            apply_clarity: Enable clarity enhancement
            apply_normalization: Enable loudness normalization
            
        Returns:
            Enhanced audio tensor (same shape as input)
        """
        # Handle both torch tensors and numpy arrays
        import torch
        
        if isinstance(audio, np.ndarray):
            # Already numpy, just ensure correct shape
            audio_np = audio
            was_numpy = True
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]  # Add channel dimension
        else:
            # PyTorch tensor - convert to numpy
            was_numpy = False
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add channel dimension
            audio_np = audio.cpu().numpy()
        
        # 1. Noise Reduction
        if apply_noise_reduction:
            audio_np = self._reduce_noise(audio_np, sample_rate)
            
        # 2. Dynamic Range Compression
        if apply_compression:
            audio_np = self._compress_dynamics(audio_np, sample_rate)
            
        # 3. Clarity Enhancement
        if apply_clarity:
            audio_np = self._enhance_clarity(audio_np, sample_rate)
            
        # 4. Loudness Normalization
        if apply_normalization:
            audio_np = self._normalize_loudness(audio_np, sample_rate)
            
        # Return in original format
        if was_numpy:
            return audio_np
        else:
            # Convert back to torch
            enhanced = torch.from_numpy(audio_np).float()
            return enhanced
    
    def _reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply spectral noise reduction
        
        Uses noisereduce library with stationary noise estimation
        """
        # Process each channel separately
        enhanced_channels = []
        
        for ch in range(audio.shape[0]):
            # Estimate noise from first 0.5 seconds
            noise_sample = audio[ch, :int(0.5 * sample_rate)]
            
            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio[ch],
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=self.noise_reduce_strength,
                stationary=True,
            )
            
            enhanced_channels.append(reduced)
            
        return np.stack(enhanced_channels)
    
    def _compress_dynamics(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply dynamic range compression (like ElevenLabs)
        
        Reduces the difference between loud and quiet parts
        Makes speech more consistent and professional
        """
        threshold_db = -20.0  # Compress above this level
        ratio = self.compression_ratio
        attack_time = 0.005  # 5ms attack
        release_time = 0.1   # 100ms release
        
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Apply compression
        compressed = np.copy(audio)
        
        for ch in range(audio.shape[0]):
            # Simple compression (production would use proper envelope follower)
            mask = audio_db[ch] > threshold_db
            excess_db = audio_db[ch][mask] - threshold_db
            compressed_db = threshold_db + excess_db / ratio
            
            # Convert back to linear
            gain_reduction = 10 ** ((compressed_db - audio_db[ch][mask]) / 20)
            compressed[ch][mask] *= gain_reduction
            
        return compressed
    
    def _enhance_clarity(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Enhance high-frequency clarity (ElevenLabs technique)
        
        Boosts 4-8kHz range for speech intelligibility
        """
        # Design high-shelf filter for clarity boost
        # Boost 4-8kHz by specified amount
        sos = signal.butter(
            4,
            4000,
            btype='high',
            fs=sample_rate,
            output='sos'
        )
        
        enhanced = np.copy(audio)
        
        for ch in range(audio.shape[0]):
            # Apply filter
            boosted: np.ndarray = signal.sosfilt(sos, audio[ch])  # type: ignore
            
            # Mix with original
            enhanced[ch] = audio[ch] + self.clarity_boost * boosted
            
        return enhanced
    
    def _normalize_loudness(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Normalize to target loudness using ITU-R BS.1770-4 standard
        
        This is how professional audio is normalized (Netflix, YouTube, etc.)
        ElevenLabs uses -16 LUFS for their outputs
        """
        # Convert to mono for loudness measurement
        if audio.shape[0] > 1:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio[0]
            
        # Measure current loudness
        try:
            current_loudness = self.meter.integrated_loudness(mono)
        except ValueError:
            # Audio too quiet to measure, skip normalization
            return audio
            
        # Calculate required gain
        loudness_delta = self.target_loudness - current_loudness
        gain = 10 ** (loudness_delta / 20)
        
        # Apply gain
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)
            
        return normalized
    
    def enhance_streaming(
        self,
        audio_chunk: torch.Tensor,
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """
        Enhanced audio processing optimized for streaming/chunked generation
        
        Applies lighter processing suitable for real-time use
        """
        # For streaming, use lighter processing
        return self.enhance(
            audio_chunk,
            sample_rate,
            apply_noise_reduction=False,  # Skip for speed
            apply_compression=True,
            apply_clarity=True,
            apply_normalization=True,
        )


def test_audio_enhancer():
    """Test the audio enhancer on a sample file"""
    import os
    
    print("🎙️ Testing Audio Enhancer\n")
    
    # Check for test file
    test_files = [
        "voice_profiles/pritam/samples/sample_001.wav",
        "test_sample.wav",
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
            
    if not test_file:
        print("❌ No test audio file found")
        print("   Expected: voice_profiles/pritam/samples/sample_001.wav")
        return
        
    print(f"📂 Loading: {test_file}")
    
    # Load audio
    audio, sr = ta.load(test_file)
    print(f"   Original: {audio.shape} @ {sr}Hz")
    
    # Create enhancer
    enhancer = AudioEnhancer(
        target_loudness=-16.0,
        noise_reduce_strength=0.6,
        compression_ratio=4.0,
        clarity_boost=0.3,
    )
    
    # Enhance
    print("\n⚡ Enhancing audio...")
    enhanced = enhancer.enhance(audio, sr)
    print(f"   Enhanced: {enhanced.shape}")
    
    # Save comparison
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    original_path = os.path.join(output_dir, "original.wav")
    enhanced_path = os.path.join(output_dir, "enhanced.wav")
    
    ta.save(original_path, audio, sr)
    ta.save(enhanced_path, enhanced, sr)
    
    print(f"\n✅ Saved comparison:")
    print(f"   Original: {original_path}")
    print(f"   Enhanced: {enhanced_path}")
    print(f"\n💡 Listen to both files to hear the difference!")
    
    # Calculate some metrics
    original_rms = torch.sqrt(torch.mean(audio ** 2))
    enhanced_rms = torch.sqrt(torch.mean(enhanced ** 2))
    
    print(f"\n📊 Metrics:")
    print(f"   Original RMS: {original_rms:.4f}")
    print(f"   Enhanced RMS: {enhanced_rms:.4f}")
    print(f"   RMS Change: {enhanced_rms/original_rms:.2f}x")


if __name__ == "__main__":
    test_audio_enhancer()
