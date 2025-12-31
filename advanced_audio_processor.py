"""
Advanced Audio Post-Processing for Natural Speech
Adds dynamics, prosody, and natural sentence pacing to TTS output
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import noisereduce as nr
except ImportError:
    import os
    print("Installing noisereduce...")
    os.system("pip install noisereduce")
    import noisereduce as nr

try:
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    import os
    print("Installing scipy...")
    os.system("pip install scipy")
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d


class NaturalSpeechProcessor:
    """
    Advanced processor that makes TTS output sound like natural human speech
    with proper dynamics, pacing, and prosody
    """
    
    def __init__(self, target_sr: int = 24000):
        self.target_sr = target_sr
    
    def detect_sentence_boundaries(self, audio: np.ndarray, sr: int) -> list:
        """
        Detect sentence boundaries using energy-based silence detection
        Returns list of (start, end) indices
        """
        # Calculate short-time energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute energy
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        # Smooth energy
        energy_smooth = gaussian_filter1d(energy, sigma=5)
        
        # Find silence regions (low energy)
        threshold = np.percentile(energy_smooth, 20)  # Bottom 20% = silence
        is_silence = energy_smooth < threshold
        
        # Find transitions from silence to speech (sentence starts)
        # and speech to silence (sentence ends)
        boundaries = []
        in_speech = False
        start_idx = 0
        
        for i, silent in enumerate(is_silence):
            frame_pos = i * hop_length
            
            if not silent and not in_speech:
                # Start of sentence
                start_idx = frame_pos
                in_speech = True
            elif silent and in_speech:
                # End of sentence
                if frame_pos - start_idx > sr * 0.3:  # At least 300ms
                    boundaries.append((start_idx, frame_pos))
                in_speech = False
        
        # Add final boundary if still in speech
        if in_speech:
            boundaries.append((start_idx, len(audio)))
        
        return boundaries
    
    def add_natural_dynamics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Add natural volume dynamics - louder at start, softer at end of sentences
        This mimics how humans naturally speak
        """
        # Detect sentence boundaries
        boundaries = self.detect_sentence_boundaries(audio, sr)
        
        if len(boundaries) == 0:
            return audio
        
        output = audio.copy()
        
        for start, end in boundaries:
            sentence = audio[start:end]
            length = end - start
            
            if length < sr * 0.1:  # Skip very short segments
                continue
            
            # Create natural dynamics envelope
            # Start: 100%, Middle: 105%, End: 85%
            t = np.linspace(0, 1, length)
            
            # Natural speech pattern: slight boost in middle, fade at end
            dynamics = np.ones_like(t)
            
            # Gradual fade-in at start (first 10%)
            fade_in_len = int(length * 0.1)
            if fade_in_len > 0:
                dynamics[:fade_in_len] = np.linspace(0.95, 1.0, fade_in_len)
            
            # Slight emphasis in middle
            mid_start = int(length * 0.2)
            mid_end = int(length * 0.7)
            if mid_end > mid_start:
                dynamics[mid_start:mid_end] *= 1.05
            
            # Natural fade at end (last 20%)
            fade_out_len = int(length * 0.2)
            if fade_out_len > 0:
                dynamics[-fade_out_len:] = np.linspace(1.0, 0.85, fade_out_len)
            
            # Apply dynamics
            output[start:end] = sentence * dynamics
        
        return output
    
    def add_sentence_pauses(self, audio: np.ndarray, sr: int, 
                           pause_duration: float = 0.15) -> np.ndarray:
        """
        Add natural pauses between sentences (like humans do)
        """
        boundaries = self.detect_sentence_boundaries(audio, sr)
        
        if len(boundaries) <= 1:
            return audio
        
        # Create output with pauses
        segments = []
        pause_samples = int(pause_duration * sr)
        pause = np.zeros(pause_samples)
        
        for i, (start, end) in enumerate(boundaries):
            # Add sentence
            segments.append(audio[start:end])
            
            # Add pause between sentences (not after last one)
            if i < len(boundaries) - 1:
                segments.append(pause)
        
        return np.concatenate(segments)
    
    def reduce_noise_aggressive(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Aggressive noise reduction specifically for TTS artifacts
        """
        # Multi-pass noise reduction
        # Pass 1: Remove stationary noise (background hiss)
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.95,  # Very aggressive
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
        
        # Pass 2: Remove non-stationary noise (clicks, pops)
        reduced = nr.reduce_noise(
            y=reduced,
            sr=sr,
            stationary=False,
            prop_decrease=0.7,
            time_constant_s=1.0
        )
        
        return reduced
    
    def apply_de_essing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Reduce harsh sibilance (S, SH sounds) that TTS often exaggerates
        """
        # High-pass filter to isolate sibilants (5-10kHz)
        sos_high = signal.butter(4, 5000, btype='high', fs=sr, output='sos')
        sibilants = signal.sosfilt(sos_high, audio)
        
        # Detect where sibilants are loud
        sibilant_envelope = np.abs(sibilants)
        sibilant_envelope = gaussian_filter1d(sibilant_envelope, sigma=int(0.01*sr))
        
        # Create de-essing gain (reduce by 6dB where sibilants are loud)
        threshold = np.percentile(sibilant_envelope, 95)
        gain = np.ones_like(audio)
        mask = sibilant_envelope > threshold
        gain[mask] = 0.5  # -6dB reduction
        
        # Smooth the gain to avoid artifacts
        gain = gaussian_filter1d(gain, sigma=int(0.005*sr))
        
        return audio * gain
    
    def apply_warmth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Add warmth by boosting low-mids and reducing harsh highs
        Makes voice sound less "synthetic" and more "human"
        """
        # Boost 200-500Hz (warmth, body)
        sos_low = signal.butter(2, [200, 500], btype='band', fs=sr, output='sos')
        low_mids = signal.sosfilt(sos_low, audio)
        
        # Reduce 8-12kHz (harshness)
        sos_harsh = signal.butter(2, [8000, 12000], btype='band', fs=sr, output='sos')
        harsh = signal.sosfilt(sos_harsh, audio)
        
        # Mix back
        output = audio + (low_mids * 0.3) - (harsh * 0.4)
        
        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * (0.95 / max_val)
        
        return output
    
    def apply_broadcast_limiter(self, audio: np.ndarray, 
                               threshold: float = -3.0,
                               ceiling: float = -0.5) -> np.ndarray:
        """
        Professional broadcast-style limiting
        Prevents clipping while maintaining dynamics
        """
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio_db)
        over_threshold = audio_db > threshold
        gain_reduction[over_threshold] = audio_db[over_threshold] - threshold
        
        # Apply soft knee
        ratio = 10  # 10:1 compression above threshold
        gain_reduction = gain_reduction / ratio
        
        # Smooth gain reduction (attack/release)
        gain_reduction_smooth = gaussian_filter1d(gain_reduction, sigma=int(0.003*self.target_sr))
        
        # Apply gain reduction
        output = audio * (10 ** (-gain_reduction_smooth / 20))
        
        # Hard ceiling
        output = np.clip(output, -10**(ceiling/20), 10**(ceiling/20))
        
        return output
    
    def process_tts_output(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Complete pipeline to make TTS sound natural
        """
        print("   🔊 Adding natural speech dynamics...")
        
        # 1. Aggressive noise reduction (remove TTS artifacts)
        audio = self.reduce_noise_aggressive(audio, sr)
        print("   ✓ Noise reduced")
        
        # 2. De-essing (reduce harsh S sounds)
        audio = self.apply_de_essing(audio, sr)
        print("   ✓ De-essed")
        
        # 3. Add warmth (less synthetic)
        audio = self.apply_warmth(audio, sr)
        print("   ✓ Warmth added")
        
        # 4. Add natural dynamics (volume variation)
        audio = self.add_natural_dynamics(audio, sr)
        print("   ✓ Natural dynamics added")
        
        # 5. Add sentence pauses
        audio = self.add_sentence_pauses(audio, sr, pause_duration=0.15)
        print("   ✓ Sentence pacing added")
        
        # 6. Normalize to professional level
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-18/20)  # -18 dBFS (podcast standard)
        if rms > 0:
            audio = audio * (target_rms / rms)
        print("   ✓ Normalized")
        
        # 7. Final broadcast limiting
        audio = self.apply_broadcast_limiter(audio, threshold=-6.0, ceiling=-1.0)
        print("   ✓ Limited")
        
        # 8. Final safety clip
        audio = np.clip(audio, -0.99, 0.99)
        
        return audio
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a TTS output file to sound natural
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            
            # Process
            audio_processed = self.process_tts_output(audio, sr)
            
            # Save
            sf.write(output_path, audio_processed, sr, subtype='PCM_16')
            
            return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False


def enhance_tts_output(input_path: str, output_path: Optional[str] = None):
    """
    Quick function to enhance TTS output
    """
    if output_path is None:
        output_path = input_path.replace('.wav', '_enhanced.wav')
    
    processor = NaturalSpeechProcessor()
    
    print(f"\n🎙️  Processing: {Path(input_path).name}")
    success = processor.process_file(input_path, output_path)
    
    if success:
        print(f"✅ Enhanced audio saved to: {output_path}")
    else:
        print(f"❌ Processing failed")
    
    return output_path if success else None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_audio_processor.py <input.wav> [output.wav]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    enhance_tts_output(input_file, output_file)
