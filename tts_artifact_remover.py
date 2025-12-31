"""
TTS Artifact Remover - Removes synthetic/robotic quality from TTS output
Makes it sound like real human speech
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


class TTSArtifactRemover:
    """Remove TTS-specific artifacts that make audio sound synthetic"""
    
    def __init__(self, sr: int = 24000):
        self.sr = sr
    
    def remove_robotic_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove unnatural harmonics that TTS models add
        These sound like buzzing or synthetic overtones
        """
        # Use harmonic-percussive separation
        # TTS artifacts often appear in harmonic component as unnatural overtones
        harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
        
        # Reduce overly strong harmonics (TTS artifacts)
        # Keep 70% harmonic, 100% percussive
        cleaned = harmonic * 0.7 + percussive
        
        return cleaned
    
    def add_natural_breathiness(self, audio: np.ndarray) -> np.ndarray:
        """
        Add subtle breath noise that humans have but TTS lacks
        This makes it sound less synthetic
        """
        # Generate subtle pink noise (1/f noise - natural sounding)
        noise = np.random.randn(len(audio))
        
        # Apply 1/f spectrum (pink noise)
        fft_noise = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1/self.sr)
        # 1/f falloff
        fft_noise = fft_noise / (np.sqrt(freqs + 1))
        noise = np.fft.irfft(fft_noise, n=len(audio))
        
        # Normalize noise
        noise = noise / (np.max(np.abs(noise)) + 1e-10)
        
        # Only add noise where voice is present (not silence)
        envelope = np.abs(audio)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.01*self.sr))
        
        # Noise gate - only add where speech is active
        threshold = np.percentile(envelope_smooth, 30)
        mask = envelope_smooth > threshold
        
        # Very subtle breathiness (1-2% of signal)
        breath_amount = 0.015
        audio_with_breath = audio + (noise * mask * breath_amount)
        
        return audio_with_breath
    
    def remove_metallic_resonance(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove metallic/synthetic resonances in 2-4kHz range
        TTS models often have unnatural resonances here
        """
        # Notch filter for metallic frequencies
        # Sweep through 2-4kHz to remove synthetic resonances
        filtered = audio.copy()
        
        for center_freq in [2200, 2800, 3400]:
            # Create narrow notch
            sos = signal.butter(2, [center_freq-100, center_freq+100], 
                               btype='bandstop', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, filtered)
        
        # Blend 60% filtered + 40% original
        output = filtered * 0.6 + audio * 0.4
        
        return output
    
    def enhance_formants(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance natural voice formants while reducing synthetic ones
        Makes voice sound more human
        """
        # Boost natural voice formants (200-600 Hz)
        sos_formant = signal.butter(4, [200, 600], btype='band', fs=self.sr, output='sos')
        formants = signal.sosfilt(sos_formant, audio)
        
        # Reduce harsh upper formants (4-8kHz where TTS artifacts live)
        sos_harsh = signal.butter(2, [4000, 8000], btype='band', fs=self.sr, output='sos')
        harsh = signal.sosfilt(sos_harsh, audio)
        
        # Mix: boost formants +15%, reduce harsh -40%
        output = audio + (formants * 0.15) - (harsh * 0.4)
        
        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * (0.95 / max_val)
        
        return output
    
    def apply_tube_saturation(self, audio: np.ndarray, amount: float = 0.3) -> np.ndarray:
        """
        Subtle analog-style saturation - makes digital TTS sound less sterile
        Adds harmonic richness that real voices have
        """
        # Soft clipping (tube-like saturation)
        # tanh provides smooth saturation
        saturated = np.tanh(audio * (1 + amount))
        
        # Mix with dry signal
        output = saturated * 0.3 + audio * 0.7
        
        return output
    
    def add_natural_fluctuation(self, audio: np.ndarray) -> np.ndarray:
        """
        Add subtle pitch/volume fluctuations that humans have
        TTS is too perfect - this adds natural imperfection
        """
        # Very subtle amplitude modulation (natural voice variation)
        mod_freq = 0.5  # Hz - very slow fluctuation
        t = np.linspace(0, len(audio)/self.sr, len(audio))
        
        # Create subtle variation envelope
        mod = 1.0 + 0.02 * np.sin(2 * np.pi * mod_freq * t)
        
        # Apply variation
        return audio * mod
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete TTS artifact removal pipeline
        """
        print("   🔧 Removing robotic harmonics...")
        audio = self.remove_robotic_harmonics(audio)
        
        print("   🌬️  Adding natural breathiness...")
        audio = self.add_natural_breathiness(audio)
        
        print("   ✂️  Removing metallic resonances...")
        audio = self.remove_metallic_resonance(audio)
        
        print("   🎵 Enhancing natural formants...")
        audio = self.enhance_formants(audio)
        
        print("   📻 Applying tube saturation...")
        audio = self.apply_tube_saturation(audio, amount=0.2)
        
        print("   🌊 Adding natural fluctuation...")
        audio = self.add_natural_fluctuation(audio)
        
        # Final normalization
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-16/20)  # -16 dBFS
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        # Safety clip
        audio = np.clip(audio, -0.99, 0.99)
        
        return audio
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """Process a file"""
        try:
            print(f"\n🎙️  Processing: {Path(input_path).name}")
            print("="*70)
            
            # Load
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            
            # Process
            audio_clean = self.process(audio)
            
            # Save
            sf.write(output_path, audio_clean, sr, subtype='PCM_16')
            
            print(f"\n✅ Cleaned audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\n🎙️  TTS ARTIFACT REMOVER")
        print("="*70)
        print("Removes synthetic/robotic quality from TTS output")
        print("\nUsage:")
        print("  python tts_artifact_remover.py <input.wav> [output.wav]")
        print("\nExample:")
        print("  python tts_artifact_remover.py output.wav output_clean.wav")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.wav', '_humanized.wav')
    
    remover = TTSArtifactRemover()
    remover.process_file(input_file, output_file)


if __name__ == "__main__":
    main()
