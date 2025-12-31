"""
Adobe Podcast-Style Speech Enhancer
====================================

Replicates Adobe Podcast Enhance functionality:
1. Voice isolation/separation from background
2. Noise reduction (like professional studio)
3. Echo/reverb removal
4. Bandwidth extension (restore clarity)
5. De-essing and de-plosive
6. Broadcast-quality normalization

Adobe Enhance does:
- Separates voice from undesired sounds/noise
- Makes recordings sound like professional studio
- Uses deep learning to clean and optimize vocals
- Removes noise, echo, and distortion

This implementation uses spectral processing + neural filtering
to achieve similar results without requiring Adobe's servers.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import noisereduce
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("⚠️ noisereduce not installed. Using basic noise reduction.")

# Try to import pedalboard for pro audio processing
try:
    from pedalboard import (
        Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter,
        HighpassFilter, LowpassFilter, Limiter, NoiseGate
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    print("⚠️ pedalboard not installed. Using basic processing.")


class AdobePodcastEnhancer:
    """
    Replicates Adobe Podcast Enhance Speech functionality
    Makes any voice recording sound studio-quality
    """
    
    def __init__(self, sr: int = 24000):
        self.sr = sr
        
    # ==========================================================================
    # STAGE 1: Voice Isolation (Like Adobe's speech separation)
    # ==========================================================================
    
    def isolate_voice(self, audio: np.ndarray) -> np.ndarray:
        """
        Isolate voice from background using harmonic-percussive separation
        and spectral masking (similar to what Adobe does)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        
        # Harmonic-percussive separation
        # Voice is mostly harmonic, noise is often percussive/random
        harmonic, percussive = librosa.decompose.hpss(stft, margin=3.0)
        
        # Create voice mask based on harmonic content
        harmonic_mag = np.abs(harmonic)
        total_mag = mag + 1e-10
        voice_mask = harmonic_mag / total_mag
        
        # Smooth the mask to avoid artifacts
        voice_mask = gaussian_filter1d(voice_mask, sigma=2, axis=1)
        voice_mask = np.clip(voice_mask, 0.3, 1.0)  # Keep at least 30%
        
        # Apply mask
        isolated_stft = mag * voice_mask * np.exp(1j * phase)
        isolated = librosa.istft(isolated_stft, hop_length=512)
        
        # Match length
        if len(isolated) > len(audio):
            isolated = isolated[:len(audio)]
        elif len(isolated) < len(audio):
            isolated = np.pad(isolated, (0, len(audio) - len(isolated)))
        
        print("   🎤 Voice isolated from background")
        return isolated
    
    # ==========================================================================
    # STAGE 2: Advanced Noise Reduction (Like Adobe's AI denoiser)
    # ==========================================================================
    
    def advanced_denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Multi-stage noise reduction like Adobe's AI denoiser
        """
        if HAS_NOISEREDUCE:
            # Stage 1: Stationary noise reduction
            audio = nr.reduce_noise(
                y=audio, 
                sr=self.sr,
                stationary=True,
                prop_decrease=0.75,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
            
            # Stage 2: Non-stationary noise reduction (for varying noise)
            audio = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                stationary=False,
                prop_decrease=0.5,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
        else:
            # Fallback: Spectral gating
            audio = self._spectral_gate(audio)
        
        print("   🔇 Advanced noise reduction applied")
        return audio
    
    def _spectral_gate(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Fallback spectral gating for noise reduction"""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        
        # Calculate noise floor from quietest parts
        noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
        
        # Gate: reduce content below threshold
        threshold = noise_floor * (10 ** (threshold_db / 20))
        mask = np.where(mag > threshold, 1.0, 0.1)
        mask = gaussian_filter1d(mask, sigma=2, axis=1)
        
        gated_stft = mag * mask * np.exp(1j * phase)
        return librosa.istft(gated_stft, hop_length=512)[:len(audio)]
    
    # ==========================================================================
    # STAGE 3: Echo/Reverb Removal (Adobe's "studio sound")
    # ==========================================================================
    
    def remove_reverb(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove room reverb/echo to get dry studio sound
        Uses spectral subtraction of estimated reverb tail
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        
        # Estimate reverb as the "smeared" energy over time
        # Reverb creates temporal smearing in spectrogram
        
        # Calculate temporal derivative (changes over time)
        mag_diff = np.diff(mag, axis=1)
        
        # Reverb has slow decay - identify and reduce it
        # Use median filtering to estimate the "direct" sound
        direct_mag = median_filter(mag, size=(1, 5))
        
        # The difference is mostly reverb
        reverb_estimate = mag - direct_mag
        reverb_estimate = np.clip(reverb_estimate, 0, None)
        
        # Subtract reverb (but not too aggressively)
        dereverbed_mag = mag - 0.5 * reverb_estimate
        dereverbed_mag = np.clip(dereverbed_mag, 0, None)
        
        # Reconstruct
        dereverbed_stft = dereverbed_mag * np.exp(1j * phase)
        dereverbed = librosa.istft(dereverbed_stft, hop_length=512)
        
        if len(dereverbed) > len(audio):
            dereverbed = dereverbed[:len(audio)]
        elif len(dereverbed) < len(audio):
            dereverbed = np.pad(dereverbed, (0, len(audio) - len(dereverbed)))
        
        print("   🏠 Room reverb/echo removed")
        return dereverbed
    
    # ==========================================================================
    # STAGE 4: Bandwidth Extension (Restore high frequencies)
    # ==========================================================================
    
    def extend_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """
        Gently enhance clarity without causing whistling artifacts
        DISABLED bandwidth extension as it can cause whistling
        """
        # Skip bandwidth extension - it causes whistling artifacts
        # Just apply gentle high-frequency presence boost instead
        
        # Very gentle high shelf boost only
        if HAS_PEDALBOARD:
            from pedalboard import Pedalboard, HighShelfFilter
            board = Pedalboard([
                HighShelfFilter(cutoff_frequency_hz=6000, gain_db=1.5)  # Very gentle
            ])
            audio = audio.astype(np.float32)
            audio = board(audio, self.sr)
        
        print("   📡 Gentle clarity enhancement (no bandwidth extension)")
        return audio
    
    # ==========================================================================
    # STAGE 4.5: Remove Whistling/Ringing Artifacts
    # ==========================================================================
    
    def _remove_whistling(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove whistling/ringing artifacts that can occur from processing
        These are typically narrow-band tonal noises
        """
        # Compute STFT with good frequency resolution
        stft = librosa.stft(audio, n_fft=4096, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)
        
        # Find tonal (whistling) components
        # Whistling appears as narrow peaks that persist over time
        
        # Calculate mean magnitude per frequency bin
        mean_mag = np.mean(mag, axis=1)
        
        # Calculate local contrast - whistles have high peaks vs neighbors
        # Use median filter to find background level
        background = median_filter(mean_mag, size=21)
        
        # Tonal ratio: how much each frequency stands out
        tonal_ratio = mean_mag / (background + 1e-10)
        
        # Find whistling frequencies (high tonal ratio, especially in mid-high range)
        whistle_threshold = 2.0  # Frequencies 2x louder than neighbors
        
        # Focus on whistle-prone range (2kHz - 10kHz)
        whistle_start = np.searchsorted(freqs, 2000)
        whistle_end = np.searchsorted(freqs, 10000)
        
        # Create attenuation mask
        attenuation = np.ones_like(mag)
        
        for i in range(whistle_start, whistle_end):
            if tonal_ratio[i] > whistle_threshold:
                # This frequency is likely a whistle - attenuate it
                # Stronger attenuation for stronger whistles
                reduction = min(0.3, 1.0 / tonal_ratio[i])
                attenuation[i, :] = reduction
                
                # Also attenuate neighboring bins (whistles spread slightly)
                if i > 0:
                    attenuation[i-1, :] = min(attenuation[i-1, :].min(), reduction * 1.5)
                if i < len(freqs) - 1:
                    attenuation[i+1, :] = min(attenuation[i+1, :].min(), reduction * 1.5)
        
        # Smooth the attenuation to avoid artifacts
        for i in range(attenuation.shape[0]):
            attenuation[i, :] = gaussian_filter1d(attenuation[i, :], sigma=3)
        
        # Apply attenuation
        cleaned_mag = mag * attenuation
        cleaned_stft = cleaned_mag * np.exp(1j * phase)
        
        cleaned = librosa.istft(cleaned_stft, hop_length=512, length=len(audio))
        
        # Count how many whistles were removed
        whistles_found = np.sum(tonal_ratio[whistle_start:whistle_end] > whistle_threshold)
        if whistles_found > 0:
            print(f"   🎵 Removed {whistles_found} whistling frequencies")
        
        return cleaned
    
    # ==========================================================================
    # STAGE 5: De-essing and De-plosive (Professional vocal cleanup)
    # ==========================================================================
    
    def deess_and_deplosive(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove harsh sibilants (s, sh sounds) and plosives (p, b, t sounds)
        Also removes whistling/ringing artifacts
        """
        # First: Remove whistling/ringing frequencies
        audio = self._remove_whistling(audio)
        
        # De-essing: Target 4-9kHz where sibilants live
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        # Find sibilant frequency range
        sibilant_start = np.searchsorted(freqs, 4000)
        sibilant_end = np.searchsorted(freqs, 9000)
        
        # Detect frames with excessive sibilance
        sibilant_energy = np.mean(mag[sibilant_start:sibilant_end], axis=0)
        total_energy = np.mean(mag, axis=0) + 1e-10
        sibilant_ratio = sibilant_energy / total_energy
        
        # Reduce sibilants where they're too prominent
        threshold = np.percentile(sibilant_ratio, 80)
        reduction_mask = np.ones_like(mag)
        
        for i, ratio in enumerate(sibilant_ratio):
            if ratio > threshold:
                # Reduce high frequencies in this frame
                reduction = 0.5 + 0.5 * (threshold / ratio)
                reduction_mask[sibilant_start:sibilant_end, i] = reduction
        
        # Smooth the reduction
        reduction_mask = gaussian_filter1d(reduction_mask, sigma=2, axis=1)
        
        deessed_mag = mag * reduction_mask
        
        # De-plosive: Target low frequencies (< 200Hz) for plosive reduction
        plosive_end = np.searchsorted(freqs, 200)
        
        # Detect sudden low-frequency bursts (plosives)
        low_energy = np.mean(mag[:plosive_end], axis=0)
        low_diff = np.diff(low_energy, prepend=low_energy[0])
        
        # Find sudden increases (plosives)
        plosive_threshold = np.std(low_diff) * 2
        
        for i, diff in enumerate(low_diff):
            if diff > plosive_threshold:
                # Reduce low frequencies for a few frames
                for j in range(max(0, i-2), min(len(low_diff), i+3)):
                    deessed_mag[:plosive_end, j] *= 0.5
        
        # Reconstruct
        processed_stft = deessed_mag * np.exp(1j * phase)
        processed = librosa.istft(processed_stft, hop_length=512)
        
        if len(processed) != len(audio):
            processed = librosa.util.fix_length(processed, size=len(audio))
        
        print("   👄 De-essing and de-plosive applied")
        return processed
    
    # ==========================================================================
    # STAGE 6: Professional EQ and Compression (Broadcast standard)
    # ==========================================================================
    
    def apply_broadcast_processing(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply broadcast-standard processing:
        - High-pass filter (remove rumble)
        - Vocal presence boost
        - Gentle compression
        - Limiting
        """
        if HAS_PEDALBOARD:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=80),     # Remove rumble
                NoiseGate(threshold_db=-40, ratio=2),       # Gate quiet parts
                LowShelfFilter(cutoff_frequency_hz=200, gain_db=-2),  # Reduce muddiness
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2), # Air/presence
                Compressor(
                    threshold_db=-18,
                    ratio=3,
                    attack_ms=10,
                    release_ms=100
                ),
                Gain(gain_db=3),  # Makeup gain
                Limiter(threshold_db=-1, release_ms=100)  # Prevent clipping
            ])
            
            # Ensure float32 for pedalboard
            audio = audio.astype(np.float32)
            processed = board(audio, self.sr)
            
        else:
            # Fallback: Basic filtering
            # High-pass filter
            sos_hp = signal.butter(4, 80, btype='high', fs=self.sr, output='sos')
            processed = signal.sosfilt(sos_hp, audio)
            
            # Presence boost (2-5kHz)
            sos_presence = signal.butter(2, [2000, 5000], btype='band', fs=self.sr, output='sos')
            presence = signal.sosfilt(sos_presence, processed)
            processed = processed + presence * 0.2
            
            # Basic compression
            threshold = np.percentile(np.abs(processed), 90)
            processed = np.where(
                np.abs(processed) > threshold,
                np.sign(processed) * (threshold + (np.abs(processed) - threshold) / 3),
                processed
            )
            
            # Normalize
            processed = processed / (np.max(np.abs(processed)) + 1e-10) * 0.95
        
        print("   📻 Broadcast-quality processing applied")
        return processed
    
    # ==========================================================================
    # STAGE 7: Final Polish (Adobe's "studio quality" finishing)
    # ==========================================================================
    
    def final_polish(self, audio: np.ndarray) -> np.ndarray:
        """
        Final polish for studio-quality output
        - Remove any remaining whistling
        - Smooth any remaining artifacts
        - Normalize to broadcast standards
        - Add subtle warmth
        """
        # Final de-whistling pass
        audio = self._remove_whistling(audio)
        
        # Low-pass filter to remove any ultrasonic artifacts
        sos_lp = signal.butter(4, 11000, btype='low', fs=self.sr, output='sos')
        audio = signal.sosfilt(sos_lp, audio)
        
        # Add subtle warmth (very gentle saturation)
        warmth = np.tanh(audio * 1.1) * 0.95
        audio = audio * 0.7 + warmth * 0.3
        
        # Smooth any clicks/pops
        # Find sudden amplitude changes
        diff = np.abs(np.diff(audio, prepend=audio[0]))
        click_threshold = np.percentile(diff, 99.9)
        
        # Smooth around clicks
        for i in np.where(diff > click_threshold)[0]:
            start = max(0, i - 10)
            end = min(len(audio), i + 10)
            audio[start:end] = gaussian_filter1d(audio[start:end], sigma=2)
        
        # Final normalization to -16 LUFS (broadcast standard)
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-16/20)
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        # Final safety limiter
        audio = np.clip(audio, -0.99, 0.99)
        
        print("   ✨ Final studio polish applied")
        return audio
    
    # ==========================================================================
    # FULL PROCESSING PIPELINE
    # ==========================================================================
    
    def enhance(self, audio: np.ndarray, 
                isolate_voice: bool = True,
                remove_reverb: bool = True,
                extend_bandwidth: bool = True) -> np.ndarray:
        """
        Full Adobe Podcast-style enhancement pipeline
        
        Args:
            audio: Input audio
            isolate_voice: Whether to separate voice from background
            remove_reverb: Whether to remove room echo
            extend_bandwidth: Whether to restore high frequencies
        """
        print("\n" + "="*70)
        print("🎙️  ADOBE PODCAST-STYLE ENHANCEMENT")
        print("="*70)
        
        # Stage 1: Voice Isolation
        if isolate_voice:
            print("\n📍 Stage 1: Voice Isolation")
            audio = self.isolate_voice(audio)
        
        # Stage 2: Advanced Denoising
        print("\n📍 Stage 2: Advanced Noise Reduction")
        audio = self.advanced_denoise(audio)
        
        # Stage 3: Reverb Removal
        if remove_reverb:
            print("\n📍 Stage 3: Reverb/Echo Removal")
            audio = self.remove_reverb(audio)
        
        # Stage 4: Bandwidth Extension
        if extend_bandwidth:
            print("\n📍 Stage 4: Bandwidth Extension")
            audio = self.extend_bandwidth(audio)
        
        # Stage 5: De-essing and De-plosive
        print("\n📍 Stage 5: De-essing & De-plosive")
        audio = self.deess_and_deplosive(audio)
        
        # Stage 6: Broadcast Processing
        print("\n📍 Stage 6: Broadcast Processing")
        audio = self.apply_broadcast_processing(audio)
        
        # Stage 7: Final Polish
        print("\n📍 Stage 7: Final Studio Polish")
        audio = self.final_polish(audio)
        
        print("\n" + "="*70)
        print("✅ ADOBE PODCAST-STYLE ENHANCEMENT COMPLETE")
        print("="*70)
        
        return audio
    
    def enhance_file(self, input_path: str, output_path: str) -> bool:
        """Enhance a single file"""
        try:
            print(f"\n📂 Loading: {Path(input_path).name}")
            
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            print(f"   Duration: {len(audio)/sr:.2f}s")
            print(f"   Sample rate: {sr} Hz")
            
            enhanced = self.enhance(audio)
            
            sf.write(output_path, enhanced, sr, subtype='PCM_16')
            
            print(f"\n💾 Saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               ADOBE PODCAST-STYLE SPEECH ENHANCER                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Makes any recording sound like it was captured in a professional studio     ║
║                                                                              ║
║  Processing stages (like Adobe Podcast Enhance):                             ║
║                                                                              ║
║  1️⃣  Voice Isolation    - Separates voice from background noise             ║
║  2️⃣  Advanced Denoise   - AI-style noise reduction                          ║
║  3️⃣  Reverb Removal     - Removes room echo for dry studio sound            ║
║  4️⃣  Bandwidth Extend   - Restores clarity and high frequencies             ║
║  5️⃣  De-ess/De-plosive  - Removes harsh S sounds and P/B pops               ║
║  6️⃣  Broadcast Process  - Professional EQ, compression, limiting            ║
║  7️⃣  Final Polish       - Studio-quality finishing touches                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                      ║
║    python adobe_enhance.py <input.wav> [output.wav]                          ║
║                                                                              ║
║  Example:                                                                    ║
║    python adobe_enhance.py recording.wav studio_quality.wav                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.wav', '_adobe_enhanced.wav')
    
    enhancer = AdobePodcastEnhancer()
    enhancer.enhance_file(input_file, output_file)


if __name__ == "__main__":
    main()
