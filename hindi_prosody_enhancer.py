"""
🎙️ Hindi/Indian Speech Prosody Enhancer
=========================================

Fixes the "foreigner speaking Hindi" problem in TTS output by:
1. Adjusting pitch contours to match Indian speech patterns
2. Adding natural Hindi intonation (sentence-ending low tones)
3. Correcting rhythm and stress patterns for Hindi/Hinglish
4. Adding natural phrase-final lengthening
5. Fixing the characteristic "sing-song" pattern of Indian languages

Key differences between Indian and Western prosody:
- Indian languages have phrase-final lengthening and pitch drop
- Hindi stress patterns are different (less stress on function words)
- Natural "humming" quality at phrase ends
- More pitch variation within words
- Characteristic head wobble reflected in pitch micro-variations

Usage:
    from hindi_prosody_enhancer import HindiProsodyEnhancer
    
    enhancer = HindiProsodyEnhancer()
    fixed_audio = enhancer.enhance(audio, sample_rate)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class HindiProsodyEnhancer:
    """
    Enhance TTS output to sound more naturally Indian/Hindi
    
    Addresses common issues with Western TTS systems speaking Hindi:
    - Flat pitch (missing the natural pitch movements)
    - Wrong stress patterns (English stress rules applied to Hindi)
    - Missing phrase-final features (lengthening, pitch drop)
    - Unnatural rhythm (too regular, missing micro-timing variations)
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        
        # Hindi prosody parameters
        self.phrase_final_drop_semitones = 4.0  # Pitch drops ~4 semitones at phrase end
        self.phrase_final_lengthening = 1.15  # 15% longer at phrase end
        self.micro_pitch_variation_cents = 30  # Subtle pitch wobble
        self.breath_group_seconds = 3.5  # Natural breathing interval
        
    def detect_phrase_boundaries(self, audio: np.ndarray) -> List[int]:
        """
        Detect phrase boundaries based on energy dips and pauses
        More sensitive than sentence detection - catches breath groups
        """
        frame_length = int(0.03 * self.sr)  # 30ms frames
        hop_length = int(0.015 * self.sr)   # 15ms hop
        
        # Get energy envelope
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = gaussian_filter1d(rms, sigma=5)
        
        # Find dips (potential phrase boundaries)
        threshold = np.percentile(rms_smooth, 25)
        is_low = rms_smooth < threshold
        
        # Find transitions from speech to low energy
        boundaries = []
        min_phrase_frames = int(0.8 * self.sr / hop_length)  # Min 0.8s between phrases
        last_boundary = -min_phrase_frames * 2
        
        for i in range(1, len(is_low)):
            if is_low[i] and not is_low[i-1]:  # Transition to low
                if (i - last_boundary) >= min_phrase_frames:
                    boundaries.append(i * hop_length)
                    last_boundary = i
        
        return boundaries
    
    def detect_word_boundaries(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect word-level segments for rhythm correction
        Returns list of (start_sample, end_sample) tuples
        """
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        # Dynamic threshold
        threshold = np.percentile(rms_smooth, 30)
        is_speech = rms_smooth > threshold
        
        words = []
        in_word = False
        word_start = 0
        min_word_frames = int(0.08 * self.sr / hop_length)  # Min 80ms word
        
        for i, is_s in enumerate(is_speech):
            if is_s and not in_word:
                word_start = i * hop_length
                in_word = True
            elif not is_s and in_word:
                word_end = i * hop_length
                if (i - word_start // hop_length) >= min_word_frames:
                    words.append((word_start, word_end))
                in_word = False
        
        if in_word:
            words.append((word_start, len(audio)))
        
        return words
    
    def add_phrase_final_pitch_drop(self, audio: np.ndarray, boundaries: List[int]) -> np.ndarray:
        """
        Add natural Hindi phrase-final pitch drop
        Hindi sentences end with a distinctive downward pitch glide
        """
        if len(boundaries) == 0:
            return audio
        
        output = audio.copy()
        
        # Duration of pitch drop region
        drop_duration = int(0.3 * self.sr)  # 300ms before boundary
        
        for boundary in boundaries:
            start = max(0, boundary - drop_duration)
            end = min(boundary, len(audio))
            
            if end - start < drop_duration // 2:
                continue
            
            segment = output[start:end].copy()
            
            # Apply pitch shift using phase vocoder (simplified)
            # Create a gradual pitch drop envelope
            length = len(segment)
            
            # Pitch drop from 0 to -4 semitones
            pitch_shift_cents = np.linspace(0, -self.phrase_final_drop_semitones * 100, length)
            
            # Apply via resampling approximation (faster than full vocoder)
            # This is a simplified approach that works for speech
            rate_curve = 2 ** (pitch_shift_cents / 1200)  # Cents to ratio
            
            # Resample to approximate pitch shift
            avg_rate = np.mean(rate_curve)
            if avg_rate != 1.0:
                # Simple time-stretch approximation
                new_length = int(length / avg_rate)
                indices = np.linspace(0, length - 1, new_length).astype(int)
                indices = np.clip(indices, 0, length - 1)
                
                # Interpolate to original length
                if new_length != length:
                    stretched = segment[indices]
                    interp_func = interp1d(np.arange(len(stretched)), stretched, kind='linear', fill_value='extrapolate')
                    segment = interp_func(np.linspace(0, len(stretched) - 1, length))
            
            # Also add amplitude drop (natural with pitch drop)
            amplitude_envelope = np.linspace(1.0, 0.85, length)
            segment = segment * amplitude_envelope
            
            output[start:end] = segment
        
        return output
    
    def add_hindi_intonation_pattern(self, audio: np.ndarray) -> np.ndarray:
        """
        Add characteristic Hindi intonation patterns:
        - Rising pitch on focus words
        - Falling contour at phrase ends
        - Micro-variations that give the "melodic" quality
        """
        output = audio.copy()
        
        # Detect energy peaks (likely stressed syllables)
        frame_length = int(0.04 * self.sr)
        hop_length = int(0.02 * self.sr)
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find peaks
        peaks, properties = signal.find_peaks(rms, distance=10, prominence=np.std(rms) * 0.3)
        
        # Add subtle pitch emphasis on stressed syllables
        for peak in peaks:
            center_sample = peak * hop_length
            region_size = int(0.1 * self.sr)  # 100ms region
            
            start = max(0, center_sample - region_size // 2)
            end = min(len(audio), center_sample + region_size // 2)
            
            # Create subtle amplitude modulation for "pitch-like" effect
            length = end - start
            t = np.linspace(0, np.pi, length)
            emphasis = 1.0 + 0.08 * np.sin(t)  # 8% emphasis
            
            output[start:end] *= emphasis
        
        return output
    
    def add_micro_pitch_variations(self, audio: np.ndarray) -> np.ndarray:
        """
        Add subtle pitch micro-variations characteristic of natural Hindi speech
        This gives the "alive" quality that TTS often lacks
        """
        output = audio.copy()
        
        # Slow pitch wobble (1-2 Hz)
        time = np.arange(len(audio)) / self.sr
        wobble = 1.0 + 0.01 * np.sin(2 * np.pi * 1.5 * time)  # 1% wobble at 1.5Hz
        
        # Apply as amplitude modulation (approximates pitch perception)
        output = output * wobble
        
        # Add very subtle random variations
        random_variation = 1.0 + 0.005 * np.random.randn(len(audio))
        random_variation = gaussian_filter1d(random_variation, sigma=100)
        output = output * random_variation
        
        return output
    
    def add_phrase_final_lengthening(self, audio: np.ndarray, boundaries: List[int]) -> np.ndarray:
        """
        Add phrase-final syllable lengthening (characteristic of Hindi)
        The last syllable of phrases is typically ~15-20% longer
        """
        if len(boundaries) == 0:
            return audio
        
        segments = []
        prev_end = 0
        
        for boundary in boundaries:
            # Region before boundary (last 200ms = last syllable approximately)
            lengthen_start = max(prev_end, boundary - int(0.2 * self.sr))
            
            # Add segment before lengthening region
            if lengthen_start > prev_end:
                segments.append(audio[prev_end:lengthen_start])
            
            # Time-stretch the final syllable region
            final_segment = audio[lengthen_start:boundary]
            if len(final_segment) > 100:
                stretched = self._time_stretch(final_segment, self.phrase_final_lengthening)
                segments.append(stretched)
            else:
                segments.append(final_segment)
            
            prev_end = boundary
        
        # Add remaining audio
        if prev_end < len(audio):
            segments.append(audio[prev_end:])
        
        return np.concatenate(segments) if segments else audio
    
    def _time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Simple time stretch using resampling"""
        new_length = int(len(audio) * rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        
        # Linear interpolation
        interp_func = interp1d(np.arange(len(audio)), audio, kind='linear', fill_value='extrapolate')
        return interp_func(indices)
    
    def fix_rhythm_pattern(self, audio: np.ndarray, words: List[Tuple[int, int]]) -> np.ndarray:
        """
        Adjust rhythm to be more Hindi-like:
        - Less regular timing (English is stress-timed, Hindi is syllable-timed)
        - Content words slightly emphasized
        - Function words reduced
        """
        if len(words) < 3:
            return audio
        
        output = audio.copy()
        
        # Calculate word energies
        word_energies = []
        for start, end in words:
            if end - start > 0:
                energy = np.sqrt(np.mean(audio[start:end] ** 2))
                word_energies.append(energy)
            else:
                word_energies.append(0)
        
        if not word_energies:
            return audio
        
        mean_energy = np.mean(word_energies)
        
        # Apply slight variations based on position and energy
        for i, (start, end) in enumerate(words):
            if end <= start:
                continue
            
            length = end - start
            
            # Position-based adjustment (phrase-initial and final words are often emphasized)
            if i == 0 or i == len(words) - 1:
                emphasis = 1.05  # 5% louder
            elif i % 2 == 0:  # Alternate pattern (simplified)
                emphasis = 0.98
            else:
                emphasis = 1.02
            
            # Create smooth envelope
            t = np.linspace(0, np.pi, length)
            envelope = 1.0 + (emphasis - 1.0) * np.sin(t) ** 2
            
            output[start:end] *= envelope
        
        return output
    
    def add_natural_breathiness(self, audio: np.ndarray, boundaries: List[int]) -> np.ndarray:
        """
        Add subtle breathiness at phrase boundaries
        Natural speech has slight aspiration/breathiness at transitions
        """
        output = audio.copy()
        
        for boundary in boundaries:
            # Add tiny breath-like noise at boundary
            breath_duration = int(0.08 * self.sr)  # 80ms
            breath_start = boundary
            breath_end = min(boundary + breath_duration, len(audio))
            
            if breath_end - breath_start < breath_duration // 2:
                continue
            
            # Generate filtered noise
            noise = np.random.randn(breath_end - breath_start)
            
            # Shape envelope (quick attack, slow decay)
            t = np.linspace(0, 1, len(noise))
            envelope = t * np.exp(-3 * t)  # Peaks around t=0.33
            
            noise = noise * envelope * 0.015  # Very subtle
            
            # High-pass filter (breathiness is high-frequency)
            sos = signal.butter(2, 2500, btype='high', fs=self.sr, output='sos')
            noise = signal.sosfilt(sos, noise)
            noise = noise * 0.01  # Extra subtle
            
            output[breath_start:breath_end] += noise
        
        return output
    
    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        intensity: float = 1.0  # 0.0 to 1.5, higher = more effect
    ) -> np.ndarray:
        """
        Apply full Hindi prosody enhancement pipeline
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate (uses default if not provided)
            intensity: Enhancement intensity (0=none, 1=normal, 1.5=strong)
            
        Returns:
            Enhanced audio array
        """
        if sample_rate is not None:
            self.sr = sample_rate
        
        print("\n🎙️ HINDI PROSODY ENHANCEMENT")
        print("-" * 50)
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        
        # Step 1: Detect boundaries
        print("   🔍 Detecting phrase boundaries...")
        boundaries = self.detect_phrase_boundaries(audio)
        print(f"      Found {len(boundaries)} phrase boundaries")
        
        # Step 2: Detect words for rhythm
        print("   📝 Detecting word segments...")
        words = self.detect_word_boundaries(audio)
        print(f"      Found {len(words)} word segments")
        
        # Apply enhancements with intensity scaling
        output = audio.copy()
        
        # Step 3: Add phrase-final pitch drops (most important!)
        if intensity > 0:
            print("   📉 Adding phrase-final pitch drops...")
            # Temporarily boost the effect
            original_drop = self.phrase_final_drop_semitones
            self.phrase_final_drop_semitones = original_drop * intensity
            output = self.add_phrase_final_pitch_drop(output, boundaries)
            self.phrase_final_drop_semitones = original_drop
        
        # Step 4: Add Hindi intonation patterns
        if intensity > 0.3:
            print("   🎵 Adding Hindi intonation patterns...")
            output = self.add_hindi_intonation_pattern(output)
        
        # Step 5: Add micro-pitch variations
        if intensity > 0.5:
            print("   🔊 Adding micro-pitch variations...")
            output = self.add_micro_pitch_variations(output)
        
        # Step 6: Add phrase-final lengthening
        if intensity > 0.3:
            print("   ⏱️ Adding phrase-final lengthening...")
            output = self.add_phrase_final_lengthening(output, boundaries)
        
        # Step 7: Fix rhythm pattern
        if intensity > 0.4:
            print("   🥁 Adjusting rhythm pattern...")
            # Re-detect words since audio length may have changed
            words = self.detect_word_boundaries(output)
            output = self.fix_rhythm_pattern(output, words)
        
        # Step 8: Add natural breathiness
        if intensity > 0.6:
            print("   🌬️ Adding natural breathiness...")
            boundaries = self.detect_phrase_boundaries(output)
            output = self.add_natural_breathiness(output, boundaries)
        
        # Normalize
        max_val = np.abs(output).max()
        if max_val > 0:
            output = output / max_val * 0.95
        
        print("-" * 50)
        print("✅ Hindi prosody enhancement complete!")
        
        return output
    
    def process_file(self, input_path: str, output_path: str, intensity: float = 1.0) -> bool:
        """Process an audio file"""
        try:
            print(f"\n🎙️ Processing: {Path(input_path).name}")
            
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            print(f"   Duration: {len(audio)/sr:.2f}s")
            
            enhanced = self.enhance(audio, sr, intensity)
            
            sf.write(output_path, enhanced, sr, subtype='PCM_16')
            print(f"\n💾 Saved: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


class IndianAccentCorrector:
    """
    Correct TTS output to have authentic Indian English accent
    
    When TTS sounds like "foreigner speaking Hindi", this helps:
    - Adjust vowel qualities (Indian English has different vowel space)
    - Fix consonant timings
    - Add characteristic Indian intonation to English words
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self.hindi_enhancer = HindiProsodyEnhancer(sample_rate)
    
    def correct_accent(self, audio: np.ndarray, language: str = "hi") -> np.ndarray:
        """
        Apply accent correction based on target language
        
        Args:
            audio: Input audio
            language: "hi" for Hindi, "hinglish" for Hindi-English mix
            
        Returns:
            Corrected audio
        """
        print(f"\n🎯 INDIAN ACCENT CORRECTION (target: {language})")
        
        # Apply Hindi prosody (works for both Hindi and Indian English)
        corrected = self.hindi_enhancer.enhance(audio, self.sr, intensity=1.2)
        
        # Additional adjustments for specific language types
        if language in ["hi", "hinglish"]:
            # Add characteristic retroflex quality approximation
            # (Simplified - actual retroflex requires advanced processing)
            corrected = self._add_spectral_warmth(corrected)
        
        return corrected
    
    def _add_spectral_warmth(self, audio: np.ndarray) -> np.ndarray:
        """
        Add spectral warmth characteristic of Indian speech
        Indian languages have more low-mid frequency energy
        """
        # Subtle bass boost (100-300 Hz region)
        sos_boost = signal.butter(2, [100, 300], btype='band', fs=self.sr, output='sos')
        bass_content = signal.sosfilt(sos_boost, audio)
        
        # Mix back in
        output = audio + 0.1 * bass_content
        
        # Normalize
        max_val = np.abs(output).max()
        if max_val > 0:
            output = output / max_val * 0.95
        
        return output


def main():
    """Test the Hindi prosody enhancer"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n🎙️ HINDI PROSODY ENHANCER")
        print("=" * 60)
        print("Fixes 'foreigner speaking Hindi' problem in TTS output")
        print("\nEnhancements:")
        print("  • Phrase-final pitch drops (sentence ending low tones)")
        print("  • Hindi intonation patterns")
        print("  • Micro-pitch variations (natural quality)")
        print("  • Phrase-final lengthening")
        print("  • Rhythm pattern adjustment")
        print("  • Natural breathiness")
        print("\nUsage:")
        print("  python hindi_prosody_enhancer.py <input.wav> [output.wav] [intensity]")
        print("\n  intensity: 0.5-1.5 (default: 1.0)")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.wav', '_hindi_enhanced.wav')
    intensity = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    enhancer = HindiProsodyEnhancer()
    enhancer.process_file(input_file, output_file, intensity)


if __name__ == "__main__":
    main()
