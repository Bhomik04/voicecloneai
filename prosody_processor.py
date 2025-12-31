"""
Prosody Processor - Adds natural speech flow to TTS output
- Sentence-ending dips (volume drops at end of sentences)
- Natural pauses between sentences
- Rising/falling intonation patterns
- Conversational/storytelling rhythm
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


class ProsodyProcessor:
    """Add natural speech prosody to flat TTS output"""
    
    def __init__(self, sr: int = 24000):
        self.sr = sr
        
    def detect_sentence_boundaries(self, audio: np.ndarray) -> list:
        """
        Detect likely sentence boundaries based on:
        - Actual silent regions (not just low energy)
        - Must be real pauses, not within-word dips
        """
        # Calculate energy envelope with larger window
        frame_length = int(0.050 * self.sr)  # 50ms frames (larger = more stable)
        hop_length = int(0.025 * self.sr)    # 25ms hop
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Heavy smoothing to avoid within-word dips
        rms_smooth = gaussian_filter1d(rms, sigma=10)
        
        # Only detect REAL silence (very low energy)
        # Use 10th percentile - only the quietest parts
        threshold = np.percentile(rms_smooth, 10)
        is_silent = rms_smooth < threshold
        
        # Find continuous silent regions that are at least 150ms
        # (real pauses between sentences, not consonant gaps)
        min_silence_frames = int(0.15 * self.sr / hop_length)  # 150ms minimum silence
        min_speech_duration = int(1.0 * self.sr / hop_length)  # At least 1s of speech before boundary
        
        boundaries = []
        silence_start = None
        last_boundary = -min_speech_duration * 2
        
        for i, silent in enumerate(is_silent):
            if silent:
                if silence_start is None:
                    silence_start = i
            else:
                if silence_start is not None:
                    silence_duration = i - silence_start
                    # Only mark as boundary if:
                    # 1. Silence is long enough (real pause)
                    # 2. Far enough from last boundary (not too many)
                    if silence_duration >= min_silence_frames:
                        if (silence_start - last_boundary) >= min_speech_duration:
                            # Put boundary at the START of silence (after speech ends)
                            sample_idx = silence_start * hop_length
                            boundaries.append(sample_idx)
                            last_boundary = silence_start
                    silence_start = None
        
        # DON'T add energy dips - they cause mid-word cuts
        
        print(f"   📍 Found {len(boundaries)} sentence boundaries (conservative)")
        
        return boundaries
    
    def add_sentence_ending_dips(self, audio: np.ndarray, boundaries: list) -> np.ndarray:
        """
        Add natural volume dip at the end of each sentence
        Human voice naturally drops ~15-20% at statement endings
        More gentle to avoid cutting words
        """
        output = audio.copy()
        dip_duration = int(0.25 * self.sr)  # 250ms dip (shorter, gentler)
        
        for boundary in boundaries:
            # Create a gentle fade BEFORE the boundary (in the speech, not after)
            start = max(0, boundary - dip_duration)
            end = boundary  # End at the boundary, not after
            
            if end - start < dip_duration // 4:
                continue
                
            # Create envelope that gently dips to 85% at boundary
            length = end - start
            t = np.linspace(0, np.pi/2, length)
            
            # Gentle cosine dip - only 15% drop
            dip_envelope = 1.0 - 0.15 * np.sin(t)
            
            output[start:end] *= dip_envelope
        
        print(f"   📉 Added gentle volume dips at {len(boundaries)} sentence endings")
        return output
    
    def add_natural_pauses(self, audio: np.ndarray, boundaries: list) -> np.ndarray:
        """
        Extend existing natural pauses between sentences slightly
        DON'T insert new pauses - just make existing ones a bit longer
        """
        if len(boundaries) == 0:
            return audio
        
        # Instead of inserting pauses, we'll stretch existing silent regions
        # This is safer and won't cut words
        
        frame_length = int(0.050 * self.sr)
        hop_length = int(0.025 * self.sr)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = gaussian_filter1d(rms, sigma=10)
        threshold = np.percentile(rms_smooth, 15)
        
        # Find actual silent regions and extend them
        segments = []
        prev_end = 0
        pause_added = 0
        
        for boundary in boundaries:
            # Find the actual silent region around this boundary
            frame_idx = boundary // hop_length
            
            # Look for silence region
            start_frame = max(0, frame_idx - 5)
            end_frame = min(len(rms_smooth), frame_idx + 10)
            
            # Find where silence actually is
            silence_mask = rms_smooth[start_frame:end_frame] < threshold
            
            if not np.any(silence_mask):
                # No real silence here, skip
                continue
            
            # Find the actual start of silence
            silence_start_rel = np.argmax(silence_mask)
            silence_start = (start_frame + silence_start_rel) * hop_length
            
            if silence_start <= prev_end:
                continue
            
            # Add segment up to silence
            segments.append(audio[prev_end:silence_start])
            
            # Add a small pause extension (100-200ms of quiet)
            pause_extension = int(np.random.uniform(0.1, 0.2) * self.sr)
            quiet_pause = np.random.randn(pause_extension) * 0.001  # Almost silent
            segments.append(quiet_pause)
            pause_added += pause_extension
            
            prev_end = silence_start
        
        # Add remaining audio
        segments.append(audio[prev_end:])
        
        output = np.concatenate(segments)
        
        added_time = pause_added / self.sr
        print(f"   ⏸️  Extended pauses by {added_time:.2f}s total")
        
        return output
    
    def add_pitch_contour(self, audio: np.ndarray) -> np.ndarray:
        """
        Add natural pitch variations using time-stretching
        Sentences typically: rise in middle, fall at end
        """
        # This is complex with pitch shifting, so we simulate with amplitude
        # Pitch perception is partially tied to amplitude
        
        # Create a natural sentence-like contour
        # Rises in first third, peaks, then falls
        length = len(audio)
        
        # Detect major segments
        segment_length = int(2.0 * self.sr)  # ~2 second segments
        num_segments = length // segment_length + 1
        
        output = audio.copy()
        
        for seg in range(num_segments):
            start = seg * segment_length
            end = min((seg + 1) * segment_length, length)
            seg_len = end - start
            
            if seg_len < segment_length // 2:
                continue
            
            # Natural sentence contour: slight rise then fall
            t = np.linspace(0, 1, seg_len)
            
            # Bell curve - rises to peak at 40%, then falls
            contour = 0.95 + 0.10 * np.exp(-((t - 0.4) ** 2) / 0.1)
            
            # Add slight drop at end (declarative statement pattern)
            contour[-int(seg_len * 0.2):] *= np.linspace(1.0, 0.85, int(seg_len * 0.2))
            
            output[start:end] *= contour
        
        print(f"   🎵 Applied natural pitch/amplitude contour")
        return output
    
    def add_breath_points(self, audio: np.ndarray, boundaries: list) -> np.ndarray:
        """
        Add subtle breath sounds at natural breathing points
        Humans breathe every 5-8 seconds during speech
        """
        output = audio.copy()
        
        # Add breath sound before some sentence starts
        breath_duration = int(0.15 * self.sr)  # 150ms breath
        
        breath_count = 0
        for i, boundary in enumerate(boundaries):
            # Only add breath every 2-3 sentences (natural pattern)
            if i % 3 != 0:
                continue
                
            breath_start = boundary + int(0.05 * self.sr)  # After pause starts
            breath_end = min(breath_start + breath_duration, len(audio))
            
            if breath_end - breath_start < breath_duration // 2:
                continue
            
            # Create breath sound (filtered noise burst)
            breath = np.random.randn(breath_end - breath_start)
            
            # Shape it like an inhale (quick rise, slow fall)
            t = np.linspace(0, 1, len(breath))
            breath_envelope = np.exp(-3 * (t - 0.3) ** 2)
            breath = breath * breath_envelope * 0.03  # Very subtle
            
            # High-pass filter (breath is mostly high frequency)
            sos = signal.butter(2, 2000, btype='high', fs=self.sr, output='sos')
            breath = signal.sosfilt(sos, breath)
            breath = breath * 0.02  # Extra quiet
            
            # Add to output
            output[breath_start:breath_end] += breath
            breath_count += 1
        
        print(f"   🌬️  Added {breath_count} natural breath points")
        return output
    
    def add_word_emphasis_variation(self, audio: np.ndarray) -> np.ndarray:
        """
        Add natural emphasis variation on words
        Not all words are spoken with equal energy
        """
        # Detect word-level segments (short energy peaks)
        frame_length = int(0.05 * self.sr)  # 50ms
        hop_length = int(0.025 * self.sr)   # 25ms
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find peaks (likely stressed syllables)
        peaks = signal.find_peaks(rms, distance=8, prominence=np.std(rms)*0.5)[0]
        
        output = audio.copy()
        
        # Randomly vary emphasis on some words
        for peak in peaks:
            if np.random.random() > 0.3:  # 70% of words get variation
                continue
                
            center_sample = peak * hop_length
            word_length = int(0.2 * self.sr)  # ~200ms word
            
            start = max(0, center_sample - word_length // 2)
            end = min(len(audio), center_sample + word_length // 2)
            
            # Random emphasis (90-110% of original)
            emphasis = np.random.uniform(0.90, 1.10)
            
            # Smooth application
            length = end - start
            window = np.hanning(length)
            
            # Blend emphasis
            output[start:end] *= (1 - window) + (window * emphasis)
        
        print(f"   💬 Added natural word emphasis variation")
        return output
    
    def smooth_transitions(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure all transitions are smooth (no clicks or pops)
        """
        # Apply very light smoothing at zero-crossings
        # This helps with any artifacts from previous processing
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        
        # Smooth around a subset of zero crossings
        window_size = 5
        
        for zc in zero_crossings[::10]:  # Every 10th crossing
            start = max(0, zc - window_size)
            end = min(len(audio), zc + window_size)
            
            # Apply tiny smoothing
            audio[start:end] = gaussian_filter1d(audio[start:end], sigma=1)
        
        return audio
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete prosody processing pipeline
        """
        print("\n🎭 ADDING NATURAL PROSODY")
        print("-" * 50)
        
        # Step 1: Detect sentence boundaries
        print("   🔍 Detecting sentence boundaries...")
        boundaries = self.detect_sentence_boundaries(audio)
        
        # Step 2: Add pitch/amplitude contour
        print("   🎵 Adding natural intonation contour...")
        audio = self.add_pitch_contour(audio)
        
        # Step 3: Add sentence-ending dips
        print("   📉 Adding sentence-ending dips...")
        audio = self.add_sentence_ending_dips(audio, boundaries)
        
        # Step 4: Add word emphasis variation
        print("   💬 Adding word emphasis variation...")
        audio = self.add_word_emphasis_variation(audio)
        
        # Step 5: Add breath points
        print("   🌬️  Adding breath points...")
        audio = self.add_breath_points(audio, boundaries)
        
        # Step 6: Add natural pauses
        print("   ⏸️  Inserting natural pauses...")
        audio = self.add_natural_pauses(audio, boundaries)
        
        # Step 7: Smooth transitions
        print("   ✨ Smoothing transitions...")
        audio = self.smooth_transitions(audio)
        
        # Final normalization
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-16/20)
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        audio = np.clip(audio, -0.99, 0.99)
        
        print("-" * 50)
        print("✅ Prosody processing complete!")
        
        return audio
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """Process a file"""
        try:
            print(f"\n🎙️  Processing: {Path(input_path).name}")
            print("=" * 70)
            
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            print(f"   Duration: {len(audio)/sr:.2f}s")
            
            audio_processed = self.process(audio)
            
            sf.write(output_path, audio_processed, sr, subtype='PCM_16')
            
            new_duration = len(audio_processed) / sr
            print(f"\n📁 Output: {output_path}")
            print(f"   New duration: {new_duration:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\n🎭 PROSODY PROCESSOR")
        print("=" * 70)
        print("Adds natural speech flow to TTS output:")
        print("  • Sentence-ending dips (voice drops at end)")
        print("  • Natural pauses between sentences")
        print("  • Rising/falling intonation")
        print("  • Breath points")
        print("  • Word emphasis variation")
        print("\nUsage:")
        print("  python prosody_processor.py <input.wav> [output.wav]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.wav', '_natural.wav')
    
    processor = ProsodyProcessor()
    processor.process_file(input_file, output_file)


if __name__ == "__main__":
    main()
