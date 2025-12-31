"""
Professional Audio Enhancement & Processing
Podcast-quality audio preprocessing and post-processing for voice cloning
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import noisereduce as nr
except ImportError:
    print("⚠️  Installing noisereduce...")
    os.system("pip install noisereduce")
    import noisereduce as nr

try:
    from pedalboard import Pedalboard, Compressor, Gain, NoiseGate, LowShelfFilter, HighpassFilter, Limiter
    from pedalboard.io import AudioFile
except ImportError:
    print("⚠️  Installing pedalboard...")
    os.system("pip install pedalboard")
    from pedalboard import Pedalboard, Compressor, Gain, NoiseGate, LowShelfFilter, HighpassFilter, Limiter
    from pedalboard.io import AudioFile


class PodcastAudioProcessor:
    """
    Professional audio processing for podcast-quality voice cloning
    """
    
    def __init__(self, target_sr: int = 24000):
        """
        Initialize audio processor
        
        Args:
            target_sr: Target sample rate (24000 Hz recommended for voice cloning)
        """
        self.target_sr = target_sr
        
        # Professional podcast processing chain
        self.enhancement_chain = Pedalboard([
            # Remove low-frequency rumble and noise
            HighpassFilter(cutoff_frequency_hz=80),
            
            # Noise gate to remove background noise during silence
            NoiseGate(threshold_db=-40, ratio=10, attack_ms=1.0, release_ms=100),
            
            # Compression for consistent volume
            Compressor(threshold_db=-20, ratio=4, attack_ms=5.0, release_ms=50),
            
            # Boost voice presence
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=6, q=0.7),
            
            # Overall gain adjustment
            Gain(gain_db=3),
            
            # Final limiter to prevent clipping
            Limiter(threshold_db=-1.0, release_ms=50)
        ])
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to mono if needed"""
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return audio, sr
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target level in dB
        
        Args:
            audio: Input audio array
            target_level: Target level in dB (default: -20 dB)
        """
        # Calculate current RMS level
        rms = np.sqrt(np.mean(audio**2))
        current_db = 20 * np.log10(rms + 1e-10)
        
        # Calculate gain needed
        gain_db = target_level - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def reduce_noise(self, audio: np.ndarray, sr: int, 
                     prop_decrease: float = 0.8,
                     stationary: bool = True) -> np.ndarray:
        """
        Advanced noise reduction using spectral gating
        
        Args:
            audio: Input audio array
            sr: Sample rate
            prop_decrease: Proportion of noise to reduce (0.0 to 1.0)
            stationary: Use stationary (True) or non-stationary (False) algorithm
        """
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            n_std_thresh_stationary=1.5
        )
        
        return reduced
    
    def enhance_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply professional podcast-style audio enhancement
        
        Args:
            audio: Input audio array
            sr: Sample rate
        """
        # Ensure audio is 2D for pedalboard (channels, samples)
        if audio.ndim == 1:
            audio_2d = audio.reshape(1, -1)
        else:
            audio_2d = audio
        
        # Apply enhancement chain
        enhanced = self.enhancement_chain(audio_2d, sr)
        
        # Convert back to 1D if input was 1D
        if audio.ndim == 1:
            enhanced = enhanced.flatten()
        
        return enhanced
    
    def remove_silence(self, audio: np.ndarray, sr: int,
                      top_db: int = 30,
                      frame_length: int = 2048,
                      hop_length: int = 512) -> np.ndarray:
        """
        Remove leading and trailing silence
        
        Args:
            audio: Input audio array
            sr: Sample rate
            top_db: Threshold for silence detection
        """
        # Trim silence from beginning and end
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        return trimmed
    
    def process_sample(self, input_path: str, output_path: str,
                      denoise: bool = True,
                      enhance: bool = True,
                      trim_silence: bool = True,
                      normalize: bool = True) -> bool:
        """
        Complete audio processing pipeline for training samples
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            denoise: Apply noise reduction
            enhance: Apply professional enhancement
            trim_silence: Remove leading/trailing silence
            normalize: Normalize audio levels
        
        Returns:
            bool: True if successful
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Remove silence first
            if trim_silence:
                audio = self.remove_silence(audio, sr)
            
            # Noise reduction
            if denoise:
                audio = self.reduce_noise(audio, sr, prop_decrease=0.8)
            
            # Normalization
            if normalize:
                audio = self.normalize_audio(audio, target_level=-20.0)
            
            # Professional enhancement
            if enhance:
                audio = self.enhance_audio(audio, sr)
            
            # Final normalization to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0.99:
                audio = audio * (0.99 / max_val)
            
            # Save processed audio
            sf.write(output_path, audio, sr, subtype='PCM_16')
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {input_path}: {str(e)}")
            return False
    
    def process_tts_output(self, input_path: str, output_path: str) -> bool:
        """
        Post-process TTS-generated audio for podcast quality
        
        Args:
            input_path: Path to raw TTS output
            output_path: Path to save enhanced output
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Light noise reduction (TTS output is usually clean)
            audio = self.reduce_noise(audio, sr, prop_decrease=0.5)
            
            # Normalize
            audio = self.normalize_audio(audio, target_level=-18.0)
            
            # Apply enhancement
            audio = self.enhance_audio(audio, sr)
            
            # Final limiting
            max_val = np.max(np.abs(audio))
            if max_val > 0.99:
                audio = audio * (0.99 / max_val)
            
            # Save enhanced audio
            sf.write(output_path, audio, sr, subtype='PCM_16')
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error enhancing TTS output: {str(e)}")
            return False


def process_folder(input_folder: str, output_folder: str,
                  processor: Optional[PodcastAudioProcessor] = None,
                  file_extensions: tuple = ('.mp3', '.wav', '.m4a', '.flac')):
    """
    Process all audio files in a folder
    
    Args:
        input_folder: Folder containing raw audio samples
        output_folder: Folder to save processed samples
        processor: PodcastAudioProcessor instance (creates new if None)
        file_extensions: Tuple of audio file extensions to process
    """
    if processor is None:
        processor = PodcastAudioProcessor()
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))
        audio_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not audio_files:
        print(f"❌ No audio files found in {input_folder}")
        return
    
    print(f"🎙️  Processing {len(audio_files)} audio file(s) for podcast quality")
    print(f"📁 Input:  {input_folder}")
    print(f"📁 Output: {output_folder}")
    print("="*70)
    
    success_count = 0
    failed_files = []
    
    for i, audio_file in enumerate(audio_files, 1):
        # Generate output filename (always save as WAV for best quality)
        output_filename = audio_file.stem + '.wav'
        output_file = output_path / output_filename
        
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        if processor.process_sample(str(audio_file), str(output_file)):
            # Get file size
            size_kb = output_file.stat().st_size / 1024
            print(f"   ✅ Enhanced: {output_filename} ({size_kb:.1f} KB)")
            success_count += 1
        else:
            failed_files.append(audio_file.name)
    
    # Summary
    print("\n" + "="*70)
    print(f"✨ Processing Complete!")
    print(f"   ✅ Successful: {success_count}/{len(audio_files)}")
    
    if failed_files:
        print(f"   ❌ Failed: {len(failed_files)}")
        print("\nFailed files:")
        for filename in failed_files:
            print(f"   • {filename}")


if __name__ == "__main__":
    # Example usage
    print("🎙️  Professional Audio Enhancement for Voice Cloning")
    print("="*70)
    
    # Process raw audio samples
    INPUT_FOLDER = "D:\\voice cloning\\voice_profiles\\pritam\\samples"
    OUTPUT_FOLDER = "D:\\voice cloning\\voice_profiles\\pritam\\samples_enhanced"
    
    processor = PodcastAudioProcessor(target_sr=24000)
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, processor)
    
    print(f"\n✅ Enhanced samples saved to: {OUTPUT_FOLDER}")
    print("\n📝 Next steps:")
    print("   1. Use the enhanced samples for better voice cloning results")
    print("   2. Apply post-processing to TTS output using process_tts_output()")
    print("   3. Enjoy podcast-quality audio! 🎧")
