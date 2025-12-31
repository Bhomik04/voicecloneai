"""
F5-TTS Engine - Ultra Fast Voice Cloning (5-10x faster)
========================================================

F5-TTS uses diffusion-based synthesis which is much faster than 
autoregressive models like ChatterboxTTS.

Key advantages:
- 5-10x faster generation
- Good voice quality
- Works with same voice samples as ChatterboxTTS
- Lower VRAM usage

Note: F5-TTS is best for English. For Hindi, ChatterboxTTS is still recommended.
"""

# Configure model paths to D: drive BEFORE importing any ML libraries
import model_paths

import os
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import tempfile


class F5TTSEngine:
    """
    F5-TTS wrapper for fast voice cloning
    
    Uses diffusion-based TTS for 5-10x faster generation compared to
    autoregressive models. Best for English generation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        cache_dir: str = r"D:\voice cloning\models_cache\f5tts",
    ):
        """
        Initialize F5-TTS engine
        
        Args:
            device: 'cuda' or 'cpu'
            cache_dir: Directory for model cache (on D: drive)
        """
        self.device = device
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Model (lazy loaded)
        self.model = None
        self.vocoder = None
        self.loaded = False
        
        print(f"⚡ F5-TTS Engine initialized (cache: {cache_dir})")
        
    def load_model(self):
        """Load F5-TTS model (downloads to D: drive on first run)"""
        if self.loaded:
            return
            
        print("🔄 Loading F5-TTS model...")
        print(f"   Cache directory: {self.cache_dir}")
        
        try:
            from f5_tts.api import F5TTS
            
            # Initialize F5-TTS
            self.model = F5TTS(device=self.device)
            
            self.loaded = True
            print("✅ F5-TTS model loaded successfully!")
            
        except ImportError as e:
            print(f"❌ F5-TTS not installed: {e}")
            print("   Install with: pip install f5-tts")
            raise
        except Exception as e:
            print(f"❌ Error loading F5-TTS: {e}")
            raise
    
    def generate(
        self,
        text: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        output_sr: int = 24000,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech using F5-TTS
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file
            reference_text: Transcript of reference audio (auto-detected if None)
            speed: Speed factor (0.5 = slow, 1.0 = normal, 2.0 = fast)
            output_sr: Output sample rate
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        # Load model if needed
        if not self.loaded:
            self.load_model()
        
        print(f"⚡ F5-TTS generating: '{text[:50]}...'")
        
        try:
            # Generate audio
            audio, sr, _ = self.model.infer(
                ref_file=reference_audio,
                ref_text=reference_text if reference_text else "",
                gen_text=text,
                speed=speed,
            )
            
            # Convert to torch tensor
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            
            # Ensure proper shape [1, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                
            # Resample to target sample rate if needed
            if sr != output_sr:
                resampler = ta.transforms.Resample(sr, output_sr)
                audio = resampler(audio)
                sr = output_sr
                
            print(f"   ✅ Generated {audio.shape[-1] / sr:.2f}s of audio")
            
            return audio, sr
            
        except Exception as e:
            print(f"❌ F5-TTS generation failed: {e}")
            raise
    
    def generate_from_profile(
        self,
        text: str,
        profile_samples: list,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech using a voice profile (list of samples)
        
        Uses the first sample as reference. For better quality,
        consider concatenating multiple samples.
        
        Args:
            text: Text to synthesize
            profile_samples: List of audio file paths from profile
            reference_text: Optional transcript
            speed: Speed factor
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        if not profile_samples:
            raise ValueError("Profile must have at least one sample")
            
        # Use first sample as reference
        reference_audio = profile_samples[0]
        
        return self.generate(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            speed=speed,
        )
    
    def generate_long(
        self,
        text: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
        chunk_size: int = 200,  # characters per chunk
        speed: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate long-form audio by chunking text
        
        F5-TTS works best with shorter segments. This method
        chunks long text and concatenates the results.
        
        Args:
            text: Long text to synthesize
            reference_audio: Path to reference audio
            reference_text: Optional transcript
            chunk_size: Characters per chunk
            speed: Speed factor
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        import re
        
        # Split text into chunks at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        print(f"⚡ F5-TTS long-form: {len(chunks)} chunks")
        
        # Generate each chunk
        audio_chunks = []
        sr = None
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   [{i}/{len(chunks)}] '{chunk[:40]}...'")
            audio, sr = self.generate(
                text=chunk,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speed=speed,
            )
            audio_chunks.append(audio)
            
        # Concatenate with crossfade
        if len(audio_chunks) == 1:
            return audio_chunks[0], sr
            
        combined = self._crossfade_concat(audio_chunks, sr)
        
        return combined, sr
    
    def _crossfade_concat(
        self,
        chunks: list,
        sr: int,
        crossfade_ms: int = 50,
    ) -> torch.Tensor:
        """Concatenate audio chunks with smooth crossfade"""
        crossfade_samples = int(crossfade_ms * sr / 1000)
        
        result = chunks[0]
        
        for chunk in chunks[1:]:
            # Create crossfade
            if crossfade_samples > 0 and result.shape[-1] > crossfade_samples and chunk.shape[-1] > crossfade_samples:
                # Fade out end of result
                fade_out = torch.linspace(1, 0, crossfade_samples)
                result[..., -crossfade_samples:] *= fade_out
                
                # Fade in start of chunk
                fade_in = torch.linspace(0, 1, crossfade_samples)
                chunk[..., :crossfade_samples] *= fade_in
                
                # Overlap-add
                result[..., -crossfade_samples:] += chunk[..., :crossfade_samples]
                result = torch.cat([result, chunk[..., crossfade_samples:]], dim=-1)
            else:
                result = torch.cat([result, chunk], dim=-1)
                
        return result
    
    def get_speed_for_rtf(self, target_rtf: float = 5.0) -> float:
        """
        Calculate speed setting to achieve target Real-Time Factor
        
        Args:
            target_rtf: Target RTF (e.g., 5.0 means 5x real-time)
            
        Returns:
            Speed setting to use
        """
        # F5-TTS default is about 5-10x RTF
        # Adjust speed to fine-tune
        return 1.0  # Default is usually good
        
    def benchmark(self, text: str = None, reference_audio: str = None) -> dict:
        """
        Benchmark F5-TTS generation speed
        
        Args:
            text: Test text (uses default if None)
            reference_audio: Reference audio path
            
        Returns:
            Benchmark results dict
        """
        import time
        
        if text is None:
            text = "This is a benchmark test for F5-TTS generation speed."
            
        if reference_audio is None:
            print("⚠️ No reference audio provided for benchmark")
            return {}
            
        # Load model
        self.load_model()
        
        # Warm up
        print("🔥 Warming up...")
        _ = self.generate(text[:20], reference_audio)
        
        # Benchmark
        print("⏱️ Running benchmark...")
        
        start_time = time.time()
        audio, sr = self.generate(text, reference_audio)
        generation_time = time.time() - start_time
        
        audio_duration = audio.shape[-1] / sr
        rtf = audio_duration / generation_time
        
        results = {
            'text_length': len(text),
            'audio_duration_s': audio_duration,
            'generation_time_s': generation_time,
            'rtf': rtf,  # Real-Time Factor
            'speedup': f"{rtf:.1f}x real-time",
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Text: {len(text)} characters")
        print(f"   Audio: {audio_duration:.2f}s")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Speed: {rtf:.1f}x real-time 🚀")
        
        return results


def test_f5tts():
    """Test F5-TTS engine"""
    import os
    import glob
    
    print("⚡ Testing F5-TTS Engine\n")
    
    # Find a voice sample
    profile_dir = "voice_profiles/pritam/samples"
    
    if not os.path.exists(profile_dir):
        print(f"❌ Profile directory not found: {profile_dir}")
        print("   Create a profile first or update the path")
        return
        
    samples = glob.glob(os.path.join(profile_dir, "*.wav"))
    
    if not samples:
        print(f"❌ No samples found in {profile_dir}")
        return
        
    print(f"📂 Found {len(samples)} voice samples")
    print(f"   Using: {samples[0]}")
    
    # Initialize engine
    engine = F5TTSEngine(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test generation
    test_text = "Hello! This is a test of the F5-TTS engine for fast voice cloning."
    
    try:
        print(f"\n🎙️ Generating: '{test_text}'")
        audio, sr = engine.generate(test_text, samples[0])
        
        # Save output
        output_path = "test_f5tts_output.wav"
        ta.save(output_path, audio, sr)
        
        duration = audio.shape[-1] / sr
        print(f"\n✅ Success!")
        print(f"   Output: {output_path}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sr}Hz")
        
        # Run benchmark
        print("\n📊 Running benchmark...")
        engine.benchmark(test_text, samples[0])
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_f5tts()
