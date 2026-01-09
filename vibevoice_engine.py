"""
🎙️ VibeVoice Hindi TTS Engine
================================

Seamless integration of VibeVoice-Hindi-7B with existing voice cloning infrastructure.

Features:
✅ Native Hindi + English support with natural prosody
✅ Voice cloning from 10-30 second reference samples
✅ Emotion-aware generation (integrates with emotion_analyzer.py)
✅ Natural prosody processing (integrates with prosody_processor.py)
✅ Long-form generation (up to 45 minutes context)
✅ Optimized for T2000 4GB VRAM with CPU offloading

Usage:
    from vibevoice_engine import VibeVoiceEngine
    
    engine = VibeVoiceEngine()
    audio = engine.generate(
        text="नमस्ते! आज का मौसम बहुत अच्छा है।",
        reference_audio="path/to/voice_sample.wav",
        emotion="conversational"
    )
"""

# IMPORTANT: Configure model paths to D: drive BEFORE importing ML libraries
import model_paths  # This sets HF_HOME, TORCH_HOME, etc. to D: drive

import os
import sys
import gc
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Literal
from dataclasses import dataclass
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VibeVoiceConfig:
    """Configuration for VibeVoice engine"""
    # Model settings
    model_path: str = "D:/voice cloning/models_cache/vibevoice-hindi-7b"
    fallback_model: str = "tarun7r/vibevoice-hindi-7b"
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_flash_attention: bool = True  # Use flash_attention_2 if available
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Generation settings
    cfg_scale: float = 1.3  # Classifier-free guidance scale
    ddpm_steps: int = 10  # Diffusion steps (higher = better quality, slower)
    max_tokens: Optional[int] = None  # Auto-detect based on text length
    
    # Voice cloning settings
    enable_prefill: bool = True  # Enable voice cloning via prefill
    
    # Memory optimization for T2000 4GB VRAM
    enable_cpu_offload: bool = True
    low_vram_mode: bool = True
    clear_cache_after_gen: bool = True
    
    # Output settings
    sample_rate: int = 24000
    output_dir: str = "D:/voice cloning/audio_output"


# ============================================================================
# VIBEVOICE ENGINE
# ============================================================================

class VibeVoiceEngine:
    """
    VibeVoice Hindi TTS Engine with emotion and prosody integration
    
    Integrates seamlessly with existing:
    - emotion_analyzer.py for emotion detection
    - prosody_processor.py for natural speech flow
    - audio_enhancer.py for output enhancement
    """
    
    def __init__(self, config: Optional[VibeVoiceConfig] = None):
        """
        Initialize VibeVoice engine
        
        Args:
            config: Optional configuration object
        """
        self.config = config or VibeVoiceConfig()
        self.model = None
        self.processor = None
        self._initialized = False
        
        # Setup CUDA optimizations
        self._setup_cuda()
        
        # Import optional integrations
        self._load_integrations()
        
        print(f"🎙️ VibeVoice Engine initialized")
        print(f"   Model: {self.config.model_path}")
        print(f"   Device: {self.config.device}")
        print(f"   VRAM Mode: {'Low (4GB)' if self.config.low_vram_mode else 'Standard'}")
    
    def _setup_cuda(self):
        """Configure CUDA for optimal performance on T2000 4GB"""
        if torch.cuda.is_available():
            # Enable TF32 for faster matrix operations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Memory optimization for 4GB VRAM
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
    
    def _load_integrations(self):
        """Load optional integration modules"""
        # Emotion analyzer
        try:
            from emotion_analyzer import EmotionAnalyzer
            self.emotion_analyzer = EmotionAnalyzer(mode="rule-based")
            self._has_emotion_analyzer = True
            print("   ✅ Emotion Analyzer: Loaded")
        except ImportError:
            self.emotion_analyzer = None
            self._has_emotion_analyzer = False
            print("   ⚠️ Emotion Analyzer: Not available")
        
        # Prosody processor
        try:
            from prosody_processor import ProsodyProcessor
            self.prosody_processor = ProsodyProcessor()
            self._has_prosody_processor = True
            print("   ✅ Prosody Processor: Loaded")
        except ImportError:
            self.prosody_processor = None
            self._has_prosody_processor = False
            print("   ⚠️ Prosody Processor: Not available")
        
        # Audio enhancer
        try:
            from audio_enhancer import AudioEnhancer
            self.audio_enhancer = AudioEnhancer()
            self._has_audio_enhancer = True
            print("   ✅ Audio Enhancer: Loaded")
        except ImportError:
            self.audio_enhancer = None
            self._has_audio_enhancer = False
            print("   ⚠️ Audio Enhancer: Not available")
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load VibeVoice model and processor
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loaded successfully
        """
        if self._initialized and not force_reload:
            print("✅ Model already loaded")
            return True
        
        print("🔄 Loading VibeVoice model...")
        start_time = time.time()
        
        try:
            # Import VibeVoice components
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            
            # Determine model path
            model_path = self.config.model_path
            if not os.path.exists(model_path):
                print(f"   Local model not found, using HuggingFace: {self.config.fallback_model}")
                model_path = self.config.fallback_model
            
            # Load processor
            print("   Loading processor...")
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Determine attention implementation
            if self.config.use_flash_attention and self.config.device == "cuda":
                attn_impl = "flash_attention_2"
            else:
                attn_impl = "sdpa"
            
            # Load model with memory optimizations
            print(f"   Loading model (attention: {attn_impl})...")
            
            try:
                if self.config.device == "cuda":
                    if self.config.low_vram_mode:
                        # For 4GB VRAM: Load to CPU first, then selective GPU
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=self.config.dtype,
                            device_map="auto",  # Auto device mapping with CPU offload
                            attn_implementation=attn_impl,
                            low_cpu_mem_usage=True,
                        )
                    else:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=self.config.dtype,
                            device_map="cuda",
                            attn_implementation=attn_impl,
                        )
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        attn_implementation="sdpa",
                    )
            except Exception as e:
                # Fallback to SDPA if flash_attention fails
                if "flash" in str(e).lower():
                    print(f"   ⚠️ Flash attention not available, falling back to SDPA")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        torch_dtype=self.config.dtype,
                        device_map="auto" if self.config.device == "cuda" else "cpu",
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=True,
                    )
                else:
                    raise e
            
            # Set model to eval mode
            self.model.eval()
            
            # Configure DDPM steps
            self.model.set_ddpm_inference_steps(num_steps=self.config.ddpm_steps)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        emotion: Optional[str] = None,
        auto_detect_emotion: bool = True,
        apply_prosody: bool = True,
        enhance_output: bool = True,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate speech from text with voice cloning
        
        Args:
            text: Text to synthesize (Hindi, English, or mixed)
            reference_audio: Path to reference audio for voice cloning (10-30s)
            emotion: Target emotion (neutral, excited, calm, dramatic, conversational, storytelling)
            auto_detect_emotion: Auto-detect emotion from text if emotion not specified
            apply_prosody: Apply prosody processing for natural flow
            enhance_output: Apply audio enhancement post-processing
            cfg_scale: Override CFG scale (default: 1.3)
            seed: Random seed for reproducibility
            output_path: Optional path to save output audio
            
        Returns:
            Dict with 'audio' (numpy array), 'sample_rate', 'path' (if saved), 'metadata'
        """
        # Ensure model is loaded
        if not self._initialized:
            if not self.load_model():
                return None
        
        print(f"\n🎙️ Generating speech...")
        print(f"   Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        # Auto-detect emotion if needed
        if emotion is None and auto_detect_emotion and self._has_emotion_analyzer:
            emotion = self.emotion_analyzer.analyze(text)
            print(f"   Auto-detected emotion: {emotion}")
        elif emotion:
            print(f"   Emotion: {emotion}")
        
        # Apply prosody markers if available
        processed_text = text
        if apply_prosody and self._has_prosody_processor:
            try:
                processed_text = self.prosody_processor.process(text, emotion=emotion)
                print(f"   Prosody processing applied")
            except Exception as e:
                print(f"   ⚠️ Prosody processing failed: {e}")
                processed_text = text
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"   Seed: {seed}")
        
        # Prepare voice sample
        voice_samples = []
        if reference_audio and os.path.exists(reference_audio):
            voice_samples = [reference_audio]
            print(f"   Reference: {os.path.basename(reference_audio)}")
        else:
            print(f"   ⚠️ No reference audio - using default voice")
        
        try:
            # Prepare script format for VibeVoice
            script = f"Speaker 1: {processed_text}"
            
            # Prepare inputs
            inputs = self.processor(
                text=[script],
                voice_samples=[voice_samples] if voice_samples else None,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.config.device)
            
            # Generate
            start_time = time.time()
            
            gen_cfg_scale = cfg_scale or self.config.cfg_scale
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    cfg_scale=gen_cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                    is_prefill=self.config.enable_prefill and len(voice_samples) > 0,
                )
            
            generation_time = time.time() - start_time
            print(f"   Generation time: {generation_time:.2f}s")
            
            # Extract audio from outputs
            if hasattr(outputs, 'audios'):
                audio_array = outputs.audios[0]
            elif isinstance(outputs, dict) and 'audio' in outputs:
                audio_array = outputs['audio']
            elif isinstance(outputs, torch.Tensor):
                audio_array = outputs.cpu().numpy()
            else:
                # Try to extract from the output directly
                audio_array = np.array(outputs)
            
            # Ensure correct shape
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(1, -1)
            
            # Apply audio enhancement if available
            if enhance_output and self._has_audio_enhancer:
                try:
                    audio_tensor = torch.from_numpy(audio_array).float()
                    enhanced = self.audio_enhancer.enhance(audio_tensor, self.config.sample_rate)
                    audio_array = enhanced.numpy()
                    print(f"   Audio enhancement applied")
                except Exception as e:
                    print(f"   ⚠️ Enhancement failed: {e}")
            
            # Calculate audio duration
            duration = audio_array.shape[-1] / self.config.sample_rate
            rtf = generation_time / duration if duration > 0 else 0
            print(f"   Duration: {duration:.2f}s (RTF: {rtf:.2f}x)")
            
            # Save output if path provided
            saved_path = None
            if output_path:
                saved_path = self._save_audio(audio_array, output_path)
                print(f"   Saved: {saved_path}")
            
            # Clear CUDA cache
            if self.config.clear_cache_after_gen and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return {
                'audio': audio_array,
                'sample_rate': self.config.sample_rate,
                'path': saved_path,
                'metadata': {
                    'text': text,
                    'emotion': emotion,
                    'duration': duration,
                    'generation_time': generation_time,
                    'rtf': rtf,
                    'reference': reference_audio,
                    'cfg_scale': gen_cfg_scale,
                    'seed': seed,
                }
            }
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return None
    
    def _save_audio(self, audio: np.ndarray, path: str) -> str:
        """Save audio array to file"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Convert to torch tensor for torchaudio
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure 2D shape
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        torchaudio.save(path, audio_tensor, self.config.sample_rate)
        return path
    
    def generate_long_form(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        chunk_size: int = 500,
        overlap: int = 50,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate long-form audio by chunking text
        
        Args:
            text: Long text to synthesize
            reference_audio: Path to reference audio
            chunk_size: Maximum characters per chunk
            overlap: Overlap characters for smooth transitions
            **kwargs: Additional args passed to generate()
            
        Returns:
            Combined audio result
        """
        # VibeVoice supports up to 45 minutes, but chunk for very long texts
        if len(text) <= chunk_size * 2:
            return self.generate(text, reference_audio, **kwargs)
        
        print(f"📜 Long-form generation ({len(text)} chars)")
        
        # Split into sentences first
        import re
        sentences = re.split(r'(?<=[.!?।॥])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"   Split into {len(chunks)} chunks")
        
        # Generate each chunk
        all_audio = []
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i+1}/{len(chunks)}")
            result = self.generate(chunk, reference_audio, **kwargs)
            if result and result['audio'] is not None:
                all_audio.append(result['audio'])
        
        if not all_audio:
            return None
        
        # Concatenate with crossfade
        combined = self._crossfade_concat(all_audio)
        
        duration = combined.shape[-1] / self.config.sample_rate
        
        return {
            'audio': combined,
            'sample_rate': self.config.sample_rate,
            'metadata': {
                'text': text,
                'duration': duration,
                'chunks': len(chunks),
            }
        }
    
    def _crossfade_concat(self, audio_chunks: List[np.ndarray], fade_ms: int = 50) -> np.ndarray:
        """Concatenate audio chunks with crossfade"""
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        fade_samples = int(self.config.sample_rate * fade_ms / 1000)
        
        result = audio_chunks[0]
        for chunk in audio_chunks[1:]:
            # Ensure same shape
            if result.ndim != chunk.ndim:
                if result.ndim == 1:
                    result = result.reshape(1, -1)
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
            
            # Simple concatenation with short crossfade
            if result.shape[-1] > fade_samples and chunk.shape[-1] > fade_samples:
                # Create fade curves
                fade_out = np.linspace(1, 0, fade_samples)
                fade_in = np.linspace(0, 1, fade_samples)
                
                # Apply fades
                result[..., -fade_samples:] *= fade_out
                chunk_faded = chunk.copy()
                chunk_faded[..., :fade_samples] *= fade_in
                
                # Overlap-add
                result[..., -fade_samples:] += chunk_faded[..., :fade_samples]
                result = np.concatenate([result, chunk_faded[..., fade_samples:]], axis=-1)
            else:
                result = np.concatenate([result, chunk], axis=-1)
        
        return result
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Model unloaded")
    
    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions"""
        if self._has_emotion_analyzer:
            return self.emotion_analyzer.EMOTIONS
        return ["neutral", "excited", "calm", "dramatic", "conversational", "storytelling"]


# ============================================================================
# UNIFIED TTS INTERFACE
# ============================================================================

class UnifiedTTSEngine:
    """
    Unified interface for multiple TTS backends
    
    Supports:
    - VibeVoice (Hindi/English with natural prosody)
    - Chatterbox (English with emotion control)
    - F5-TTS (fast synthesis)
    
    Automatically selects best engine based on language and requirements
    """
    
    def __init__(self):
        self.engines = {}
        self._current_engine = None
        
        print("🔊 Unified TTS Engine")
    
    def load_engine(self, engine_name: Literal["vibevoice", "chatterbox", "f5"]) -> bool:
        """Load a specific TTS engine"""
        if engine_name in self.engines:
            self._current_engine = engine_name
            return True
        
        try:
            if engine_name == "vibevoice":
                engine = VibeVoiceEngine()
                if engine.load_model():
                    self.engines["vibevoice"] = engine
                    self._current_engine = "vibevoice"
                    return True
            elif engine_name == "chatterbox":
                from myvoiceclone import VoiceCloneSystem
                self.engines["chatterbox"] = VoiceCloneSystem()
                self._current_engine = "chatterbox"
                return True
            elif engine_name == "f5":
                from f5_engine import F5TTSEngine
                self.engines["f5"] = F5TTSEngine()
                self._current_engine = "f5"
                return True
        except Exception as e:
            print(f"❌ Failed to load {engine_name}: {e}")
        
        return False
    
    def generate(
        self,
        text: str,
        engine: Optional[str] = None,
        auto_select: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate speech using specified or auto-selected engine
        
        Args:
            text: Text to synthesize
            engine: Engine to use (vibevoice, chatterbox, f5)
            auto_select: Auto-select best engine for the text
            **kwargs: Engine-specific arguments
        """
        # Auto-select engine based on text
        if engine is None and auto_select:
            engine = self._select_engine(text)
        
        engine = engine or self._current_engine or "vibevoice"
        
        # Load engine if needed
        if engine not in self.engines:
            if not self.load_engine(engine):
                return None
        
        # Generate
        return self.engines[engine].generate(text, **kwargs)
    
    def _select_engine(self, text: str) -> str:
        """Auto-select best engine based on text content"""
        # Check for Hindi characters
        hindi_pattern = r'[\u0900-\u097F]'
        import re
        
        if re.search(hindi_pattern, text):
            return "vibevoice"  # Best for Hindi
        
        # Default to vibevoice for natural prosody
        return "vibevoice"
    
    def unload_all(self):
        """Unload all engines"""
        for name, engine in self.engines.items():
            if hasattr(engine, 'unload_model'):
                engine.unload_model()
        self.engines.clear()
        self._current_engine = None


# ============================================================================
# CLI AND TESTING
# ============================================================================

def test_vibevoice():
    """Test VibeVoice engine"""
    print("\n" + "="*60)
    print("🧪 VibeVoice Engine Test")
    print("="*60)
    
    # Initialize engine
    engine = VibeVoiceEngine()
    
    # Load model
    if not engine.load_model():
        print("❌ Failed to load model")
        return
    
    # Test Hindi text
    hindi_text = "नमस्ते! मेरा नाम प्रीतम है। आज का मौसम बहुत अच्छा है।"
    
    print("\n📝 Testing Hindi generation...")
    result = engine.generate(
        text=hindi_text,
        emotion="conversational",
        output_path="D:/voice cloning/audio_output/vibevoice_test_hindi.wav"
    )
    
    if result:
        print(f"✅ Hindi test passed! Duration: {result['metadata']['duration']:.2f}s")
    else:
        print("❌ Hindi test failed")
    
    # Test English text
    english_text = "Hello! This is a test of the VibeVoice text-to-speech engine."
    
    print("\n📝 Testing English generation...")
    result = engine.generate(
        text=english_text,
        emotion="conversational",
        output_path="D:/voice cloning/audio_output/vibevoice_test_english.wav"
    )
    
    if result:
        print(f"✅ English test passed! Duration: {result['metadata']['duration']:.2f}s")
    else:
        print("❌ English test failed")
    
    # Cleanup
    engine.unload_model()
    print("\n✅ Tests complete!")


if __name__ == "__main__":
    test_vibevoice()
