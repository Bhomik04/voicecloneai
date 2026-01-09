"""
🎙️ Unified TTS Compatibility Layer
====================================

Provides seamless switching between Chatterbox and VibeVoice TTS engines
while handling their conflicting dependencies gracefully.

Features:
✅ Unified API for both TTS engines
✅ Automatic engine selection based on language/requirements
✅ Voice profile compatibility across engines
✅ Shared emotion and prosody processing
✅ Graceful fallback when one engine is unavailable

Dependencies:
- Chatterbox: transformers==4.46.3, gradio==5.44.1
- VibeVoice: transformers==4.51.3, gradio==5.50.0

Note: Due to conflicting versions, only one engine can be fully loaded at a time.
The compatibility layer handles lazy loading and switching.

Usage:
    from tts_compatibility import UnifiedTTS
    
    tts = UnifiedTTS()
    
    # Auto-select best engine
    audio = tts.generate("Hello world", profile="pritam")
    
    # Force specific engine
    audio = tts.generate("नमस्ते", profile="pritam", engine="vibevoice")
"""

# IMPORTANT: Configure model paths BEFORE importing ML libraries
import model_paths

import os
import sys
import gc
import re
import json
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

class TTSEngine(Enum):
    """Available TTS engines"""
    CHATTERBOX = "chatterbox"
    VIBEVOICE = "vibevoice"
    AUTO = "auto"  # Auto-select based on text


@dataclass
class TTSConfig:
    """Unified TTS configuration"""
    # Engine settings
    default_engine: TTSEngine = TTSEngine.AUTO
    fallback_engine: TTSEngine = TTSEngine.CHATTERBOX
    
    # Voice profile settings
    profiles_dir: str = "D:/voice cloning/voice_profiles"
    
    # Generation settings
    default_emotion: str = "conversational"
    auto_detect_emotion: bool = True
    apply_prosody: bool = True
    enhance_audio: bool = True
    
    # Hardware optimization (T2000 4GB VRAM)
    device: str = "cuda"
    use_fp16: bool = True
    low_vram_mode: bool = True
    clear_cache_after_gen: bool = True
    
    # Output settings
    output_dir: str = "D:/voice cloning/audio_output"
    sample_rate: int = 24000
    
    # VibeVoice specific
    vibevoice_model_path: str = "D:/voice cloning/models_cache/vibevoice-hindi-7b"
    vibevoice_cfg_scale: float = 1.3
    
    # Chatterbox specific
    chatterbox_exaggeration: float = 0.5
    chatterbox_cfg_weight: float = 0.22


@dataclass
class GenerationResult:
    """Unified result from TTS generation"""
    audio: np.ndarray
    sample_rate: int
    engine_used: str
    duration: float
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def rtf(self) -> float:
        """Real-time factor (generation_time / duration)"""
        return self.generation_time / self.duration if self.duration > 0 else 0
    
    def save(self, path: str) -> str:
        """Save audio to file"""
        import torchaudio
        import torch
        
        audio_tensor = torch.from_numpy(self.audio).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torchaudio.save(path, audio_tensor, self.sample_rate)
        return path


# ============================================================================
# VOICE PROFILE MANAGER
# ============================================================================

class VoiceProfileManager:
    """
    Unified voice profile management for both engines
    
    Reads existing Chatterbox profiles and makes them compatible with VibeVoice
    """
    
    def __init__(self, profiles_dir: str = "D:/voice cloning/voice_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles: Dict[str, Dict] = {}
        self._scan_profiles()
    
    def _scan_profiles(self):
        """Scan for available voice profiles"""
        if not self.profiles_dir.exists():
            print(f"⚠️ Profiles directory not found: {self.profiles_dir}")
            return
        
        for profile_dir in self.profiles_dir.iterdir():
            if profile_dir.is_dir() and not profile_dir.name.startswith('.'):
                metadata_path = profile_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Find audio samples
                        samples_dir = profile_dir / "samples"
                        enhanced_dir = profile_dir / "samples_enhanced"
                        
                        samples = []
                        if enhanced_dir.exists():
                            samples = list(enhanced_dir.glob("*.wav"))
                        elif samples_dir.exists():
                            samples = list(samples_dir.glob("*.wav"))
                        
                        self.profiles[profile_dir.name] = {
                            "name": profile_dir.name,
                            "path": str(profile_dir),
                            "metadata": metadata,
                            "samples": [str(s) for s in samples],
                            "languages": metadata.get("languages", ["en"]),
                        }
                    except Exception as e:
                        print(f"⚠️ Failed to load profile {profile_dir.name}: {e}")
        
        print(f"📁 Found {len(self.profiles)} voice profiles")
    
    def get_profile(self, name: str) -> Optional[Dict]:
        """Get profile by name"""
        # Try exact match first
        if name in self.profiles:
            return self.profiles[name]
        
        # Try case-insensitive match
        for profile_name in self.profiles:
            if profile_name.lower() == name.lower():
                return self.profiles[profile_name]
        
        return None
    
    def get_best_reference(self, name: str) -> Optional[str]:
        """Get best reference audio for a profile"""
        profile = self.get_profile(name)
        if not profile or not profile["samples"]:
            return None
        
        # Look for enhanced samples first
        for sample in profile["samples"]:
            if "enhanced" in sample.lower():
                return sample
        
        # Look for best sample marker
        for sample in profile["samples"]:
            if "best" in sample.lower() or "sample_012" in sample.lower():
                return sample
        
        # Return first sample as fallback
        return profile["samples"][0]
    
    def list_profiles(self) -> List[str]:
        """List available profile names"""
        return list(self.profiles.keys())
    
    def get_profile_languages(self, name: str) -> List[str]:
        """Get languages supported by a profile"""
        profile = self.get_profile(name)
        return profile["languages"] if profile else ["en"]


# ============================================================================
# BASE TTS ENGINE INTERFACE
# ============================================================================

class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine can be loaded"""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if engine is currently loaded"""
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """Load the engine models"""
        pass
    
    @abstractmethod
    def unload(self):
        """Unload the engine to free memory"""
        pass
    
    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: str,
        emotion: str = "neutral",
        **kwargs
    ) -> Optional[GenerationResult]:
        """Generate speech"""
        pass
    
    @abstractmethod
    def supports_language(self, lang: str) -> bool:
        """Check if engine supports a language"""
        pass


# ============================================================================
# CHATTERBOX ENGINE WRAPPER
# ============================================================================

class ChatterboxEngineWrapper(BaseTTSEngine):
    """Wrapper for Chatterbox TTS engine"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._model = None
        self._loaded = False
        self._available = None
    
    @property
    def name(self) -> str:
        return "chatterbox"
    
    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                # Check if chatterbox is importable
                sys.path.insert(0, str(Path("D:/voice cloning/chatterbox/src")))
                from chatterbox.tts import ChatterboxTTS
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None
    
    def load(self) -> bool:
        if self._loaded:
            return True
        
        if not self.is_available:
            print("❌ Chatterbox not available")
            return False
        
        try:
            import torch
            print("🔄 Loading Chatterbox...")
            
            sys.path.insert(0, str(Path("D:/voice cloning/chatterbox/src")))
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            device = self.config.device
            
            # Load English model
            self._english_model = ChatterboxTTS.from_pretrained(device=device)
            
            # Load multilingual model for Hindi/Hinglish
            self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
                device=device
            )
            
            self._model = self._english_model  # Primary model reference
            self._loaded = True
            print("✅ Chatterbox loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Chatterbox: {e}")
            return False
    
    def unload(self):
        if hasattr(self, '_english_model'):
            del self._english_model
        if hasattr(self, '_multilingual_model'):
            del self._multilingual_model
        self._model = None
        self._loaded = False
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Chatterbox unloaded")
    
    def generate(
        self,
        text: str,
        reference_audio: str,
        emotion: str = "neutral",
        language: Optional[str] = None,
        **kwargs
    ) -> Optional[GenerationResult]:
        if not self.is_loaded:
            if not self.load():
                return None
        
        import torch
        import torchaudio
        
        start_time = time.time()
        
        try:
            # Detect language if not specified
            if language is None:
                language = self._detect_language(text)
            
            # Get emotion preset
            emotion_preset = self._get_emotion_preset(emotion)
            
            # Select appropriate model
            if language in ["hi", "hi-IN", "hinglish"]:
                model = self._multilingual_model
                # Generate with multilingual model
                with torch.inference_mode():
                    wav = model.generate(
                        text=text,
                        language_id="hi-IN" if language in ["hi", "hi-IN"] else "en-US",
                        audio_prompt_path=reference_audio,
                        exaggeration=emotion_preset["exaggeration"],
                        cfg_weight=emotion_preset["cfg_weight"],
                        temperature=emotion_preset["temperature"],
                    )
            else:
                model = self._english_model
                # Generate with English model
                with torch.inference_mode():
                    wav = model.generate(
                        text=text,
                        audio_prompt_path=reference_audio,
                        exaggeration=emotion_preset["exaggeration"],
                        cfg_weight=emotion_preset["cfg_weight"],
                        temperature=emotion_preset["temperature"],
                    )
            
            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                audio_array = wav.cpu().numpy()
            else:
                audio_array = np.array(wav)
            
            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(1, -1)
            
            generation_time = time.time() - start_time
            duration = audio_array.shape[-1] / model.sr
            
            # Clear cache
            if self.config.clear_cache_after_gen and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return GenerationResult(
                audio=audio_array,
                sample_rate=model.sr,
                engine_used="chatterbox",
                duration=duration,
                generation_time=generation_time,
                metadata={
                    "text": text,
                    "emotion": emotion,
                    "language": language,
                    "reference": reference_audio,
                }
            )
            
        except Exception as e:
            print(f"❌ Chatterbox generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def supports_language(self, lang: str) -> bool:
        return lang.lower() in ["en", "hi", "hi-in", "hinglish", "english", "hindi"]
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text"""
        # Check for Hindi characters
        hindi_pattern = r'[\u0900-\u097F]'
        if re.search(hindi_pattern, text):
            return "hi"
        
        # Check for common Hinglish words
        hinglish_words = {'aur', 'hai', 'main', 'mein', 'kya', 'nahi', 'haan', 'bhi', 'toh'}
        words = set(text.lower().split())
        if words & hinglish_words:
            return "hinglish"
        
        return "en"
    
    def _get_emotion_preset(self, emotion: str) -> Dict:
        """Get emotion preset parameters"""
        presets = {
            "neutral": {"exaggeration": 0.4, "temperature": 0.6, "cfg_weight": 0.22},
            "excited": {"exaggeration": 0.7, "temperature": 0.75, "cfg_weight": 0.25},
            "calm": {"exaggeration": 0.3, "temperature": 0.5, "cfg_weight": 0.25},
            "dramatic": {"exaggeration": 0.9, "temperature": 0.8, "cfg_weight": 0.23},
            "conversational": {"exaggeration": 0.5, "temperature": 0.7, "cfg_weight": 0.20},
            "storytelling": {"exaggeration": 0.6, "temperature": 0.7, "cfg_weight": 0.24},
        }
        return presets.get(emotion, presets["neutral"])


# ============================================================================
# VIBEVOICE ENGINE WRAPPER
# ============================================================================

class VibeVoiceEngineWrapper(BaseTTSEngine):
    """Wrapper for VibeVoice TTS engine"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._engine = None
        self._loaded = False
        self._available = None
    
    @property
    def name(self) -> str:
        return "vibevoice"
    
    @property
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from vibevoice_engine import VibeVoiceEngine
                # Check if model files exist
                model_path = Path(self.config.vibevoice_model_path)
                self._available = model_path.exists() and any(model_path.glob("*.safetensors"))
            except ImportError:
                self._available = False
        return self._available
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._engine is not None
    
    def load(self) -> bool:
        if self._loaded:
            return True
        
        if not self.is_available:
            print("❌ VibeVoice not available (model not downloaded)")
            return False
        
        try:
            print("🔄 Loading VibeVoice...")
            
            from vibevoice_engine import VibeVoiceEngine, VibeVoiceConfig
            
            vv_config = VibeVoiceConfig(
                model_path=self.config.vibevoice_model_path,
                device=self.config.device,
                low_vram_mode=self.config.low_vram_mode,
                cfg_scale=self.config.vibevoice_cfg_scale,
            )
            
            self._engine = VibeVoiceEngine(config=vv_config)
            
            if self._engine.load_model():
                self._loaded = True
                print("✅ VibeVoice loaded")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to load VibeVoice: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload(self):
        if self._engine:
            self._engine.unload_model()
        self._engine = None
        self._loaded = False
        print("✅ VibeVoice unloaded")
    
    def generate(
        self,
        text: str,
        reference_audio: str,
        emotion: str = "neutral",
        **kwargs
    ) -> Optional[GenerationResult]:
        if not self.is_loaded:
            if not self.load():
                return None
        
        start_time = time.time()
        
        try:
            result = self._engine.generate(
                text=text,
                reference_audio=reference_audio,
                emotion=emotion,
                auto_detect_emotion=False,  # We handle emotion externally
                apply_prosody=self.config.apply_prosody,
                enhance_output=self.config.enhance_audio,
                **kwargs
            )
            
            if result is None:
                return None
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                audio=result['audio'],
                sample_rate=result['sample_rate'],
                engine_used="vibevoice",
                duration=result['metadata']['duration'],
                generation_time=generation_time,
                metadata={
                    "text": text,
                    "emotion": emotion,
                    "reference": reference_audio,
                    **result.get('metadata', {})
                }
            )
            
        except Exception as e:
            print(f"❌ VibeVoice generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def supports_language(self, lang: str) -> bool:
        # VibeVoice Hindi-7B supports Hindi and English with natural prosody
        return lang.lower() in ["en", "hi", "hi-in", "hinglish", "english", "hindi"]


# ============================================================================
# UNIFIED TTS INTERFACE
# ============================================================================

class UnifiedTTS:
    """
    Unified TTS interface supporting both Chatterbox and VibeVoice
    
    Features:
    - Automatic engine selection based on text language
    - Seamless fallback between engines
    - Shared voice profiles
    - Unified emotion and prosody processing
    
    Example:
        tts = UnifiedTTS()
        
        # Auto-select engine based on text
        result = tts.generate("नमस्ते, कैसे हो?", profile="pritam")
        result.save("output.wav")
        
        # Force specific engine
        result = tts.generate("Hello world", profile="pritam", engine="chatterbox")
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        
        # Initialize profile manager
        self.profiles = VoiceProfileManager(self.config.profiles_dir)
        
        # Initialize engine wrappers (lazy loading)
        self._engines: Dict[str, BaseTTSEngine] = {
            "chatterbox": ChatterboxEngineWrapper(self.config),
            "vibevoice": VibeVoiceEngineWrapper(self.config),
        }
        
        self._current_engine: Optional[str] = None
        
        # Load optional components
        self._load_emotion_analyzer()
        self._load_prosody_processor()
        
        print("\n🎙️ Unified TTS System Initialized")
        print(f"   Default Engine: {self.config.default_engine.value}")
        print(f"   Profiles: {len(self.profiles.list_profiles())}")
        self._print_engine_status()
    
    def _load_emotion_analyzer(self):
        """Load emotion analyzer if available"""
        try:
            from emotion_analyzer import EmotionAnalyzer
            self._emotion_analyzer = EmotionAnalyzer(mode="rule-based")
            print("   ✅ Emotion Analyzer: Loaded")
        except ImportError:
            self._emotion_analyzer = None
    
    def _load_prosody_processor(self):
        """Load prosody processor if available"""
        try:
            from prosody_processor import ProsodyProcessor
            self._prosody_processor = ProsodyProcessor()
            print("   ✅ Prosody Processor: Loaded")
        except ImportError:
            self._prosody_processor = None
    
    def _print_engine_status(self):
        """Print status of available engines"""
        for name, engine in self._engines.items():
            status = "✅ Available" if engine.is_available else "❌ Not available"
            loaded = " (loaded)" if engine.is_loaded else ""
            print(f"   {name.capitalize()}: {status}{loaded}")
    
    def generate(
        self,
        text: str,
        profile: str,
        emotion: Optional[str] = None,
        engine: Optional[str] = None,
        auto_detect_emotion: bool = True,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[GenerationResult]:
        """
        Generate speech from text
        
        Args:
            text: Text to synthesize
            profile: Voice profile name (e.g., "pritam")
            emotion: Target emotion (neutral, excited, calm, dramatic, conversational, storytelling)
            engine: Force specific engine ("chatterbox", "vibevoice", or None for auto)
            auto_detect_emotion: Auto-detect emotion from text
            output_path: Optional path to save audio
            **kwargs: Additional engine-specific arguments
            
        Returns:
            GenerationResult with audio data and metadata
        """
        print(f"\n{'='*60}")
        print(f"🎙️ UNIFIED TTS GENERATION")
        print(f"{'='*60}")
        
        # Get reference audio from profile
        reference_audio = self.profiles.get_best_reference(profile)
        if not reference_audio:
            print(f"❌ Profile '{profile}' not found or has no audio samples")
            return None
        
        print(f"👤 Profile: {profile}")
        print(f"📝 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"🎵 Reference: {Path(reference_audio).name}")
        
        # Detect or set emotion
        if emotion is None and auto_detect_emotion and self._emotion_analyzer:
            emotion = self._emotion_analyzer.analyze(text)
            print(f"😊 Emotion (auto): {emotion}")
        else:
            emotion = emotion or self.config.default_emotion
            print(f"😊 Emotion: {emotion}")
        
        # Select engine
        selected_engine = self._select_engine(text, engine)
        if selected_engine is None:
            print("❌ No TTS engine available")
            return None
        
        print(f"🔧 Engine: {selected_engine}")
        
        # Switch engine if needed
        if self._current_engine != selected_engine:
            self._switch_engine(selected_engine)
        
        # Get the engine
        tts_engine = self._engines[selected_engine]
        
        # Generate
        result = tts_engine.generate(
            text=text,
            reference_audio=reference_audio,
            emotion=emotion,
            **kwargs
        )
        
        if result:
            print(f"✅ Generated {result.duration:.2f}s audio in {result.generation_time:.2f}s (RTF: {result.rtf:.2f}x)")
            
            # Save if path provided
            if output_path:
                saved_path = result.save(output_path)
                print(f"💾 Saved: {saved_path}")
        
        return result
    
    def _select_engine(self, text: str, requested: Optional[str] = None) -> Optional[str]:
        """Select best engine for the text"""
        # If engine is explicitly requested
        if requested:
            if requested in self._engines and self._engines[requested].is_available:
                return requested
            print(f"⚠️ Requested engine '{requested}' not available")
        
        # Auto-select based on text
        if self.config.default_engine == TTSEngine.AUTO:
            # Check for Hindi content - prefer VibeVoice
            hindi_pattern = r'[\u0900-\u097F]'
            if re.search(hindi_pattern, text):
                if self._engines["vibevoice"].is_available:
                    return "vibevoice"
            
            # For pure English, Chatterbox is well-tested
            if self._engines["chatterbox"].is_available:
                return "chatterbox"
            
            # Fallback to any available
            for name, engine in self._engines.items():
                if engine.is_available:
                    return name
        else:
            # Use default engine if available
            default_name = self.config.default_engine.value
            if default_name in self._engines and self._engines[default_name].is_available:
                return default_name
        
        # Try fallback
        fallback_name = self.config.fallback_engine.value
        if fallback_name in self._engines and self._engines[fallback_name].is_available:
            return fallback_name
        
        return None
    
    def _switch_engine(self, new_engine: str):
        """Switch to a different engine, unloading the current one"""
        # Unload current engine to free memory
        if self._current_engine and self._current_engine != new_engine:
            old_engine = self._engines.get(self._current_engine)
            if old_engine and old_engine.is_loaded:
                print(f"🔄 Unloading {self._current_engine}...")
                old_engine.unload()
        
        self._current_engine = new_engine
    
    def load_engine(self, name: str) -> bool:
        """Pre-load a specific engine"""
        if name not in self._engines:
            print(f"❌ Unknown engine: {name}")
            return False
        
        return self._engines[name].load()
    
    def unload_all(self):
        """Unload all engines to free memory"""
        for engine in self._engines.values():
            if engine.is_loaded:
                engine.unload()
        self._current_engine = None
    
    def list_profiles(self) -> List[str]:
        """List available voice profiles"""
        return self.profiles.list_profiles()
    
    def list_engines(self) -> Dict[str, Dict]:
        """List engines with their status"""
        return {
            name: {
                "available": engine.is_available,
                "loaded": engine.is_loaded,
            }
            for name, engine in self._engines.items()
        }
    
    def get_available_emotions(self) -> List[str]:
        """Get list of supported emotions"""
        return ["neutral", "excited", "calm", "dramatic", "conversational", "storytelling"]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance for easy access
_global_tts: Optional[UnifiedTTS] = None


def get_tts() -> UnifiedTTS:
    """Get or create global TTS instance"""
    global _global_tts
    if _global_tts is None:
        _global_tts = UnifiedTTS()
    return _global_tts


def speak(
    text: str,
    profile: str = "pritam",
    emotion: Optional[str] = None,
    engine: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[GenerationResult]:
    """
    Quick function to generate speech
    
    Args:
        text: Text to speak
        profile: Voice profile name
        emotion: Target emotion
        engine: Engine to use (auto if None)
        save_path: Optional path to save audio
        
    Returns:
        GenerationResult or None if failed
    """
    tts = get_tts()
    return tts.generate(
        text=text,
        profile=profile,
        emotion=emotion,
        engine=engine,
        output_path=save_path,
    )


# ============================================================================
# TESTING
# ============================================================================

def test_compatibility_layer():
    """Test the compatibility layer"""
    print("\n" + "="*60)
    print("🧪 UNIFIED TTS COMPATIBILITY TEST")
    print("="*60)
    
    # Initialize
    tts = UnifiedTTS()
    
    # List profiles
    print(f"\n📁 Available profiles: {tts.list_profiles()}")
    print(f"🔧 Engine status: {tts.list_engines()}")
    
    # Test with first available profile
    profiles = tts.list_profiles()
    if not profiles:
        print("❌ No profiles found")
        return
    
    profile = profiles[0]
    print(f"\n🎤 Testing with profile: {profile}")
    
    # Test English
    print("\n--- English Test ---")
    result = tts.generate(
        text="Hello! This is a test of the unified TTS system.",
        profile=profile,
        emotion="conversational",
        output_path="D:/voice cloning/audio_output/unified_test_english.wav"
    )
    if result:
        print(f"✅ English test passed! Engine: {result.engine_used}")
    
    # Test Hindi
    print("\n--- Hindi Test ---")
    result = tts.generate(
        text="नमस्ते! यह एक परीक्षण है।",
        profile=profile,
        emotion="conversational",
        output_path="D:/voice cloning/audio_output/unified_test_hindi.wav"
    )
    if result:
        print(f"✅ Hindi test passed! Engine: {result.engine_used}")
    
    # Cleanup
    tts.unload_all()
    print("\n✅ Tests complete!")


if __name__ == "__main__":
    test_compatibility_layer()
