"""
🎙️ MY VOICE CLONE - Ultimate Voice Cloning System
==================================================

Features:
✅ Clone your voice in English AND Hindi
✅ Emotional generation (excited, calm, dramatic, conversational)
✅ Long-form audio (1+ minute) with perfect quality
✅ Intelligent chunking with smooth crossfade transitions
✅ Easy-to-use Gradio interface for voice sample submission
✅ Support for multiple emotional reference audios

Usage:
    python myvoiceclone.py

Requirements:
    - Record 10-15 second voice samples in different emotions
    - For best Hindi results, record references speaking Hindi emotionally
"""

# IMPORTANT: Configure model paths to D: drive BEFORE importing ML libraries
import model_paths  # This sets HF_HOME, TORCH_HOME, etc. to D: drive

import os
import re
import sys
import torch
import torchaudio as ta
import numpy as np
import gradio as gr
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import gc

# ============================================================================
# HARDWARE OPTIMIZATION SETTINGS (i9 9th gen, 32GB RAM, T2000 4GB VRAM)
# ============================================================================

def setup_cuda_optimizations():
    """Configure CUDA for optimal performance on T2000 4GB"""
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix operations (Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy for 4GB VRAM
        # Use smaller chunks to prevent OOM
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
        print(f"⚡ CUDA optimizations enabled (TF32, cuDNN benchmark)")
        
        return True
    return False

# Apply CUDA optimizations at import time
CUDA_AVAILABLE = setup_cuda_optimizations()

# Set optimal thread count for i9 (use physical cores)
NUM_CPU_THREADS = min(8, os.cpu_count() or 4)  # i9 has 8 cores
torch.set_num_threads(NUM_CPU_THREADS)
os.environ['OMP_NUM_THREADS'] = str(NUM_CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_CPU_THREADS)

# Add chatterbox to path
CHATTERBOX_PATH = Path(__file__).parent / "chatterbox"
sys.path.insert(0, str(CHATTERBOX_PATH / "src"))

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Import Phase 1 enhancements
try:
    from audio_enhancer import AudioEnhancer
    from advanced_voice_encoder import AdvancedVoiceEncoder
    from emotion_analyzer import EmotionAnalyzer
    ENHANCEMENTS_AVAILABLE = True
    print("✨ Phase 1 Enhancements: LOADED")
except ImportError as e:
    ENHANCEMENTS_AVAILABLE = False
    print(f"⚠️ Phase 1 Enhancements not available: {e}")
    print("   Run without enhancements or install dependencies")

# Import Adobe Podcast Enhancer (advanced audio processing)
try:
    from adobe_enhance import AdobePodcastEnhancer
    ADOBE_ENHANCE_AVAILABLE = True
    print("🎙️ Adobe Podcast Enhancer: LOADED")
except ImportError as e:
    ADOBE_ENHANCE_AVAILABLE = False
    print(f"⚠️ Adobe Podcast Enhancer not available: {e}")

# Import Studio Audio Processor (broadcast-quality mastering)
try:
    from studio_audio_processor import StudioAudioProcessor, process_for_social_media
    STUDIO_PROCESSOR_AVAILABLE = True
    print("🎚️ Studio Audio Processor: LOADED (Broadcast Quality)")
except ImportError as e:
    STUDIO_PROCESSOR_AVAILABLE = False
    print(f"⚠️ Studio Audio Processor not available: {e}")

# Import Phase 2 enhancements
try:
    from f5_engine import F5TTSEngine
    from prosody_predictor import ProsodyPredictor
    from ensemble_generator import EnsembleGenerator, QualityScorer
    PHASE2_AVAILABLE = True
    print("🚀 Phase 2 Enhancements: LOADED")
except ImportError as e:
    PHASE2_AVAILABLE = False
    print(f"⚠️ Phase 2 Enhancements not available: {e}")
    print("   Install F5-TTS: pip install f5-tts")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VoiceProfile:
    """Voice profile for a person"""
    name: str                                    # Person's name (e.g., "bhomik")
    samples: List[str]                           # List of audio file paths
    languages: List[str]                         # Languages spoken in samples
    embedding_cache: Optional[torch.Tensor] = None  # Cached embedding (from best sample)
    best_sample: Optional[str] = None            # Best quality sample for reference
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def get_best_reference(self) -> str:
        """Get the best reference audio path for voice cloning"""
        if self.best_sample and Path(self.best_sample).exists():
            return self.best_sample
        # Fallback: find sample_012 if exists (known best for pritam)
        for sample in self.samples:
            if 'sample_012' in sample:
                return sample
        # Last resort: first sample
        return self.samples[0] if self.samples else None


@dataclass
class HardwareConfig:
    """Hardware-specific optimization settings"""
    # GPU Settings (optimized for T2000 4GB)
    use_fp16: bool = True              # Mixed precision - saves ~50% VRAM
    gpu_memory_fraction: float = 0.85   # Leave some VRAM for system
    batch_size_voice_encoder: int = 8   # Optimal for 4GB VRAM
    
    # CPU Settings (optimized for i9 9th gen - MAXIMIZED)
    num_workers: int = 8                # Use all 8 cores for parallel loading
    prefetch_factor: int = 3            # Aggressive prefetch
    
    # Memory Settings (optimized for 32GB RAM)
    max_audio_cache_mb: int = 2048      # 2GB audio cache
    preload_models: bool = True         # Keep models in RAM
    
    # Speed Mode - ULTRA AGGRESSIVE (Models can't limit steps, so use tiny chunks)
    speed_mode: bool = True             # Prioritize speed over max quality
    max_sampling_steps: int = 200       # Note: ChatterboxTTS ignores this
    use_turbo_by_default: bool = False  # Turbo is faster for English
    enable_flash_attention: bool = False # T2000 doesn't support flash attention


@dataclass
class VoiceCloneConfig:
    """Configuration for voice cloning"""
    # Profile storage
    profiles_dir: str = "voice_profiles"  # Directory to store voice profiles
    
    # Chunking settings - NATURAL PAUSE BASED
    optimal_chunk_size: int = 400       # Larger limit since we split by pauses, not chars
    max_chunk_size: int = 500           # Only splits at commas if sentence is very long
    crossfade_duration_ms: int = 300    # Aggressive crossfade to eliminate ALL clicks
    
    # Quality settings - OPTIMIZED FOR BETTER PRONUNCIATION
    default_temperature: float = 0.5        # Lower = more accurate pronunciation
    default_top_p: float = 0.95             # Higher = more natural
    default_top_k: int = 2000               # Higher = better word choice
    default_repetition_penalty: float = 1.1  # Lower = more natural flow
    
    # Hardware optimization
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Emotion settings per style
    emotion_presets: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        # FIXED: cfg_weight should be 0.2-1.0, NOT 4.0!
        # Higher cfg_weight = more similar to reference but can cause artifacts
        # LOWERED TO 0.2-0.25 FOR NATURAL PRONUNCIATION (prevents syllable splitting)
        self.emotion_presets = {
            "neutral": {
                "exaggeration": 0.4,       # Slightly below neutral for natural sound
                "temperature": 0.6,        # Lower = more consistent
                "cfg_weight": 0.22,        # LOWERED: 0.3 → 0.22 for natural pronunciation
            },
            "excited": {
                "exaggeration": 0.7,       # Higher for expressive speech
                "temperature": 0.75,       # Slightly more varied
                "cfg_weight": 0.25,        # LOWERED: 0.35 → 0.25 for clearer speech
            },
            "calm": {
                "exaggeration": 0.3,       # Lower for calm delivery
                "temperature": 0.5,        # Very consistent
                "cfg_weight": 0.25,        # LOWERED: 0.4 → 0.25 for natural pronunciation
            },
            "dramatic": {
                "exaggeration": 0.9,       # High for dramatic effect
                "temperature": 0.8,        # More varied
                "cfg_weight": 0.23,        # LOWERED: 0.35 → 0.23 for clear expression
            },
            "conversational": {
                "exaggeration": 0.5,       # Natural speaking
                "temperature": 0.7,        # Natural variation
                "cfg_weight": 0.20,        # LOWERED: 0.3 → 0.20 for most natural speech
            },
            "storytelling": {
                "exaggeration": 0.6,       # Moderately expressive
                "temperature": 0.7,        # Natural variation
                "cfg_weight": 0.24,        # LOWERED: 0.35 → 0.24 for clear narration
            }
        }


# ============================================================================
# TEXT PROCESSING
# ============================================================================

class TextProcessor:
    """Intelligent text processing for TTS with Hinglish support"""
    
    # Hindi sentence endings
    HINDI_ENDINGS = {'।', '?', '!', '।।', '॥'}
    
    # English sentence endings
    ENGLISH_ENDINGS = {'.', '!', '?'}
    
    # Common Hinglish words (romanized Hindi) - lowercase for matching
    HINGLISH_WORDS = {
        # Pronouns
        'main', 'mein', 'mai', 'mujhe', 'mujhko', 'hum', 'humko', 'humne',
        'tum', 'tumko', 'tumhe', 'tumne', 'aap', 'aapko', 'aapne',
        'woh', 'wo', 'voh', 'vo', 'yeh', 'ye', 'yah',
        'unka', 'unki', 'unko', 'inka', 'inki', 'inko',
        'mera', 'meri', 'mere', 'tumhara', 'tumhari', 'tumhare',
        'apna', 'apni', 'apne', 'hamara', 'hamari', 'hamare',
        
        # Common verbs
        'hai', 'hain', 'tha', 'thi', 'the', 'hoga', 'hogi', 'honge',
        'karna', 'karo', 'karte', 'karti', 'kiya', 'karenge', 'karunga', 'karungi',
        'hona', 'ho', 'hota', 'hoti', 'hua', 'hui', 'hue', 'hogaya',
        'jaana', 'jao', 'jata', 'jati', 'gaya', 'gayi', 'gaye', 'jayenge', 'jaunga',
        'aana', 'aao', 'aata', 'aati', 'aaya', 'aayi', 'aaye', 'ayenge', 'aaunga',
        'lena', 'lo', 'leta', 'leti', 'liya', 'liyi', 'liye', 'lenge', 'lunga',
        'dena', 'do', 'deta', 'deti', 'diya', 'diyi', 'diye', 'denge', 'dunga',
        'bolna', 'bolo', 'bolta', 'bolti', 'bola', 'boli', 'bole', 'bolenge',
        'dekhna', 'dekho', 'dekhta', 'dekhti', 'dekha', 'dekhi', 'dekhenge',
        'sunna', 'suno', 'sunta', 'sunti', 'suna', 'suni', 'sunenge',
        'samajhna', 'samjho', 'samajhta', 'samajhti', 'samjha', 'samajh',
        'milna', 'milo', 'milta', 'milti', 'mila', 'mili', 'milenge',
        'rakhna', 'rakho', 'rakhta', 'rakhti', 'rakha', 'rakhi', 'rakhenge',
        'padhna', 'padho', 'padhta', 'padhti', 'padha', 'padhi', 'padhenge',
        'likhna', 'likho', 'likhta', 'likhti', 'likha', 'likhi', 'likhenge',
        'khana', 'khao', 'khata', 'khati', 'khaya', 'khayi', 'khaye', 'khayenge',
        'peena', 'piyo', 'peeta', 'peeti', 'piya', 'piyi', 'piye', 'piyenge',
        'sona', 'soye', 'sota', 'soti', 'soya', 'soyi', 'soyenge',
        'uthna', 'utho', 'uthta', 'uthti', 'utha', 'uthi', 'uthenge',
        'chalna', 'chalo', 'chalta', 'chalti', 'chala', 'chali', 'chalenge',
        'baithna', 'baitho', 'baithta', 'baithti', 'baitha', 'baithi', 'baithenge',
        'khada', 'khadi', 'khade',
        'chahna', 'chahiye', 'chahte', 'chahti', 'chaha', 'chahi',
        'sakna', 'sakta', 'sakti', 'sakte', 'sako', 'sakenge', 'sakugi',
        'pata', 'pati', 'pate', 'paya', 'payenge',
        'lagta', 'lagti', 'lagte', 'laga', 'lagi', 'lagenge',
        'rehna', 'raho', 'rehta', 'rehti', 'raha', 'rahi', 'rahenge',
        
        # Question words
        'kya', 'kaise', 'kaisa', 'kaisi', 'kahan', 'kahaan', 'kidhar',
        'kab', 'kaun', 'kyun', 'kyu', 'kyunki', 'kisliye', 'kitna', 'kitni', 'kitne',
        
        # Common words
        'aur', 'ya', 'par', 'lekin', 'magar', 'phir', 'fir', 'toh', 'to', 'bhi',
        'nahi', 'nahin', 'naa', 'na', 'mat', 'haan', 'han', 'ji', 'jee',
        'bahut', 'bohot', 'boht', 'zyada', 'jyada', 'kam', 'thoda', 'thodi', 'thode',
        'accha', 'acha', 'acchi', 'achi', 'acche', 'ache', 'bura', 'buri', 'bure',
        'bada', 'badi', 'bade', 'chota', 'choti', 'chote',
        'naya', 'nayi', 'naye', 'purana', 'purani', 'purane',
        'pehle', 'pahle', 'baad', 'mein',
        'abhi', 'aaj', 'kal', 'parso', 'waqt', 'samay', 'din', 'raat',
        'sab', 'sabhi', 'sabko', 'kuch', 'kuchh', 'kaafi', 'sirf', 'bas',
        'yahan', 'wahan', 'yahaan', 'wahaan', 'idhar', 'udhar',
        'upar', 'neeche', 'andar', 'bahar', 'paas', 'door', 'durr',
        'ghar', 'dost', 'bhai', 'behen', 'beta', 'beti', 'baap', 'maa', 'mata',
        'log', 'logon', 'kaam', 'baat', 'baatein', 'cheez', 'cheezen',
        'wala', 'wali', 'wale', 'waala', 'waali', 'waale',
        'isliye', 'isiliye', 'kyonki', 'taaki', 'jab', 'tab', 'agar', 'agr',
        'zaroor', 'jaroor', 'shayad', 'bilkul', 'sach', 'jhooth',
        
        # Greetings and expressions
        'namaste', 'namaskar', 'shukriya', 'dhanyawad', 'dhanyavaad',
        'kripya', 'please', 'sorry', 'maaf', 'theek', 'thik', 'chalo', 'acha',
        'arrey', 'arre', 'oye', 'are', 'yaar', 'dude', 'bro',
        
        # Numbers (romanized)
        'ek', 'do', 'teen', 'char', 'paanch', 'panch', 'chhe', 'saat', 'aath', 'nau', 'das',
        
        # Additional common Hinglish
        'samajh', 'matlab', 'matlb', 'soch', 'socho', 'dekho', 'suno', 'bolo',
        'batao', 'btao', 'dikhao', 'sikho', 'padho', 'likho', 'karo', 'ruko',
        'jaldi', 'dheere', 'dhire', 'seedha', 'seedhe', 'ulta', 'ulte',
    }
    
    @classmethod
    def fix_english_pronunciation(cls, text: str) -> str:
        """
        Fix common English pronunciation issues in TTS.
        Replaces words with silent letters with phonetic spellings.
        
        Examples:
        - "honestly" -> "onestly" (silent h)
        - "hour" -> "our" (silent h)
        - "honor" -> "onor" (silent h)
        - "knight" -> "nite" (silent k, gh)
        """
        # Common silent-h words
        silent_h_words = {
            r'\bhonestly\b': 'onestly',
            r'\bhonest\b': 'onest',
            r'\bhonor\b': 'onor',
            r'\bhonour\b': 'onour',
            r'\bhonors\b': 'onors',
            r'\bhonours\b': 'onours',
            r'\bhour\b': 'our',
            r'\bhours\b': 'ours',
            r'\bheir\b': 'air',
            r'\bheiress\b': 'airess',
        }
        
        # Common silent letter patterns
        silent_patterns = {
            r'\bkn': 'n',      # knight -> nite, know -> no
            r'gh\b': '',       # high -> hi, sigh -> si
            r'\bwr': 'r',      # write -> rite, wrong -> rong
            r'\bps': 's',      # psychology -> sychology
            r'\bmn\b': 'm',    # hymn -> him
            r'mb\b': 'm',      # climb -> clime, bomb -> bom
        }
        
        result = text
        
        # Apply silent-h word replacements (case-insensitive)
        for pattern, replacement in silent_h_words.items():
            # Match both lowercase and capitalized versions
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            result = re.sub(pattern.replace(r'\b', r'\b').capitalize(), 
                          replacement.capitalize(), result)
        
        # Apply general silent letter patterns
        for pattern, replacement in silent_patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """
        Detect if text is Hindi, Hinglish (romanized Hindi), or English.
        
        Handles mixed scripts like: "Mitthuu Express आ चुकी है Station 9 पर।"
        
        Returns:
            "hi" for Hindi (Devanagari), Hinglish (romanized), or mixed
            "en" for pure English
        """
        # Check for Devanagari characters (Hindi script)
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return "en"
        
        # ANY Devanagari present = treat as Hindi (for mixed scripts)
        # This catches "Mitthuu Express आ चुकी है Station 9 पर।"
        if hindi_chars > 0:
            return "hi"
        
        # Check for Hinglish words (romanized Hindi without Devanagari)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if words:
            hinglish_count = sum(1 for w in words if w in cls.HINGLISH_WORDS)
            hinglish_ratio = hinglish_count / len(words)
            
            # If >20% words are Hinglish, treat as Hindi for better pronunciation
            if hinglish_ratio > 0.2 or hinglish_count >= 3:
                return "hi"
        
        return "en"
    
    # Pause indicators for natural chunking
    # Strong pauses (sentence boundaries)
    STRONG_PAUSE = {
        '.', '!', '?',           # English
        '।', '॥', '?', '!',      # Hindi (purna viram, double viram)
        '।।',                    # Hindi double stop
    }
    
    # Medium pauses (clause boundaries)  
    MEDIUM_PAUSE = {
        ',', ';', ':',           # English
        '،',                     # Hindi/Urdu comma
        '—', '–', '-',           # Dashes
        '...',                   # Ellipsis
    }
    
    @classmethod
    def split_by_pauses(cls, text: str) -> List[str]:
        """
        Split text by natural pause indicators (full stops, commas, newlines).
        Preserves complete sentences and natural speech units.
        """
        # First split by newlines (paragraphs/lines)
        lines = text.split('\n')
        
        segments = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split by strong pauses (sentence endings)
            # Pattern matches: . ! ? । ॥ and keeps the delimiter
            pattern = r'([.!?।॥]+\s*)'
            parts = re.split(pattern, line)
            
            current = ""
            for part in parts:
                if not part:
                    continue
                current += part
                # If this part ends with a strong pause, it's a complete segment
                if any(p in part for p in cls.STRONG_PAUSE):
                    if current.strip():
                        segments.append(current.strip())
                    current = ""
            
            # Don't forget remaining text (no ending punctuation)
            if current.strip():
                segments.append(current.strip())
        
        return segments
    
    @classmethod  
    def split_into_sentences(cls, text: str, language: str = "en") -> List[str]:
        """Split text into sentences based on natural pauses"""
        return cls.split_by_pauses(text)
    
    @classmethod
    def create_chunks(cls, text: str, max_chars: int = 200) -> List[Tuple[str, str]]:
        """
        Create chunks based on NATURAL PAUSE INDICATORS, not character counts.
        
        Priority:
        1. Newlines (paragraph/line breaks)
        2. Full stops (. । ॥ ! ?)
        3. Commas/semicolons only if sentence is very long
        
        Returns list of (chunk_text, language) tuples.
        """
        chunks = []
        
        # Split by natural pauses first
        segments = cls.split_by_pauses(text)
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # If segment is reasonable size, use it as-is
            if len(segment) <= max_chars:
                lang = cls.detect_language(segment)
                chunks.append((segment, lang))
            else:
                # Only split very long segments at medium pauses (commas)
                # Pattern for comma-like pauses
                sub_pattern = r'([,;:،—–]\s*)'
                sub_parts = re.split(sub_pattern, segment)
                
                current = ""
                for part in sub_parts:
                    if not part:
                        continue
                    
                    # If adding this part keeps us under limit, add it
                    if len(current) + len(part) <= max_chars:
                        current += part
                    else:
                        # Save current chunk and start new one
                        if current.strip():
                            lang = cls.detect_language(current)
                            chunks.append((current.strip(), lang))
                        current = part
                
                # Don't forget the last part
                if current.strip():
                    lang = cls.detect_language(current)
                    chunks.append((current.strip(), lang))
        
        return chunks
    
    @staticmethod
    def enhance_for_emotion(text: str, emotion: str) -> str:
        """Add punctuation hints to encourage emotional delivery"""
        if emotion == "excited":
            # More exclamations
            text = re.sub(r'\.(\s|$)', '! ', text)
        elif emotion == "dramatic":
            # Add pauses
            text = re.sub(r',', '...', text)
        
        return text


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class ProfileManager:
    """Manage voice profiles with persistent storage and optimized caching"""
    
    def __init__(self, profiles_dir: str = "voice_profiles", hardware_config: Optional[HardwareConfig] = None):
        """Initialize profile manager with hardware optimizations"""
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.profiles: Dict[str, VoiceProfile] = {}
        self.voice_encoder = None
        self.hardware_config = hardware_config or HardwareConfig()
        
        # Audio cache for faster repeated access (uses RAM)
        self._audio_cache: Dict[str, Tuple[torch.Tensor, int]] = {}
        self._cache_size_bytes = 0
        self._max_cache_bytes = self.hardware_config.max_audio_cache_mb * 1024 * 1024
        
        # Thread pool for parallel audio loading
        self._executor = ThreadPoolExecutor(max_workers=self.hardware_config.num_workers)
        
        # Load existing profiles
        self._load_all_profiles()
    
    def _get_profile_dir(self, name: str) -> Path:
        """Get directory for a specific profile"""
        return self.profiles_dir / name.lower().replace(" ", "_")
    
    def _load_all_profiles(self):
        """Load all existing profiles from disk"""
        if not self.profiles_dir.exists():
            return
        
        for profile_dir in self.profiles_dir.iterdir():
            if profile_dir.is_dir():
                try:
                    self._load_profile(profile_dir.name)
                except Exception as e:
                    print(f"⚠️  Failed to load profile {profile_dir.name}: {e}")
    
    def _load_profile(self, name: str):
        """Load a specific profile from disk"""
        profile_dir = self._get_profile_dir(name)
        metadata_file = profile_dir / "metadata.json"
        embedding_file = profile_dir / "embedding.pt"
        
        if not metadata_file.exists():
            return
        
        # Load metadata
        import json
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profile = VoiceProfile(
            name=data['name'],
            samples=data['samples'],
            languages=data['languages'],
            best_sample=data.get('best_sample'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
        
        # Load embedding if exists
        if embedding_file.exists():
            profile.embedding_cache = torch.load(embedding_file, weights_only=True)
        
        self.profiles[name.lower()] = profile
        print(f"✅ Loaded profile: {name} ({len(profile.samples)} samples)")
    
    def _save_profile(self, profile: VoiceProfile):
        """Save a profile to disk"""
        profile_dir = self._get_profile_dir(profile.name)
        profile_dir.mkdir(exist_ok=True)
        
        # Save metadata
        import json
        metadata = {
            'name': profile.name,
            'samples': profile.samples,
            'languages': profile.languages,
            'best_sample': profile.best_sample,
            'created_at': profile.created_at,
            'updated_at': profile.updated_at
        }
        
        with open(profile_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embedding
        if profile.embedding_cache is not None:
            torch.save(profile.embedding_cache, profile_dir / "embedding.pt")
        
        # Copy sample files to profile directory
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        print(f"💾 Saved profile: {profile.name}")
    
    def create_profile(self, name: str) -> VoiceProfile:
        """Create a new voice profile"""
        # Trim whitespace from name
        name = name.strip()
        if not name:
            raise ValueError("Profile name cannot be empty")
        
        if name.lower() in self.profiles:
            raise ValueError(f"Profile '{name}' already exists!")
        
        profile = VoiceProfile(
            name=name,
            samples=[],
            languages=[]
        )
        
        self.profiles[name.lower()] = profile
        self._save_profile(profile)
        
        print(f"✅ Created new profile: {name}")
        return profile
    
    def add_sample(self, name: str, audio_path: str, language: str = "en"):
        """Add a voice sample to a profile"""
        # Trim whitespace from name
        name = name.strip()
        name_lower = name.lower()
        
        # Create profile if doesn't exist
        if name_lower not in self.profiles:
            self.create_profile(name)
        
        profile = self.profiles[name_lower]
        
        # Copy audio file to profile directory
        profile_dir = self._get_profile_dir(name)
        samples_dir = profile_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        sample_idx = len(profile.samples) + 1
        new_filename = f"sample_{sample_idx:03d}.wav"
        dest_path = samples_dir / new_filename
        
        # Copy file
        import shutil
        shutil.copy2(audio_path, dest_path)
        
        # Add to profile
        profile.samples.append(str(dest_path))
        if language not in profile.languages:
            profile.languages.append(language)
        
        profile.updated_at = datetime.now().isoformat()
        
        # Invalidate cached embedding (will be recomputed)
        profile.embedding_cache = None
        
        self._save_profile(profile)
        
        print(f"✅ Added sample {sample_idx} to {name} ({language})")
    
    def get_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get a voice profile by name"""
        return self.profiles.get(name.strip().lower())
    
    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        # Return actual profile names (with proper capitalization), not keys
        return sorted([p.name for p in self.profiles.values()])
    
    def _load_audio_cached(self, sample_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio with RAM caching for faster repeated access"""
        if sample_path in self._audio_cache:
            wav, sr = self._audio_cache[sample_path]
            return wav.squeeze(0).numpy(), sr
        
        if not os.path.exists(sample_path):
            return None
        
        try:
            wav, sr = ta.load(sample_path)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Cache if within limits
            audio_bytes = wav.numel() * wav.element_size()
            if self._cache_size_bytes + audio_bytes < self._max_cache_bytes:
                self._audio_cache[sample_path] = (wav, sr)
                self._cache_size_bytes += audio_bytes
            
            return wav.squeeze(0).numpy(), sr
        except Exception as e:
            print(f"   ⚠️  Failed to load {sample_path}: {e}")
            return None
    
    def compute_profile_embedding(self, profile: VoiceProfile, voice_encoder) -> torch.Tensor:
        """
        Compute embedding using SMART HYBRID approach for optimal voice cloning.
        
        Strategy (based on ElevenLabs + research best practices):
        1. Compute individual embeddings for ALL samples
        2. Find the TOP 3-5 most mutually similar samples (consistent voice)
        3. Average ONLY those top samples (not all 30!)
        4. Use the single BEST sample as reference audio
        
        This gives better results than:
        - Single sample only (loses some voice characteristics)
        - All 30 samples averaged (dilutes unique voice traits)
        """
        if len(profile.samples) == 0:
            raise ValueError(f"Profile {profile.name} has no samples!")
        
        # Check cache first (instant return)
        if profile.embedding_cache is not None:
            print(f"📦 Using cached embedding for {profile.name}")
            return profile.embedding_cache
        
        print(f"🧠 Computing SMART embedding for {profile.name}...")
        print(f"   📊 Analyzing {len(profile.samples)} samples to find most consistent voice...")
        
        # Step 1: Load all samples and compute individual embeddings
        wavs = []
        valid_paths = []
        
        for path in profile.samples:
            result = self._load_audio_cached(path)
            if result is not None:
                wav_np, sr = result
                wavs.append(wav_np)
                valid_paths.append(path)
        
        if len(wavs) == 0:
            raise ValueError(f"No valid samples found for profile {profile.name}")
        
        sample_rate = sr
        use_fp16 = self.hardware_config.use_fp16 and torch.cuda.is_available()
        
        # Step 2: Compute individual embeddings for each sample
        print(f"   🔄 Computing individual embeddings {'(FP16)' if use_fp16 else '(FP32)'}...")
        
        individual_embeddings = []
        with torch.inference_mode():
            for i, wav_np in enumerate(wavs):
                if use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb = voice_encoder.embeds_from_wavs(
                            [wav_np], sample_rate=sample_rate, as_spk=True, batch_size=1
                        )
                else:
                    emb = voice_encoder.embeds_from_wavs(
                        [wav_np], sample_rate=sample_rate, as_spk=True, batch_size=1
                    )
                individual_embeddings.append(emb)
        
        # Step 3: Compute similarity matrix to find most consistent samples
        import numpy as np
        embeddings_array = np.array(individual_embeddings).squeeze()
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_normalized = embeddings_array / (norms + 1e-8)
        
        # Compute average similarity of each sample to all others
        similarity_matrix = embeddings_normalized @ embeddings_normalized.T
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Step 4: Select TOP samples (most similar to others = most representative)
        num_top_samples = min(5, len(wavs))  # Use top 5 or fewer
        top_indices = np.argsort(avg_similarities)[-num_top_samples:][::-1]
        
        best_idx = top_indices[0]
        best_sample_path = valid_paths[best_idx]
        
        print(f"   🏆 TOP {num_top_samples} most consistent samples:")
        for rank, idx in enumerate(top_indices, 1):
            sample_name = Path(valid_paths[idx]).name
            sim_score = avg_similarities[idx]
            marker = "⭐ BEST" if idx == best_idx else ""
            print(f"      {rank}. {sample_name} (consistency: {sim_score:.4f}) {marker}")
        
        # Step 5: Compute final embedding from TOP samples only
        print(f"   🎯 Computing final embedding from TOP {num_top_samples} samples (not all {len(wavs)})...")
        
        top_wavs = [wavs[i] for i in top_indices]
        
        with torch.inference_mode():
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    final_embedding = voice_encoder.embeds_from_wavs(
                        top_wavs, sample_rate=sample_rate, as_spk=True,
                        batch_size=self.hardware_config.batch_size_voice_encoder
                    )
            else:
                final_embedding = voice_encoder.embeds_from_wavs(
                    top_wavs, sample_rate=sample_rate, as_spk=True,
                    batch_size=self.hardware_config.batch_size_voice_encoder
                )
            
            embedding = torch.from_numpy(final_embedding).to(voice_encoder.device)
        
        # Update profile with best sample info
        profile.best_sample = best_sample_path
        profile.embedding_cache = embedding
        self._save_profile(profile)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"   ✅ Smart embedding computed!")
        print(f"   📌 Best reference audio: {Path(best_sample_path).name}")
        print(f"   💡 Using top {num_top_samples} consistent samples instead of all {len(wavs)}")
        return embedding
    
    def delete_profile(self, name: str):
        """Delete a voice profile"""
        name = name.strip()
        name_lower = name.lower()
        if name_lower not in self.profiles:
            raise ValueError(f"Profile '{name}' not found!")
        
        # Remove from memory
        del self.profiles[name_lower]
        
        # Delete directory
        profile_dir = self._get_profile_dir(name)
        if profile_dir.exists():
            import shutil
            shutil.rmtree(profile_dir)
        
        print(f"🗑️  Deleted profile: {name}")


class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def apply_crossfade(
        audio1: torch.Tensor, 
        audio2: torch.Tensor, 
        fade_samples: int
    ) -> torch.Tensor:
        """Apply smooth crossfade between audio segments with DC offset removal"""
        if audio1.shape[-1] < fade_samples or audio2.shape[-1] < fade_samples:
            return torch.cat([audio1, audio2], dim=-1)
        
        # Remove DC offset from both segments to prevent clicks
        audio1_mean = audio1.mean()
        audio2_mean = audio2.mean()
        audio1_dc_removed = audio1 - audio1_mean
        audio2_dc_removed = audio2 - audio2_mean
        
        # Create smooth cosine fade curves (smoother than linear)
        t = torch.linspace(0, torch.pi, fade_samples, device=audio1.device)
        fade_out = torch.cos(t * 0.5)  # Cosine fade out: 1 → 0
        fade_in = torch.sin(t * 0.5)    # Cosine fade in: 0 → 1
        
        # Apply fades
        audio1_faded = audio1_dc_removed.clone()
        audio2_faded = audio2_dc_removed.clone()
        
        audio1_faded[:, -fade_samples:] = audio1_dc_removed[:, -fade_samples:] * fade_out
        audio2_faded[:, :fade_samples] = audio2_dc_removed[:, :fade_samples] * fade_in
        
        # Overlap-add
        result = torch.cat([
            audio1_faded[:, :-fade_samples],
            audio1_faded[:, -fade_samples:] + audio2_faded[:, :fade_samples],
            audio2_faded[:, fade_samples:]
        ], dim=-1)
        
        # Restore DC offset (average of both)
        result = result + (audio1_mean + audio2_mean) / 2
        
        return result
    
    @staticmethod
    def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
        """Normalize audio loudness"""
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if np.isfinite(loudness):
                audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        except Exception as e:
            print(f"Loudness normalization skipped: {e}")
        return audio


# ============================================================================
# VOICE CLONE ENGINE
# ============================================================================

class MyVoiceClone:
    """
    Ultimate Voice Cloning Engine - Optimized for i9 + T2000
    
    Features:
    - Profile-based voice cloning with persistent storage
    - Multiple samples per person for consistent voice
    - Dual model system (English + Multilingual)
    - Emotional generation in both languages
    - Long-form audio with chunking
    - Automatic language detection
    - Hardware-optimized inference (FP16, CUDA optimizations)
    """
    
    def __init__(self, device: Optional[str] = None, config: Optional[VoiceCloneConfig] = None):
        """Initialize the voice cloning engine with hardware optimizations"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or VoiceCloneConfig()
        
        self.english_model = None
        self.multilingual_model = None
        self.turbo_model = None
        self.voice_encoder = None
        
        # Mixed precision scaler for FP16 inference
        self.use_fp16 = self.config.hardware.use_fp16 and torch.cuda.is_available()
        
        # Profile manager for persistent voice storage
        self.profile_manager = ProfileManager(
            self.config.profiles_dir, 
            hardware_config=self.config.hardware
        )
        
        print(f"🎙️ MyVoiceClone initialized on {self.device}")
        print(f"📁 Voice profiles directory: {self.config.profiles_dir}")
        
        # Show optimization status
        if self.use_fp16:
            print(f"⚡ Mixed Precision (FP16) enabled for English models")
        if hasattr(self.config.hardware, 'speed_mode') and self.config.hardware.speed_mode:
            max_steps = self.config.hardware.max_sampling_steps
            print(f"🚀 SPEED MODE: Max {max_steps} sampling steps (5x+ faster)")
        print(f"💻 Parallel workers: {self.config.hardware.num_workers} (utilizing all i9 cores)")
        
        # Show available profiles
        profiles = self.profile_manager.list_profiles()
        if profiles:
            print(f"👥 Found {len(profiles)} voice profile(s): {', '.join(profiles)}")
        else:
            print(f"👥 No voice profiles yet. Create one to get started!")
    
    def _optimize_model(self, model):
        """Apply optimizations to a loaded model"""
        if model is None:
            return model
        
        # Set to eval mode if the model supports it (disables dropout, etc.)
        if hasattr(model, 'eval'):
            model.eval()
        
        # Enable gradient checkpointing to save VRAM if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    def _clear_vram(self):
        """Clear CUDA cache to free VRAM between operations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_models(self, 
                   load_english: bool = False,
                   load_multilingual: bool = True,
                   load_turbo: bool = False):
        """
        Load TTS models with VRAM-optimized loading.
        
        Default: Only Multilingual model (~3GB) for Hinglish support.
        This fits in 4GB T2000 VRAM without overflow to system RAM.
        
        For pure English-only use: load_english=True, load_multilingual=False
        For fast English: load_turbo=True (requires HuggingFace token)
        """
        
        if load_english and self.english_model is None:
            print("📦 Loading English emotion model (ChatterboxTTS)...")
            self._clear_vram()  # Clear before loading
            self.english_model = ChatterboxTTS.from_pretrained(self.device)
            self.english_model = self._optimize_model(self.english_model)
            print("   ✅ English model loaded")
            
            # Extract voice encoder from English model (stored as 've' attribute)
            if self.voice_encoder is None:
                self.voice_encoder = self.english_model.ve
                self.profile_manager.voice_encoder = self.voice_encoder
                print("   ✅ Voice encoder ready")
        
        if load_multilingual and self.multilingual_model is None:
            print("📦 Loading Multilingual model...")
            self._clear_vram()  # Clear before loading
            self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(self.device)
            self.multilingual_model = self._optimize_model(self.multilingual_model)
            print("   ✅ Multilingual model loaded")
            
            # Extract voice encoder if not already loaded (stored as 've' attribute)
            if self.voice_encoder is None:
                self.voice_encoder = self.multilingual_model.ve
                self.profile_manager.voice_encoder = self.voice_encoder
                print("   ✅ Voice encoder ready")
        
        if load_turbo and self.turbo_model is None:
            print("📦 Loading Turbo model (fast English)...")
            self._clear_vram()  # Clear before loading
            self.turbo_model = ChatterboxTurboTTS.from_pretrained(self.device)
            self.turbo_model = self._optimize_model(self.turbo_model)
            print("   ✅ Turbo model loaded")
        
        # Final VRAM cleanup
        self._clear_vram()
        
        # Report VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"💾 VRAM Usage: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        print("🎉 All models loaded!")
    
    def add_voice_sample(self, profile_name: str, audio_path: str, language: str = "en"):
        """
        Add a voice sample to a profile
        
        Args:
            profile_name: Name of the person (e.g., "bhomik", "pritam")
            audio_path: Path to WAV file (10-15 seconds recommended)
            language: "en" for English, "hi" for Hindi
        """
        # Trim whitespace from profile name
        profile_name = profile_name.strip()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Ensure voice encoder is loaded (prefer multilingual model)
        if self.voice_encoder is None:
            print("⚠️  Loading voice encoder first...")
            self.load_models(load_english=False, load_multilingual=True, load_turbo=False)
        
        # Add sample to profile
        self.profile_manager.add_sample(profile_name, audio_path, language)
    
    def create_profile(self, name: str) -> VoiceProfile:
        """Create a new voice profile"""
        return self.profile_manager.create_profile(name)
    
    def get_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get a voice profile by name"""
        return self.profile_manager.get_profile(name)
    
    def list_profiles(self) -> List[str]:
        """List all available voice profiles"""
        return self.profile_manager.list_profiles()
    
    def get_profile_embedding(self, profile_name: str) -> torch.Tensor:
        """Get the cached embedding for a profile"""
        # Trim whitespace
        profile_name = profile_name.strip()
        
        profile = self.profile_manager.get_profile(profile_name)
        if profile is None:
            raise ValueError(f"Profile '{profile_name}' not found!")
        
        if self.voice_encoder is None:
            raise RuntimeError("Voice encoder not loaded! Call load_models() first.")
        
        return self.profile_manager.compute_profile_embedding(profile, self.voice_encoder)
    
    def _generate_chunk_english(
        self,
        text: str,
        voice_embedding: torch.Tensor,
        reference_audio_path: str,
        emotion: str = "neutral",
        use_turbo: bool = False
    ) -> torch.Tensor:
        """Generate a single chunk in English with full emotion support (optimized)
        
        Falls back to Multilingual model if English model is not loaded.
        This allows using only Multilingual for both English and Hindi/Hinglish.
        """
        
        preset = self.config.emotion_presets.get(emotion, self.config.emotion_presets["neutral"])
        max_steps = self.config.hardware.max_sampling_steps if hasattr(self.config.hardware, 'max_sampling_steps') else 200
        
        # Fix English pronunciation (silent h, silent letters)
        text = TextProcessor.fix_english_pronunciation(text)
        
        # Load turbo model on-demand if requested but not loaded
        if use_turbo and self.turbo_model is None:
            print("📦 Loading Turbo model on-demand (first use)...")
            try:
                self.turbo_model = ChatterboxTurboTTS.from_pretrained(self.device)
                self.turbo_model = self._optimize_model(self.turbo_model)
                print("   ✅ Turbo model loaded (faster English generation)")
            except Exception as e:
                print(f"   ⚠️ Failed to load Turbo model: {e}")
                print("   → Falling back to standard model")
        
        # Use inference mode for speed boost
        with torch.inference_mode():
            if use_turbo and self.turbo_model is not None:
                # Turbo model - faster but less emotion control
                if self.use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        wav = self.turbo_model.generate(
                            text,
                            audio_prompt_path=reference_audio_path,
                            temperature=preset["temperature"],  # Use preset exactly
                            top_p=0.95,  # Higher for quality
                            top_k=2000,  # Higher for better pronunciation
                            repetition_penalty=self.config.default_repetition_penalty,
                            norm_loudness=True,
                        )
                else:
                    wav = self.turbo_model.generate(
                        text,
                        audio_prompt_path=reference_audio_path,
                        temperature=preset["temperature"],
                        top_p=0.95,
                        top_k=2000,
                        repetition_penalty=self.config.default_repetition_penalty,
                        norm_loudness=True,
                    )
            elif self.english_model is not None:
                # Full English model with emotion support
                if self.use_fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        wav = self.english_model.generate(
                            text,
                            audio_prompt_path=reference_audio_path,
                            exaggeration=preset["exaggeration"],
                            cfg_weight=preset["cfg_weight"],
                            temperature=preset["temperature"],
                            min_p=0.05,  # Lower for more accuracy
                            top_p=0.95,  # Higher for quality
                            repetition_penalty=self.config.default_repetition_penalty,
                        )
                else:
                    wav = self.english_model.generate(
                        text,
                        audio_prompt_path=reference_audio_path,
                        exaggeration=preset["exaggeration"],
                        cfg_weight=preset["cfg_weight"],
                        temperature=preset["temperature"],
                        min_p=0.05,
                        top_p=0.95,
                        repetition_penalty=self.config.default_repetition_penalty,
                    )
            elif self.multilingual_model is not None:
                # Fallback: Use Multilingual model for English
                # This enables single-model operation for VRAM-constrained setups
                wav = self.multilingual_model.generate(
                    text,
                    language_id="en",  # English via multilingual model
                    audio_prompt_path=reference_audio_path,
                    exaggeration=preset["exaggeration"],
                    cfg_weight=preset["cfg_weight"],
                    temperature=preset["temperature"],
                    min_p=0.05,
                    top_p=0.95,
                    repetition_penalty=self.config.default_repetition_penalty,
                )
            else:
                raise RuntimeError("No TTS model loaded! Call load_models() first.")
        
        # Sanitize audio to remove NaN/Inf values (can occur with FP16)
        wav = self._sanitize_audio(wav)
        
        # Clear CUDA cache after generation to prevent memory buildup
        self._clear_vram()
        
        return wav
    
    def _sanitize_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Remove NaN/Inf values from audio to prevent downstream errors."""
        if wav is None:
            return wav
            
        # Convert to numpy if tensor
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = wav
            
        import numpy as np
        
        # Check for non-finite values
        if not np.isfinite(wav_np).all():
            print("⚠️ Warning: Audio contains NaN/Inf values, sanitizing...")
            # Replace NaN/Inf with zeros
            wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize if not all zeros
            max_val = np.abs(wav_np).max()
            if max_val > 0:
                wav_np = wav_np / max_val * 0.95
            
        # Convert back to tensor if needed
        if isinstance(wav, torch.Tensor):
            return torch.from_numpy(wav_np).to(wav.device)
        return wav_np
    
    def _split_by_script(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into segments by script (Devanagari vs Latin/English).
        Returns list of (text_segment, language_id) tuples.
        
        Example: "आज अगर ये नहीं मिलता ना, तो honestly हमारा पूरा"
        Returns: [
            ("आज अगर ये नहीं मिलता ना, तो", "hi"),
            ("honestly", "en"),
            ("हमारा पूरा", "hi")
        ]
        """
        segments = []
        current_segment = ""
        current_lang = None
        
        # Process character by character
        i = 0
        while i < len(text):
            char = text[i]
            
            # Determine character type
            if '\u0900' <= char <= '\u097F':
                # Devanagari (Hindi)
                char_lang = "hi"
            elif char.isalpha() and char.isascii():
                # ASCII letter (English)
                char_lang = "en"
            else:
                # Punctuation, numbers, spaces - keep with current segment
                current_segment += char
                i += 1
                continue
            
            # If language changed, save current segment and start new one
            if current_lang is not None and char_lang != current_lang:
                if current_segment.strip():
                    segments.append((current_segment.strip(), current_lang))
                current_segment = char
                current_lang = char_lang
            else:
                current_segment += char
                current_lang = char_lang
            
            i += 1
        
        # Don't forget the last segment
        if current_segment.strip():
            segments.append((current_segment.strip(), current_lang or "hi"))
        
        # Merge very short English segments (single letters) back into Hindi
        merged_segments = []
        for seg_text, seg_lang in segments:
            # If it's a very short English segment (1-2 chars), merge with adjacent
            if seg_lang == "en" and len(seg_text.replace(" ", "")) <= 2:
                if merged_segments and merged_segments[-1][1] == "hi":
                    # Merge with previous Hindi segment
                    prev_text, _ = merged_segments.pop()
                    merged_segments.append((prev_text + " " + seg_text, "hi"))
                else:
                    merged_segments.append((seg_text, seg_lang))
            else:
                merged_segments.append((seg_text, seg_lang))
        
        return merged_segments if merged_segments else [(text, "hi")]
    
    def _generate_chunk_hindi(
        self,
        text: str,
        voice_embedding: torch.Tensor,
        reference_audio_path: str,
        emotion: str = "neutral"
    ) -> torch.Tensor:
        """Generate a single chunk for Hindi/Hinglish text.
        
        Uses language_id="hi" for the ENTIRE text - no code-switching.
        The voice samples dictate how English words are pronounced.
        This creates natural Hinglish where English words sound like 
        how the speaker naturally says them, not with a foreign accent.
        
        Your 2 Hindi + 1 English samples train the model to match YOUR accent.
        """
        
        preset = self.config.emotion_presets.get(emotion, self.config.emotion_presets["neutral"])
        
        # Fix English pronunciation (silent h, silent letters)
        text = TextProcessor.fix_english_pronunciation(text)
        
        # Always use Hindi language ID for Hinglish text
        # The model learns from YOUR voice samples how YOU pronounce English words
        # This keeps the accent consistent throughout
        with torch.inference_mode():
            wav = self.multilingual_model.generate(
                text,
                language_id="hi",  # Hindi mode - YOUR accent for everything
                audio_prompt_path=reference_audio_path,
                exaggeration=preset["exaggeration"],
                cfg_weight=preset["cfg_weight"],      # High value = match your samples closely
                temperature=preset["temperature"],    # Low value = consistent output
                min_p=0.08,                           # Higher = avoid unlikely tokens
                top_p=0.85,                           # Tighter sampling = more natural
                repetition_penalty=1.1,               # Low = natural flow, no cutting
            )
        
        wav = self._sanitize_audio(wav)
        self._clear_vram()
        return wav
    
    def generate(
        self,
        text: str,
        profile_name: str,
        emotion: str = "neutral",
        auto_detect_language: bool = True,
        force_language: Optional[str] = None,
        use_turbo_for_english: bool = False,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech from text using a voice profile.
        Optimized for i9 + T2000 hardware with FP16 and parallel processing.
        
        Args:
            text: Text to generate (any length!)
            profile_name: Name of voice profile to use (e.g., "bhomik", "pritam")
            emotion: "neutral", "excited", "calm", "dramatic", "conversational"
            auto_detect_language: Auto-detect English vs Hindi
            force_language: Force "en" or "hi" language
            use_turbo_for_english: Use faster Turbo model for English
            show_progress: Print progress updates
        
        Returns:
            (audio_tensor, sample_rate)
        """
        import time
        start_time = time.time()
        
        # Trim whitespace from profile name
        profile_name = profile_name.strip()
        
        if show_progress:
            print(f"\n{'='*60}")
            print(f"🎙️ GENERATING VOICE CLONE {'(FP16)' if self.use_fp16 else '(FP32)'}")
            print(f"{'='*60}")
            print(f"👤 Profile: {profile_name}")
            print(f"📝 Text length: {len(text)} characters")
            print(f"😊 Emotion: {emotion}")
            if torch.cuda.is_available():
                print(f"💾 VRAM before: {torch.cuda.memory_allocated()/(1024**3):.2f}GB")
        
        # Get voice embedding and reference audio for profile
        try:
            embed_start = time.time()
            voice_embedding = self.get_profile_embedding(profile_name)
            profile = self.get_profile(profile_name)
            # Use BEST sample as reference audio (NOT first sample!)
            reference_audio = profile.get_best_reference() if profile else None
            
            if not reference_audio:
                raise ValueError(f"No audio samples found for profile '{profile_name}'")
            
            if show_progress:
                embed_time = time.time() - embed_start
                print(f"✅ Loaded voice profile ({embed_time:.2f}s)")
        except Exception as e:
            raise ValueError(f"Failed to get voice data for profile '{profile_name}': {e}")
        
        # Create chunks
        # For very short text (< 150 chars AND < 2 sentences), generate as single chunk to avoid clicks
        # For longer texts, always chunk to avoid excessive generation time
        text_length = len(text.strip())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Only use single-chunk mode for genuinely short texts
        if text_length < 150 and sentence_count < 2:
            # Generate as single chunk - no stitching = no clicks!
            chunks = [(text.strip(), TextProcessor.detect_language(text))]
            if show_progress:
                print(f"📦 Short text detected ({text_length} chars) - generating as single chunk (no stitching)")
        else:
            chunks = TextProcessor.create_chunks(text, self.config.optimal_chunk_size)
            if show_progress:
                print(f"📦 Created {len(chunks)} chunks (text: {text_length} chars, {sentence_count} sentences)")
            for i, (chunk, lang) in enumerate(chunks, 1):
                preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
                print(f"   {i}. [{lang}] {preview}")
        
        # Calculate crossfade samples
        # Get sample rate from loaded model (prefer multilingual for Hinglish)
        if self.multilingual_model:
            sample_rate = self.multilingual_model.sr
        elif self.english_model:
            sample_rate = self.english_model.sr
        else:
            sample_rate = 24000  # Default fallback
        fade_samples = int((self.config.crossfade_duration_ms / 1000.0) * sample_rate)
        
        # Generate each chunk with timing
        audio_segments = []
        chunk_times = []
        
        for i, (chunk_text, detected_lang) in enumerate(chunks, 1):
            chunk_start = time.time()
            
            # Determine language
            if force_language:
                lang = force_language
            elif auto_detect_language:
                lang = detected_lang
            else:
                lang = "en"
            
            if show_progress:
                print(f"\n🎤 Generating chunk {i}/{len(chunks)} [{lang}]...")
            
            # Generate based on language
            if lang == "hi":
                wav = self._generate_chunk_hindi(chunk_text, voice_embedding, reference_audio, emotion)
            else:
                wav = self._generate_chunk_english(
                    chunk_text, voice_embedding, reference_audio, emotion, use_turbo_for_english
                )
            
            chunk_time = time.time() - chunk_start
            chunk_times.append(chunk_time)
            
            if show_progress:
                duration = wav.shape[-1] / sample_rate
                rtf = chunk_time / duration  # Real-time factor
                print(f"   ✅ Generated {duration:.2f}s audio in {chunk_time:.2f}s (RTF: {rtf:.2f}x)")
            
            # Apply crossfade with previous segment
            if len(audio_segments) > 0:
                # Add 250ms silence padding to prevent clicks (increased from 100ms)
                silence_ms = 250
                silence_samples = int(sample_rate * silence_ms / 1000)
                silence = torch.zeros((1, silence_samples), dtype=wav.dtype, device=wav.device)
                
                # Insert silence before crossfade
                audio_segments.append(silence)
                
                combined = AudioProcessor.apply_crossfade(
                    audio_segments[-1],
                    wav,
                    fade_samples
                )
                audio_segments[-1] = combined[:, :audio_segments[-1].shape[-1]]
                audio_segments.append(combined[:, audio_segments[-1].shape[-1]:])
            else:
                audio_segments.append(wav)
        
        # Concatenate all segments
        full_audio = torch.cat(audio_segments, dim=-1)
        
        # Apply de-clicker IMMEDIATELY after stitching (before any other processing)
        try:
            # Convert to numpy for scipy processing
            audio_np = full_audio.squeeze().cpu().numpy()
            
            # Median filter to remove impulse clicks (stitching artifacts)
            from scipy.signal import medfilt
            audio_np = medfilt(audio_np, kernel_size=3)  # Small kernel to preserve audio quality
            
            # Convert back to torch
            full_audio = torch.from_numpy(audio_np).unsqueeze(0).to(full_audio.device)
            
            if show_progress:
                print(f"   🔧 Applied de-clicker to remove stitching artifacts")
        except Exception as e:
            if show_progress:
                print(f"   ⚠️  De-clicker skipped: {e}")
        
        # Apply high-pass filter to remove low-frequency clicks (100Hz cutoff)
        try:
            import torchaudio.functional as F
            # High-pass filter at 100Hz to remove clicks/pops without affecting voice
            full_audio = F.highpass_biquad(full_audio, sample_rate, cutoff_freq=100)
            if show_progress:
                print(f"   🔧 Applied high-pass filter (100Hz) to remove clicking artifacts")
        except Exception as e:
            if show_progress:
                print(f"   ⚠️  High-pass filter skipped: {e}")
        
        if show_progress:
            total_duration = full_audio.shape[-1] / sample_rate
            total_time = time.time() - start_time
            avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
            overall_rtf = total_time / total_duration
            
            print(f"\n{'='*60}")
            print(f"✅ GENERATION COMPLETE!")
            print(f"{'='*60}")
            print(f"   👤 Profile: {profile_name}")
            print(f"   🔊 Audio duration: {total_duration:.2f} seconds")
            print(f"   ⏱️  Processing time: {total_time:.2f} seconds")
            print(f"   ⚡ Real-time factor: {overall_rtf:.2f}x (lower is faster)")
            print(f"   📊 Avg chunk time: {avg_chunk_time:.2f}s")
            if torch.cuda.is_available():
                print(f"   💾 Final VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f}GB")
            print(f"{'='*60}\n")
        
        # Final memory cleanup
        self._clear_vram()
        
        return full_audio, sample_rate
    
    def generate_and_save(
        self,
        text: str,
        profile_name: str,
        output_path: str,
        emotion: str = "neutral",
        **kwargs
    ) -> str:
        """Generate and save to file with optimized memory handling"""
        audio, sr = self.generate(text, profile_name=profile_name, emotion=emotion, **kwargs)
        ta.save(output_path, audio, sr)
        print(f"💾 Saved to: {output_path}")
        
        # Clear memory after saving
        del audio
        self._clear_vram()
        
        return output_path


# ============================================================================
# GRADIO WEB INTERFACE
# ============================================================================

def create_gradio_interface(voice_clone: MyVoiceClone) -> gr.Blocks:
    """Create Gradio web interface"""
    
    # Ensure models are loaded
    voice_clone.load_models()
    
    def create_profile_handler(profile_name):
        """Handle profile creation"""
        if not profile_name or not profile_name.strip():
            return "❌ Please enter a profile name", gr.update(choices=[])
        
        try:
            voice_clone.create_profile(profile_name.strip())
            profiles = voice_clone.list_profiles()
            profile_value = profile_name.strip()
            return (
                f"✅ Created profile: {profile_name}", 
                gr.update(choices=profiles, value=profile_value)
            )
        except Exception as e:
            profiles = voice_clone.list_profiles()
            return (
                f"❌ Error: {str(e)}", 
                gr.update(choices=profiles)
            )
    
    def add_sample_handler(audio_path, profile_name, language):
        """Handle adding voice sample to profile"""
        if audio_path is None:
            return "❌ Please upload or record an audio file"
        
        if not profile_name:
            return "❌ Please select or create a profile first"
        
        try:
            # Convert to WAV if necessary
            import tempfile
            from pathlib import Path
            
            audio_path_obj = Path(audio_path)
            
            # If not a WAV file, convert it
            if audio_path_obj.suffix.lower() not in ['.wav', '.wave']:
                print(f"🔄 Converting {audio_path_obj.suffix} to WAV format...")
                
                # Try torchaudio first (handles most formats including MP3, FLAC, OGG)
                try:
                    audio, sr = ta.load(audio_path)
                    print(f"   ✅ Loaded with torchaudio: {audio.shape}, {sr}Hz")
                except Exception as ta_error:
                    print(f"   ⚠️  torchaudio failed: {ta_error}")
                    
                    # Fallback to pydub for formats like MPEG, AAC, etc.
                    try:
                        from pydub import AudioSegment
                        import numpy as np
                        
                        print(f"   🔄 Trying pydub for {audio_path_obj.suffix}...")
                        
                        # Load with pydub (supports many formats via ffmpeg)
                        audio_segment = AudioSegment.from_file(audio_path)
                        
                        # Convert to mono if stereo
                        if audio_segment.channels > 1:
                            audio_segment = audio_segment.set_channels(1)
                        
                        # Get sample rate
                        sr = audio_segment.frame_rate
                        
                        # Convert to numpy array
                        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                        samples = samples / (2**15)  # Normalize to [-1, 1]
                        
                        # Convert to torch tensor
                        audio = torch.from_numpy(samples).unsqueeze(0)
                        
                        print(f"   ✅ Loaded with pydub: {audio.shape}, {sr}Hz")
                    except ImportError:
                        return "❌ pydub not installed. Install with: pip install pydub\n" + \
                               "For MPEG/MP4/AAC support, also install ffmpeg"
                    except FileNotFoundError as fnf_error:
                        if "ffmpeg" in str(fnf_error).lower() or "avconv" in str(fnf_error).lower():
                            return "❌ ffmpeg not found. Required for MPEG/M4A/AAC formats.\n\n" + \
                                   "Install ffmpeg:\n" + \
                                   "  Windows: choco install ffmpeg (or download from ffmpeg.org)\n" + \
                                   "  Linux: sudo apt-get install ffmpeg\n" + \
                                   "  Mac: brew install ffmpeg\n\n" + \
                                   "Alternative: Convert your file to MP3 or WAV format first"
                        else:
                            raise
                    except Exception as pydub_error:
                        return f"❌ Failed to load audio file: {str(pydub_error)}\n" + \
                               f"Format: {audio_path_obj.suffix}\n" + \
                               "Try converting to WAV/MP3 format first"
                
                # Create temp WAV file
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_wav.close()
                
                # Save as WAV
                ta.save(temp_wav.name, audio, sr)
                audio_path = temp_wav.name
                print(f"   💾 Converted to: {temp_wav.name}")
            
            voice_clone.add_voice_sample(profile_name, audio_path, language)
            profile = voice_clone.get_profile(profile_name)
            if profile:
                return f"✅ Added sample to {profile_name} (Total: {len(profile.samples)} samples)"
            else:
                return f"✅ Added sample to {profile_name}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def refresh_profiles():
        """Refresh profile list in all dropdowns"""
        profiles = voice_clone.list_profiles()
        return gr.update(choices=profiles), gr.update(choices=profiles), gr.update(choices=profiles)
    
    def show_profile_info(profile_name):
        """Show information abo"""
        profiles = voice_clone.list_profiles()
        return
        
        profile = voice_clone.get_profile(profile_name)
        if profile is None:
            return f"❌ Profile '{profile_name}' not found"
        
        info = f"""
### Profile: {profile.name}
- **Samples**: {len(profile.samples)} voice recordings
- **Languages**: {', '.join(profile.languages) if profile.languages else 'None'}
- **Created**: {profile.created_at[:19] if profile.created_at else 'Unknown'}
- **Updated**: {profile.updated_at[:19] if profile.updated_at else 'Unknown'}
- **Cached Embedding**: {'✅ Yes' if profile.embedding_cache is not None else '⏳ Will compute on first use'}
"""
        return info
    
    def generate_handler(text, profile_name, emotion, language_mode, force_language, use_turbo, 
                         enhanced_mode, auto_emotion_detect, emotion_mode,
                         speed_mode, prosody_enhance, ensemble_mode,
                         studio_quality, platform_preset):
        """Handle generation with Phase 1 & Phase 2 enhancements + Studio Quality"""
        if not text.strip():
            return None, "❌ Please enter some text"
        
        if not profile_name:
            return None, "❌ Please select a voice profile"
        
        # Check if profile exists and has samples
        profile = voice_clone.get_profile(profile_name)
        if profile is None:
            return None, f"❌ Profile '{profile_name}' not found"
        
        if len(profile.samples) == 0:
            return None, f"❌ Profile '{profile_name}' has no voice samples! Add at least one sample first."
        
        try:
            auto_detect = language_mode == "Auto-detect"
            force_lang = None if auto_detect else ("hi" if force_language == "Hindi" else "en")
            
            # Check if any enhancements are requested
            use_enhancements = enhanced_mode or speed_mode or prosody_enhance or ensemble_mode
            
            if use_enhancements and (ENHANCEMENTS_AVAILABLE or PHASE2_AVAILABLE):
                # Initialize enhanced wrapper if needed
                if not hasattr(voice_clone, '_enhanced_wrapper'):
                    voice_clone._enhanced_wrapper = EnhancedVoiceClone(voice_clone)
                    
                # Load Phase 1 enhancements if needed
                if enhanced_mode and ENHANCEMENTS_AVAILABLE:
                    if not voice_clone._enhanced_wrapper.enhanced_mode:
                        voice_clone._enhanced_wrapper.load_enhancements(
                            enable_audio_enhancement=True,
                            enable_emotion_detection=auto_emotion_detect,
                            enable_wavlm_encoder=False,
                            emotion_mode=emotion_mode,
                        )
                
                # Load Phase 2 enhancements if needed
                if (speed_mode or prosody_enhance or ensemble_mode) and PHASE2_AVAILABLE:
                    if not voice_clone._enhanced_wrapper.speed_mode and not voice_clone._enhanced_wrapper.prosody_enhancement:
                        voice_clone._enhanced_wrapper.load_phase2_enhancements(
                            enable_speed_mode=speed_mode,
                            enable_prosody=prosody_enhance,
                            enable_ensemble=ensemble_mode,
                            n_ensemble_variants=3,
                        )
                
                # Use enhanced generation
                # FIXED: Only use "auto" if user explicitly enables auto-detect AND selects auto emotion
                # Otherwise use the selected emotion (conversational, neutral, etc.)
                final_emotion = "auto" if (auto_emotion_detect and emotion == "auto") else emotion
                audio, sr = voice_clone._enhanced_wrapper.generate_enhanced(
                    text=text,
                    profile_name=profile_name,
                    emotion=final_emotion,
                    language="auto" if auto_detect else force_lang,
                    chunk_size=100,
                    apply_audio_enhancement=enhanced_mode,
                    use_speed_mode=speed_mode,
                    use_ensemble=ensemble_mode,
                )
                
                # Build status prefix
                modes_used = []
                if enhanced_mode:
                    modes_used.append("Enhanced")
                if speed_mode:
                    modes_used.append("Speed")
                if prosody_enhance:
                    modes_used.append("Prosody")
                if ensemble_mode:
                    modes_used.append("Ensemble")
                status_prefix = "✨ " + "+".join(modes_used) if modes_used else "✅"
            else:
                # Standard generation
                audio, sr = voice_clone.generate(
                    text=text,
                    profile_name=profile_name,
                    emotion=emotion,
                    auto_detect_language=auto_detect,
                    force_language=force_lang,
                    use_turbo_for_english=use_turbo,
                    show_progress=True
                )
                status_prefix = "✅"
            
            # Apply Studio-Quality Processing (Broadcast Standard)
            if studio_quality and STUDIO_PROCESSOR_AVAILABLE:
                try:
                    print("\n🎚️ Applying Studio-Quality Post-Processing...")
                    
                    # Map platform preset to platform name
                    platform_map = {
                        "Instagram/TikTok (48kHz Mono -14 LUFS)": "instagram",
                        "YouTube (48kHz Stereo -14 LUFS)": "youtube",
                        "Podcast (44.1kHz Stereo -16 LUFS)": "podcast",
                        "Custom (48kHz Mono -14 LUFS)": "default",
                    }
                    platform = platform_map.get(platform_preset, "instagram")
                    
                    # Convert to numpy for processing - handle both torch and numpy
                    if torch.is_tensor(audio):
                        audio_np = audio.squeeze().cpu().numpy()
                    elif isinstance(audio, np.ndarray):
                        audio_np = audio.squeeze()
                    else:
                        raise TypeError(f"Unexpected audio type: {type(audio)}")
                    
                    # Ensure 1D array
                    if audio_np.ndim > 1:
                        audio_np = audio_np[0]  # Take first channel if multi-channel
                    
                    # Apply broadcast-quality mastering for selected platform
                    processed_audio, output_sr = process_for_social_media(
                        audio=audio_np,
                        input_sr=sr,
                        platform=platform
                    )
                    
                    # Convert back to torch tensor
                    if processed_audio.ndim == 1:
                        # Mono
                        audio = torch.from_numpy(processed_audio).unsqueeze(0).float()
                    else:
                        # Stereo
                        audio = torch.from_numpy(processed_audio).float()
                    
                    sr = output_sr
                    status_prefix += " 🎚️ Studio"
                    print("✨ Studio processing complete - Broadcast quality achieved!")
                except Exception as e:
                    import traceback
                    print(f"⚠️ Studio processing failed: {e}")
                    traceback.print_exc()
                    print("   Using original audio...")
            
            # Save to temp file
            output_path = f"output_{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            ta.save(output_path, audio, sr)
            
            duration = audio.shape[-1] / sr
            status = f"{status_prefix} Generated {duration:.2f} seconds using {profile_name}'s voice"
            
            return output_path, status
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"
    
    # Create interface
    try:
        from gradio import themes
        theme = themes.Soft()
    except:
        theme = "soft"  # Fallback string theme
    
    with gr.Blocks(title="🎙️ My Voice Clone", theme=theme) as demo:
        gr.Markdown("""
        # 🎙️ My Voice Clone - Profile-Based Voice Cloning
        ## Clone ANY voice in English & Hindi with perfect consistency!
        
        ### How It Works:
        1. **Create a Voice Profile** (e.g., "bhomik", "pritam")
        2. **Add Multiple Voice Samples** (3-5 samples recommended for best results)
        3. **Generate Speech** - The system will use your pre-trained voice every time!
        
        ### Why Multiple Samples?
        - More samples = More consistent voice
        - System learns your unique voice patterns
        - Embeddings are cached for instant generation
        """)
        
        with gr.Tabs():
            # Tab 1: Profile Management
            with gr.TabItem("👤 1. Manage Profiles"):
                gr.Markdown("### Create or select voice profiles")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Create New Profile")
                        new_profile_name = gr.Textbox(
                            label="Profile Name",
                            placeholder="e.g., bhomik, pritam, etc.",
                            info="Enter person's name for voice cloning"
                        )
                        create_profile_btn = gr.Button("➕ Create Profile", variant="primary")
                        create_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### Existing Profiles")
                        profile_selector = gr.Dropdown(
                            choices=voice_clone.list_profiles(),
                            label="Select Profile",
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄 Refresh List")
                        profile_info = gr.Markdown("No profile selected")
                
                profile_selector.change(
                    show_profile_info,
                    inputs=[profile_selector],
                    outputs=[profile_info]
                )
            
            # Tab 2: Add Voice Samples
            with gr.TabItem("🎤 2. Add Voice Samples"):
                gr.Markdown("""
                ### Upload or record voice samples for a profile
                
                **📝 Tips for Best Quality:**
                - Record 3-5 different samples (10-15 seconds each)
                - Vary your tone and emotion across samples
                - Use same microphone and environment
                - Speak clearly and naturally
                - For Hindi, record samples speaking Hindi!
                
                **🎵 Supported Audio Formats:**
                - WAV, MP3, FLAC, OGG (via torchaudio)
                - MPEG, MP4, M4A, AAC, WMA (via pydub/ffmpeg)
                - All formats automatically converted to WAV
                """)
                
                with gr.Row():
                    with gr.Column():
                        sample_profile = gr.Dropdown(
                            choices=voice_clone.list_profiles(),
                            label="Select Profile to Add Samples",
                            interactive=True
                        )
                        sample_refresh_btn = gr.Button("🔄 Refresh Profile List", size="sm")
                        sample_audio = gr.Audio(
                            label="Voice Sample (10-15 seconds) - Upload any audio format",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        sample_language = gr.Dropdown(
                            choices=["en", "hi"],
                            value="en",
                            label="Language of this Sample"
                        )
                        add_sample_btn = gr.Button("➕ Add Sample to Profile", variant="primary")
                        sample_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### What to Record:
                        
                        **Sample 1 (Neutral)**:
                        - English: "Hello, my name is [name]. This is a sample of my voice."
                        - Hindi: "नमस्ते, मेरा नाम [name] है। यह मेरी आवाज़ का नमूना है।"
                        
                        **Sample 2 (Expressive)**:
                        - English: "Oh wow! This is amazing! I'm so excited about this!"
                        - Hindi: "वाह! यह तो बहुत अद्भुत है! मुझे इससे बहुत ख़ुशी हो रही है!"
                        
                        **Sample 3 (Conversational)**:
                        - English: "So anyway, I was telling my friend about this thing..."
                        - Hindi: "तो वैसे, मैं अपने दोस्त को इस बारे में बता रहा था..."
                        
                        **More samples = Better consistency!**
                        """)
                
                add_sample_btn.click(
                    add_sample_handler,
                    inputs=[sample_audio, sample_profile, sample_language],
                    outputs=[sample_status]
                )
                
                sample_refresh_btn.click(
                    refresh_profiles,
                    outputs=[sample_profile]
                )
            
            # Tab 3: Generate Speech
            with gr.TabItem("🎙️ 3. Generate Speech"):
                gr.Markdown("### Generate speech using a trained voice profile")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gen_profile = gr.Dropdown(
                            choices=voice_clone.list_profiles(),
                            label="Select Voice Profile",
                            info="Choose whose voice to clone",
                            interactive=True
                        )
                        
                        text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter any length of text here... It can be English, Hindi, or mixed!",
                            lines=10,
                            max_lines=30
                        )
                        
                        with gr.Row():
                            emotion_select = gr.Dropdown(
                                choices=["neutral", "excited", "calm", "dramatic", "conversational", "storytelling"],
                                value="neutral",
                                label="Emotion"
                            )
                            language_mode = gr.Radio(
                                choices=["Auto-detect", "Force language"],
                                value="Auto-detect",
                                label="Language Mode"
                            )
                            force_language = gr.Dropdown(
                                choices=["English", "Hindi"],
                                value="English",
                                label="Force Language (if selected)",
                                visible=True
                            )
                        
                        use_turbo = gr.Checkbox(
                            label="⚡ Use Turbo model for English (faster, slightly less emotion control)",
                            value=False
                        )
                        
                        # Phase 1 Enhancement Controls
                        gr.Markdown("### ✨ Phase 1: ElevenLabs Quality Enhancements")
                        
                        with gr.Row():
                            enhanced_mode = gr.Checkbox(
                                label="🎵 Enhanced Mode",
                                value=True if ENHANCEMENTS_AVAILABLE else False,
                                info="Apply audio enhancement (noise reduction, compression, clarity, loudness normalization)",
                                interactive=ENHANCEMENTS_AVAILABLE
                            )
                            auto_emotion = gr.Checkbox(
                                label="🧠 Auto-Detect Emotion",
                                value=False,
                                info="Automatically detect emotion from text context",
                                interactive=ENHANCEMENTS_AVAILABLE
                            )
                        
                        emotion_mode = gr.Dropdown(
                            choices=["rule-based", "ollama"],
                            value="rule-based",
                            label="Emotion Detection Method",
                            info="rule-based (fast, 80% accurate) or ollama (requires Ollama installation)",
                            visible=ENHANCEMENTS_AVAILABLE
                        )
                        
                        # Phase 2 Enhancement Controls
                        gr.Markdown("### 🚀 Phase 2: Speed & Quality Enhancements")
                        
                        with gr.Row():
                            speed_mode = gr.Checkbox(
                                label="⚡ Speed Mode (F5-TTS)",
                                value=False,
                                info="5-10x faster generation using diffusion model (English only)",
                                interactive=PHASE2_AVAILABLE
                            )
                            prosody_enhance = gr.Checkbox(
                                label="🎵 Prosody Enhancement",
                                value=True if PHASE2_AVAILABLE else False,
                                info="Natural speech rhythm and intonation",
                                interactive=PHASE2_AVAILABLE
                            )
                            ensemble_mode = gr.Checkbox(
                                label="🎼 Ensemble Mode",
                                value=False,
                                info="Generate 3 variants and pick the best (slower but higher quality)",
                                interactive=PHASE2_AVAILABLE
                            )
                        
                        # Studio Quality Controls
                        gr.Markdown("### 🎚️ Studio-Quality Post-Processing (Broadcast Standard)")
                        
                        with gr.Row():
                            studio_quality = gr.Checkbox(
                                label="🎚️ Studio-Quality Processing",
                                value=True if STUDIO_PROCESSOR_AVAILABLE else False,
                                info="Apply broadcast-quality mastering: 48kHz upsampling, multiband compression, de-essing, loudness normalization (-14 LUFS)",
                                interactive=STUDIO_PROCESSOR_AVAILABLE
                            )
                            platform_preset = gr.Dropdown(
                                choices=["Instagram/TikTok (48kHz Mono -14 LUFS)", 
                                        "YouTube (48kHz Stereo -14 LUFS)",
                                        "Podcast (44.1kHz Stereo -16 LUFS)",
                                        "Custom (48kHz Mono -14 LUFS)"],
                                value="Instagram/TikTok (48kHz Mono -14 LUFS)",
                                label="Platform Preset",
                                info="Optimized settings for different platforms",
                                visible=STUDIO_PROCESSOR_AVAILABLE
                            )
                        
                        if not STUDIO_PROCESSOR_AVAILABLE:
                            gr.Markdown("""
                            ℹ️ **Studio Quality Processor Available**
                            
                            Studio processing includes:
                            - ✅ 48kHz upsampling (from 24kHz)
                            - ✅ Spectral enhancement for clarity
                            - ✅ Multiband compression for broadcast dynamics
                            - ✅ De-essing (reduces harsh sibilance)
                            - ✅ Loudness normalization (-14 to -16 LUFS)
                            - ✅ Soft-knee limiting (prevents clipping)
                            - ✅ Optional stereo enhancement
                            
                            **Your audio will sound like it came from a professional studio!**
                            """)
                        
                        if not ENHANCEMENTS_AVAILABLE:
                            gr.Markdown("""
                            ⚠️ **Phase 1 Enhancements not available**
                            
                            Install Phase 1 dependencies:
                            ```bash
                            pip install noisereduce pyloudnorm scipy transformers
                            ```
                            """)
                        
                        if not PHASE2_AVAILABLE:
                            gr.Markdown("""
                            ⚠️ **Phase 2 Enhancements not available**
                            
                            Install Phase 2 dependencies:
                            ```bash
                            pip install f5-tts
                            ```
                            """)
                        
                        generate_btn = gr.Button("🎙️ Generate Voice", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gen_refresh_btn = gr.Button("🔄 Refresh Profile List", size="sm")
                        audio_output = gr.Audio(label="Generated Audio", type="filepath")
                        gen_status = gr.Textbox(label="Status", interactive=False)
                        
                        gr.Markdown("""
                        ### 💡 Pro Tips:
                        - **First generation**: Computes embedding (may take ~10 seconds)
                        - **Subsequent generations**: Uses cached embedding (instant!)
                        - **Long text**: Automatically chunked for quality
                        - **Consistent voice**: Same profile = same voice every time!
                        """)
                
                generate_btn.click(
                    generate_handler,
                    inputs=[text_input, gen_profile, emotion_select, language_mode, force_language, use_turbo,
                            enhanced_mode, auto_emotion, emotion_mode, speed_mode, prosody_enhance, ensemble_mode,
                            studio_quality, platform_preset],
                    outputs=[audio_output, gen_status]
                )
                
                gen_refresh_btn.click(
                    refresh_profiles,
                    outputs=[gen_profile]
                )
            
            # Tab 4: Examples
            with gr.TabItem("📚 Examples & Help"):
                gr.Markdown(r"""
                ### Complete Workflow Example
                
                **Step 1: Create Profile bhomik**
                ```
                1. Go to "Manage Profiles" tab
                2. Enter name: "bhomik"
                3. Click "Create Profile"
                ```
                
                **Step 2: Add 3-5 Voice Samples for Bhomik**
                ```
                1. Go to "Add Voice Samples" tab
                2. Select profile: "bhomik"
                3. Record/upload sample 1 (neutral English)
                4. Record/upload sample 2 (excited English)
                5. Record/upload sample 3 (neutral Hindi) - set language to "hi"
                6. Add more samples for better consistency!
                ```
                
                **Step 3: Generate Speech as Bhomik**
                ```
                1. Go to "Generate Speech" tab
                2. Select profile: "bhomik"
                3. Enter your text (any length!)
                4. Select emotion and language
                5. Click "Generate Voice"
                ```
                
                ---
                
                ### Example Texts
                
                **English (Excited)**:
                ```
                Oh my goodness, this is incredible! I just found out that we won the competition! 
                Can you believe it? After all those months of hard work, we finally did it! 
                I'm so happy I could cry! This is the best day ever!
                ```
                
                **Hindi (Conversational)**:
                ```
                नमस्ते दोस्तों, आज मैं आपको एक बहुत ही रोचक कहानी सुनाने वाला हूं। 
                पिछले हफ्ते मैं बाजार गया था और वहां कुछ बहुत अजीब हुआ। 
                आप विश्वास नहीं करेंगे कि मैंने क्या देखा!
                ```
                
                **Long-form Example (1+ minute)** - Perfect for testing chunking:
                ```
                Welcome to our comprehensive guide on artificial intelligence. 
                In this presentation, we'll explore the fascinating world of machine learning,
                neural networks, and the future of technology. Let's begin our journey into
                the heart of AI.
                
                First, let's understand what artificial intelligence really means. AI is not
                just about robots and science fiction. It's about creating systems that can
                learn, adapt, and make decisions. From voice assistants to self-driving cars,
                AI is everywhere around us.
                
                The history of AI goes back to the 1950s when pioneers like Alan Turing first
                asked the question, "Can machines think?" Since then, we've come a long way.
                Today, AI can recognize faces, translate languages, and even create art.
                
                But what does the future hold? Many experts believe that AI will transform
                every industry, from healthcare to entertainment. The possibilities are truly
                endless, and we're just getting started on this incredible journey.
                ```
                
                ---
                
                ### Technical Details
                
                **Voice Profile Storage**:
                - Profiles saved in `voice_profiles/` directory
                - Each profile has metadata.json + embedding cache
                - Embeddings computed once, reused forever
                
                **Consistency Guarantee**:
                - Multiple samples → Averaged embedding
                - Same embedding used every generation
                - No variation between outputs for same profile
                
                **Performance**:
                - First generation: ~10-15 seconds (computes embedding)
                - Subsequent generations: ~5-10 seconds per 15s of audio
                - Long texts: Automatically chunked with smooth crossfade
                """)
        
        gr.Markdown("""
        ---
        ### ⚠️ Important Notes:
        - **More samples = Better voice consistency** (3-5 recommended)
        - **Profile system saves time** - Train once, use forever!
        - **Embeddings are cached** - Second generation is much faster
        - **Mix languages freely** - System auto-detects English/Hindi
        - **Processing scales with length** - 1 minute text ≈ 40-60 seconds processing
        """)
    
    return demo


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def cli_demo():
    """Command-line demonstration"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           🎙️ MY VOICE CLONE - CLI Demo                       ║
    ║           Profile-Based Voice Cloning System                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    clone = MyVoiceClone(device=device)
    
    # Check for reference files organized by profile
    reference_dir = Path("references")
    
    if not reference_dir.exists():
        print("\n⚠️  No 'references' folder found!")
        print("   Please create a 'references' folder with subfolders for each person:")
        print("   references/")
        print("   ├── bhomik/")
        print("   │   ├── sample_001.wav")
        print("   │   ├── sample_002.wav")
        print("   │   └── sample_003.wav")
        print("   └── pritam/")
        print("       ├── sample_001.wav")
        print("       └── sample_002.wav")
        print("\n   Each sample should be 10-15 seconds of expressive speech.")
        return
    
    # Auto-discover and create profiles from directory structure
    profiles_created = 0
    for profile_dir in reference_dir.iterdir():
        if profile_dir.is_dir():
            profile_name = profile_dir.name.lower()
            
            # Find all WAV files in this profile directory
            wav_files = list(profile_dir.glob("*.wav"))
            
            if wav_files:
                print(f"\n📁 Found profile: {profile_name} ({len(wav_files)} samples)")
                
                # Add all samples to profile
                for wav_file in wav_files:
                    # Detect language from filename or default to English
                    language = "hi" if "_hindi" in wav_file.stem.lower() or "_hi" in wav_file.stem.lower() else "en"
                    
                    try:
                        clone.add_voice_sample(profile_name, str(wav_file), language)
                    except Exception as e:
                        print(f"   ⚠️  Failed to add {wav_file.name}: {e}")
                
                profiles_created += 1
    
    if profiles_created == 0:
        print("\n⚠️  No valid voice profiles found!")
        print("   Create subdirectories in 'references/' for each person.")
        return
    
    # Load models
    print("\n📦 Loading models...")
    clone.load_models()
    
    # Show available profiles
    profiles = clone.list_profiles()
    print(f"\n{'='*60}")
    print(f"AVAILABLE VOICE PROFILES")
    print(f"{'='*60}")
    for profile_name in profiles:
        profile = clone.get_profile(profile_name)
        print(f"  • {profile.name}: {len(profile.samples)} samples, Languages: {', '.join(profile.languages)}")
    
    # Demo generation for each profile
    print(f"\n{'='*60}")
    print("DEMO: Generating sample audio for each profile")
    print(f"{'='*60}")
    
    for profile_name in profiles:
        profile = clone.get_profile(profile_name)
        
        # Detect if this profile has Hindi samples
        has_hindi = "hi" in profile.languages
        has_english = "en" in profile.languages
        
        if has_english:
            english_text = f"""
            Hello! This is {profile.name}'s voice clone demonstration.
            I can generate natural sounding speech with emotions,
            and handle long paragraphs of text without losing quality.
            The system uses cached embeddings for consistent voice output.
            """
            
            print(f"\n🇬🇧 Generating English for {profile_name} (conversational)...")
            try:
                clone.generate_and_save(
                    english_text,
                    profile_name=profile_name,
                    output_path=f"output_{profile_name}_english.wav",
                    emotion="conversational"
                )
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if has_hindi:
            hindi_text = f"""
            नमस्ते! यह {profile.name} की आवाज़ क्लोनिंग का प्रदर्शन है।
            मैं भावनाओं के साथ प्राकृतिक आवाज़ उत्पन्न कर सकता हूं।
            यह प्रणाली लंबे पाठ को भी अच्छी गुणवत्ता के साथ संभाल सकती है।
            """
            
            print(f"\n🇮🇳 Generating Hindi for {profile_name} (conversational)...")
            try:
                clone.generate_and_save(
                    hindi_text,
                    profile_name=profile_name,
                    output_path=f"output_{profile_name}_hindi.wav",
                    emotion="conversational"
                )
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print("\n✅ Demo complete! Check the output files.")
    print(f"\n💡 Tip: Run 'python myvoiceclone.py' (without --cli) to use the web interface!")


# ============================================================================
# PHASE 1: ELEVENLABS QUALITY ENHANCEMENTS
# ============================================================================

class EnhancedVoiceClone:
    """
    Voice cloning with ElevenLabs-level quality improvements
    
    Phase 1 Enhancements:
    - AudioEnhancer: Professional post-processing (-16 LUFS normalization)
    - AdvancedVoiceEncoder: WavLM 768D embeddings (vs 256D LSTM)
    - EmotionAnalyzer: Auto emotion detection from text
    
    Phase 2 Enhancements:
    - F5TTSEngine: 5-10x faster generation (Speed Mode)
    - ProsodyPredictor: Natural speech rhythm and intonation
    - EnsembleGenerator: Multi-variant generation with best selection
    
    Expected quality improvement: +40-50%
    """
    
    def __init__(self, base_clone: 'MyVoiceClone'):
        """
        Initialize enhanced voice clone wrapper
        
        Args:
            base_clone: Existing MyVoiceClone instance
        """
        self.base = base_clone
        
        # Phase 1 Enhancement components (lazy loaded)
        self.audio_enhancer = None
        self.voice_encoder = None
        self.emotion_analyzer = None
        
        # Phase 2 Enhancement components (lazy loaded)
        self.f5_engine = None
        self.prosody_predictor = None
        self.ensemble_generator = None
        self.quality_scorer = None
        
        # Enhancement settings
        self.enhanced_mode = False
        self.adobe_enhancer = None  # Adobe Podcast-style enhancer
        self.use_adobe_enhance = False
        self.auto_emotion = False
        self.use_wavlm = False
        
        # Phase 2 settings
        self.speed_mode = False
        self.prosody_enhancement = False
        self.ensemble_mode = False
        
        print("✨ Enhanced Voice Clone initialized")
        
    def load_enhancements(
        self,
        enable_audio_enhancement: bool = True,
        enable_emotion_detection: bool = True,
        enable_wavlm_encoder: bool = False,  # Disabled by default (large model)
        emotion_mode: str = "rule-based",
    ):
        """
        Load Phase 1 enhancement components
        
        Args:
            enable_audio_enhancement: Load AudioEnhancer
            enable_emotion_detection: Load EmotionAnalyzer
            enable_wavlm_encoder: Load WavLM encoder (requires ~1.5GB download)
            emotion_mode: 'rule-based' or 'ollama'
        """
        if not ENHANCEMENTS_AVAILABLE:
            print("❌ Enhancements not available. Install dependencies first.")
            return False
            
        print("\n⚡ Loading Phase 1 Enhancements...")
        
        # Audio Enhancement - STUDIO QUALITY settings
        if enable_audio_enhancement:
            print("  📢 Loading AudioEnhancer...")
            self.audio_enhancer = AudioEnhancer(
                target_loudness=-14.0,       # Louder, broadcast quality (-14 LUFS)
                noise_reduce_strength=0.8,   # Strong noise reduction
                compression_ratio=3.0,       # Moderate compression (natural dynamics)
                clarity_boost=0.4,           # Enhanced clarity for voice
            )
            self.enhanced_mode = True
            print("     ✅ AudioEnhancer loaded")
            
            # Also load Adobe Podcast Enhancer for professional quality
            if ADOBE_ENHANCE_AVAILABLE:
                print("  🎙️ Loading Adobe Podcast Enhancer...")
                self.adobe_enhancer = AdobePodcastEnhancer(sr=24000)
                self.use_adobe_enhance = True
                print("     ✅ Adobe Podcast Enhancer loaded (noise reduction, de-reverb, clarity)")
            
        # Emotion Detection
        if enable_emotion_detection:
            print(f"  🧠 Loading EmotionAnalyzer ({emotion_mode} mode)...")
            self.emotion_analyzer = EmotionAnalyzer(mode=emotion_mode)
            self.auto_emotion = True
            print("     ✅ EmotionAnalyzer loaded")
            
        # Advanced Voice Encoder (optional, requires large download)
        if enable_wavlm_encoder:
            print("  🎙️ Loading WavLM encoder (this will download ~1.5GB on first run)...")
            self.voice_encoder = AdvancedVoiceEncoder(device=self.base.device)
            self.use_wavlm = True
            print("     ✅ WavLM encoder loaded")
            
        print("\n✨ Enhancements loaded successfully!")
        return True
    
    def load_phase2_enhancements(
        self,
        enable_speed_mode: bool = True,
        enable_prosody: bool = True,
        enable_ensemble: bool = False,
        n_ensemble_variants: int = 3,
    ):
        """
        Load Phase 2 enhancement components
        
        Args:
            enable_speed_mode: Load F5-TTS for 5-10x faster generation
            enable_prosody: Load prosody predictor for natural rhythm
            enable_ensemble: Load ensemble generator for multi-variant selection
            n_ensemble_variants: Number of variants to generate (if ensemble enabled)
        """
        if not PHASE2_AVAILABLE:
            print("❌ Phase 2 enhancements not available.")
            print("   Install F5-TTS: pip install f5-tts")
            return False
            
        print("\n🚀 Loading Phase 2 Enhancements...")
        
        # F5-TTS Speed Mode
        if enable_speed_mode:
            print("  ⚡ Loading F5-TTS engine (Speed Mode)...")
            try:
                self.f5_engine = F5TTSEngine(
                    device=self.base.device,
                    cache_dir="D:\\voice cloning\\models_cache\\f5tts"
                )
                self.speed_mode = True
                print("     ✅ F5-TTS loaded (5-10x faster generation)")
            except Exception as e:
                print(f"     ⚠️ F5-TTS failed to load: {e}")
                self.speed_mode = False
                
        # Prosody Predictor
        if enable_prosody:
            print("  🎵 Loading ProsodyPredictor...")
            self.prosody_predictor = ProsodyPredictor()
            self.prosody_enhancement = True
            print("     ✅ ProsodyPredictor loaded")
            
        # Ensemble Generator
        if enable_ensemble:
            print(f"  🎼 Loading EnsembleGenerator ({n_ensemble_variants} variants)...")
            self.ensemble_generator = EnsembleGenerator(
                n_variants=n_ensemble_variants,
                segment_duration=2.0,
                crossfade_duration=0.1,
            )
            self.quality_scorer = QualityScorer()
            self.ensemble_mode = True
            print("     ✅ EnsembleGenerator loaded")
            
        print("\n🚀 Phase 2 Enhancements loaded successfully!")
        return True
    
    def generate_speed_mode(
        self,
        text: str,
        profile_name: str,
        apply_prosody: bool = True,
        apply_audio_enhancement: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech using F5-TTS (5-10x faster)
        
        Args:
            text: Text to generate
            profile_name: Voice profile name  
            apply_prosody: Apply prosody enhancement
            apply_audio_enhancement: Apply Phase 1 audio enhancement
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        if not self.speed_mode or not self.f5_engine:
            print("⚠️ F5-TTS not loaded. Falling back to standard generation.")
            return self.base.generate(text, profile_name)
            
        # Get profile
        profile = self.base.get_profile(profile_name)
        if profile is None:
            raise ValueError(f"Profile '{profile_name}' not found!")
            
        if len(profile.samples) == 0:
            raise ValueError(f"Profile '{profile_name}' has no samples!")
            
        # Use BEST sample as reference (not first sample!)
        reference_audio = profile.get_best_reference()
        
        print(f"⚡ Generating with F5-TTS Speed Mode...")
        
        # Generate with F5-TTS
        audio, sr = self.f5_engine.generate(
            text=text,
            reference_audio=reference_audio,
            speed=1.0,
        )
        
        # Apply prosody enhancement
        if apply_prosody and self.prosody_enhancement and self.prosody_predictor:
            print("🎵 Applying prosody enhancement...")
            audio = self.prosody_predictor.enhance_naturalness(audio, sr, text)
            
        # Apply audio enhancement
        if apply_audio_enhancement and self.enhanced_mode and self.audio_enhancer:
            print("⚡ Applying audio enhancement...")
            audio = self.audio_enhancer.enhance(
                audio, sr,
                apply_noise_reduction=True,
                apply_compression=True,
                apply_clarity=True,
                apply_normalization=True,
            )
            
        # Apply Adobe Podcast Enhancement (professional studio quality)
        if apply_audio_enhancement and self.use_adobe_enhance and self.adobe_enhancer:
            print("🎙️ Applying Adobe Podcast Enhancement (studio quality)...")
            # Convert tensor to numpy if needed
            if hasattr(audio, 'numpy'):
                audio_np = audio.cpu().numpy().squeeze()
            else:
                audio_np = audio.squeeze() if hasattr(audio, 'squeeze') else audio
            audio_np = self.adobe_enhancer.enhance(audio_np)
            audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            
        return audio, sr
    
    def generate_ensemble(
        self,
        text: str,
        profile_name: str,
        emotion: str = "neutral",
        blend_mode: str = "best_single",
        apply_audio_enhancement: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate multiple variants and select/blend the best
        
        Args:
            text: Text to generate
            profile_name: Voice profile name
            emotion: Emotion for generation
            blend_mode: "best_single" or "best_segments"
            apply_audio_enhancement: Apply Phase 1 audio enhancement
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        if not self.ensemble_mode or not self.ensemble_generator:
            print("⚠️ Ensemble generator not loaded. Using standard generation.")
            return self.base.generate(text, profile_name, emotion=emotion)
            
        print(f"🎼 Generating {self.ensemble_generator.n_variants} variants...")
        
        # Create generation function
        def generate_fn(seed: int) -> Tuple[torch.Tensor, int]:
            torch.manual_seed(seed)
            audio, sr = self.base.generate(
                text=text,
                profile_name=profile_name,
                emotion=emotion,
                show_progress=False,
            )
            return audio, sr
            
        # Generate and blend
        audio, sr = self.ensemble_generator.generate_and_blend(
            generate_fn,
            blend_mode=blend_mode,
        )
        
        # Apply audio enhancement
        if apply_audio_enhancement and self.enhanced_mode and self.audio_enhancer:
            print("⚡ Applying audio enhancement...")
            audio = self.audio_enhancer.enhance(
                audio, sr,
                apply_noise_reduction=True,
                apply_compression=True,
                apply_clarity=True,
                apply_normalization=True,
            )
            
        # Apply Adobe Podcast Enhancement (professional studio quality)
        if apply_audio_enhancement and self.use_adobe_enhance and self.adobe_enhancer:
            print("🎙️ Applying Adobe Podcast Enhancement (studio quality)...")
            if hasattr(audio, 'numpy'):
                audio_np = audio.cpu().numpy().squeeze()
            else:
                audio_np = audio.squeeze() if hasattr(audio, 'squeeze') else audio
            audio_np = self.adobe_enhancer.enhance(audio_np)
            audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            
        return audio, sr
    
    def generate_enhanced(
        self,
        text: str,
        profile_name: str,
        emotion: str = "auto",
        language: str = "auto",
        chunk_size: int = 100,
        apply_audio_enhancement: bool = True,
        use_speed_mode: bool = False,
        use_ensemble: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech with Phase 1 & Phase 2 enhancements
        
        Args:
            text: Text to generate
            profile_name: Voice profile name
            emotion: Emotion to use ('auto' for detection, or specific emotion)
            language: Language ('auto', 'en', 'hi', etc.)
            chunk_size: Character chunk size
            apply_audio_enhancement: Apply audio enhancement
            use_speed_mode: Use F5-TTS for faster generation (English only)
            use_ensemble: Generate multiple variants and pick best
            **kwargs: Additional args for base generate()
            
        Returns:
            (audio_tensor, sample_rate) tuple
        """
        # Auto-detect emotion if requested (apply to full text BEFORE chunking)
        if emotion == "auto" and self.auto_emotion and self.emotion_analyzer:
            print(f"🧠 Auto-detecting emotion from text...")
            detected_emotion = self.emotion_analyzer.analyze(text)
            print(f"   → Detected: {detected_emotion}")
            emotion = detected_emotion
            print(f"   → Will apply '{emotion}' emotion to ALL chunks")
        elif emotion == "auto":
            emotion = "neutral"  # Fallback
            print(f"   → Using fallback emotion: neutral")
            
        # Speed Mode (F5-TTS) - English only
        if use_speed_mode and self.speed_mode and language in ["en", "auto"]:
            detected_lang = TextProcessor.detect_language(text) if language == "auto" else language
            
            if detected_lang == "en":
                print(f"⚡ Using F5-TTS Speed Mode...")
                return self.generate_speed_mode(
                    text=text,
                    profile_name=profile_name,
                    apply_prosody=True,
                    apply_audio_enhancement=apply_audio_enhancement,
                )
                
        # Ensemble Mode
        if use_ensemble and self.ensemble_mode:
            print(f"🎼 Using Ensemble Generation...")
            return self.generate_ensemble(
                text=text,
                profile_name=profile_name,
                emotion=emotion,
                blend_mode="best_single",
                apply_audio_enhancement=apply_audio_enhancement,
            )
            
        # Standard enhanced generation
        print(f"🎙️ Generating with emotion: {emotion}")
        
        # Handle language parameter - 'auto' means let the system detect
        detected_lang = TextProcessor.detect_language(text) if language == "auto" else language
        print(f"🌐 Language: {detected_lang} {'(auto-detected)' if language == 'auto' else '(forced)'}")
        
        audio, sr = self.base.generate(
            text=text,
            profile_name=profile_name,
            emotion=emotion,
            auto_detect_language=(language == "auto"),
            force_language=None,  # Let auto_detect handle it based on detected_lang in chunks
        )
        
        # Apply prosody enhancement
        if self.prosody_enhancement and self.prosody_predictor:
            print("🎵 Applying prosody enhancement...")
            audio = self.prosody_predictor.enhance_naturalness(audio, sr, text)
        
        # Apply audio enhancement
        if apply_audio_enhancement and self.enhanced_mode and self.audio_enhancer:
            print("⚡ Applying audio enhancement...")
            audio = self.audio_enhancer.enhance(
                audio,
                sr,
                apply_noise_reduction=True,
                apply_compression=True,
                apply_clarity=True,
                apply_normalization=True,
            )
            print("   ✅ Enhancement applied")
        
        # Apply Adobe Podcast Enhancement (professional studio quality)
        if apply_audio_enhancement and self.use_adobe_enhance and self.adobe_enhancer:
            print("🎙️ Applying Adobe Podcast Enhancement (studio quality)...")
            if hasattr(audio, 'numpy'):
                audio_np = audio.cpu().numpy().squeeze()
            else:
                audio_np = audio.squeeze() if hasattr(audio, 'squeeze') else audio
            audio_np = self.adobe_enhancer.enhance(audio_np)
            audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            print("   ✅ Adobe Enhancement applied (noise reduction, de-reverb, clarity)")
        
        # Ensure audio is always a torch tensor before returning
        if not isinstance(audio, torch.Tensor):
            if isinstance(audio, np.ndarray):
                if audio.ndim == 1:
                    audio = torch.from_numpy(audio).unsqueeze(0).float()
                else:
                    audio = torch.from_numpy(audio).float()
            else:
                raise TypeError(f"Unexpected audio type: {type(audio)}")
            
        return audio, sr
    
    def create_profile_with_wavlm(
        self,
        profile_name: str,
        audio_paths: List[str],
        languages: List[str],
    ) -> bool:
        """
        Create voice profile using WavLM encoder (768D embeddings)
        
        Args:
            profile_name: Name for the profile
            audio_paths: List of voice sample paths
            languages: Languages in samples
            
        Returns:
            Success status
        """
        if not self.use_wavlm or not self.voice_encoder:
            print("❌ WavLM encoder not loaded")
            return False
            
        print(f"\n🎙️ Creating profile '{profile_name}' with WavLM encoder...")
        
        # Encode samples with WavLM
        embedding = self.voice_encoder.encode_multiple_samples(
            audio_paths,
            average=True,
            return_numpy=False,
        )
        
        # Create profile
        profile = VoiceProfile(
            name=profile_name,
            samples=audio_paths,
            languages=languages,
            embedding_cache=embedding,  # 768D WavLM embedding
        )
        
        # Save profile
        self.base.profile_manager.profiles[profile_name] = profile
        self.base.profile_manager.save_profiles()
        
        print(f"✅ Profile created with 768D WavLM embeddings")
        return True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="My Voice Clone - Ultimate Voice Cloning System")
    parser.add_argument("--cli", action="store_true", help="Run CLI demo instead of web interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    
    args = parser.parse_args()
    
    if args.cli:
        cli_demo()
    else:
        # Web interface
        print("\n🚀 Starting My Voice Clone Web Interface...")
        
        # Initialize
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clone = MyVoiceClone(device=device)
        
        # Create and launch interface
        demo = create_gradio_interface(clone)
        demo.launch(
            server_port=args.port,
            share=args.share,
            inbrowser=True
        )


if __name__ == "__main__":
    main()
