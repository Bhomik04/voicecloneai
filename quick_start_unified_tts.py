"""
🎙️ Quick Start - VibeVoice + Chatterbox Unified TTS
====================================================

This script provides easy-to-use functions for voice cloning
using both VibeVoice (Hindi-optimized) and Chatterbox (English-optimized).

Usage:
    python quick_start_unified_tts.py
    
Or import in your code:
    from quick_start_unified_tts import speak_hindi, speak_english, speak
"""

# Configure model paths first
import model_paths

import os
from pathlib import Path


def check_setup() -> dict:
    """Check if everything is set up correctly"""
    status = {
        "profiles_dir": False,
        "chatterbox": False,
        "vibevoice_model": False,
        "vibevoice_installed": False,
    }
    
    # Check profiles directory
    profiles_dir = Path("D:/voice cloning/voice_profiles")
    if profiles_dir.exists() and any(profiles_dir.iterdir()):
        status["profiles_dir"] = True
        profiles = [p.name for p in profiles_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        print(f"✅ Voice Profiles: {profiles}")
    else:
        print("❌ No voice profiles found")
    
    # Check Chatterbox
    try:
        import sys
        sys.path.insert(0, "D:/voice cloning/chatterbox/src")
        from chatterbox.tts import ChatterboxTTS
        status["chatterbox"] = True
        print("✅ Chatterbox: Available")
    except ImportError as e:
        print(f"❌ Chatterbox: Not available ({e})")
    
    # Check VibeVoice installation
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        status["vibevoice_installed"] = True
        print("✅ VibeVoice: Installed")
    except ImportError:
        print("❌ VibeVoice: Not installed")
    
    # Check VibeVoice model
    model_path = Path("D:/voice cloning/models_cache/vibevoice-hindi-7b")
    if model_path.exists():
        safetensors = list(model_path.glob("*.safetensors"))
        if len(safetensors) >= 8:  # 7B model has 8 safetensor files
            status["vibevoice_model"] = True
            print(f"✅ VibeVoice Hindi-7B Model: Downloaded ({len(safetensors)} files)")
        else:
            print(f"⏳ VibeVoice Hindi-7B Model: Downloading ({len(safetensors)}/8 files)")
    else:
        print("❌ VibeVoice Hindi-7B Model: Not downloaded")
    
    return status


def speak(
    text: str,
    profile: str = "pritam",
    emotion: str = "conversational",
    engine: str = None,  # "auto", "chatterbox", or "vibevoice"
    save_path: str = None,
):
    """
    Generate speech from text using the best available engine
    
    Args:
        text: Text to speak (Hindi, English, or mixed)
        profile: Voice profile name
        emotion: neutral, excited, calm, dramatic, conversational, storytelling
        engine: Force specific engine or None for auto-select
        save_path: Optional path to save audio file
        
    Returns:
        GenerationResult or None
    """
    from tts_compatibility import UnifiedTTS
    
    tts = UnifiedTTS()
    return tts.generate(
        text=text,
        profile=profile,
        emotion=emotion,
        engine=engine,
        output_path=save_path,
    )


def speak_hindi(
    text: str,
    profile: str = "pritam",
    emotion: str = "conversational",
    save_path: str = None,
):
    """
    Generate Hindi speech using VibeVoice
    
    Args:
        text: Hindi text to speak
        profile: Voice profile name
        emotion: Target emotion
        save_path: Optional path to save audio
        
    Returns:
        GenerationResult or None
    """
    return speak(text, profile, emotion, engine="vibevoice", save_path=save_path)


def speak_english(
    text: str,
    profile: str = "pritam",
    emotion: str = "conversational",
    save_path: str = None,
):
    """
    Generate English speech using Chatterbox
    
    Args:
        text: English text to speak
        profile: Voice profile name
        emotion: Target emotion
        save_path: Optional path to save audio
        
    Returns:
        GenerationResult or None
    """
    return speak(text, profile, emotion, engine="chatterbox", save_path=save_path)


def demo():
    """Run a quick demo"""
    print("\n" + "="*60)
    print("🎙️ UNIFIED TTS QUICK START DEMO")
    print("="*60)
    
    # Check setup
    print("\n📋 Checking setup...")
    status = check_setup()
    
    if not any(status.values()):
        print("\n❌ Setup incomplete. Please ensure:")
        print("   1. Voice profiles exist in D:/voice cloning/voice_profiles/")
        print("   2. Run: pip install -e vibevoice")
        print("   3. Download VibeVoice model (in progress)")
        return
    
    # Get available profile
    from tts_compatibility import VoiceProfileManager
    profiles = VoiceProfileManager()
    profile_list = profiles.list_profiles()
    
    if not profile_list:
        print("\n❌ No voice profiles found")
        return
    
    profile = profile_list[0]
    print(f"\n🎤 Using profile: {profile}")
    
    # Demo based on what's available
    output_dir = Path("D:/voice cloning/audio_output")
    output_dir.mkdir(exist_ok=True)
    
    if status["chatterbox"]:
        print("\n--- English Demo (Chatterbox) ---")
        result = speak_english(
            text="Hello! This is a demo of the unified text-to-speech system.",
            profile=profile,
            emotion="conversational",
            save_path=str(output_dir / "demo_english.wav")
        )
        if result:
            print(f"✅ Saved: demo_english.wav ({result.duration:.2f}s)")
    
    if status["vibevoice_model"] and status["vibevoice_installed"]:
        print("\n--- Hindi Demo (VibeVoice) ---")
        result = speak_hindi(
            text="नमस्ते! यह एक टेक्स्ट-टू-स्पीच सिस्टम का डेमो है।",
            profile=profile,
            emotion="conversational",
            save_path=str(output_dir / "demo_hindi.wav")
        )
        if result:
            print(f"✅ Saved: demo_hindi.wav ({result.duration:.2f}s)")
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)


# Example usage patterns
EXAMPLES = """
# ============================================================================
# EXAMPLE USAGE
# ============================================================================

from quick_start_unified_tts import speak, speak_hindi, speak_english

# 1. Auto-detect language and select best engine
speak("Hello world!", profile="pritam")

# 2. Force Hindi with VibeVoice
speak_hindi("नमस्ते, कैसे हो?", profile="pritam")

# 3. Force English with Chatterbox  
speak_english("Good morning!", profile="pritam")

# 4. With emotion and save to file
speak(
    text="This is exciting news!",
    profile="pritam",
    emotion="excited",
    save_path="output.wav"
)

# 5. Mix Hindi and English (Hinglish)
speak(
    text="Hello, mera naam Pritam hai. Main India se hoon.",
    profile="pritam",
    emotion="conversational"
)

# 6. Access result data
result = speak("Test", profile="pritam")
if result:
    print(f"Duration: {result.duration}s")
    print(f"Engine: {result.engine_used}")
    print(f"RTF: {result.rtf}x")
    result.save("my_audio.wav")
"""


if __name__ == "__main__":
    print(EXAMPLES)
    demo()
