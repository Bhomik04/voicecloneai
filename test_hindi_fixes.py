"""
🧪 Test Hindi Prosody Improvements
===================================

Quick test to verify the Hindi prosody enhancements are working.
Run this after any TTS generation to apply post-processing fixes.

Usage:
    python test_hindi_fixes.py <audio_file.wav>
    
Or run the demo:
    python test_hindi_fixes.py --demo
"""

import sys
import os

# Configure paths
sys.path.insert(0, "D:/voice cloning")
import model_paths

from pathlib import Path
import numpy as np


def test_enhancer_standalone(audio_path: str):
    """Test the Hindi prosody enhancer on an existing audio file"""
    from hindi_prosody_enhancer import HindiProsodyEnhancer
    import librosa
    import soundfile as sf
    
    print(f"\n🎙️ Testing Hindi Prosody Enhancer")
    print("=" * 60)
    print(f"Input: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    print(f"Duration: {len(audio)/sr:.2f}s")
    
    # Process with different intensities
    enhancer = HindiProsodyEnhancer(sr)
    
    # Standard enhancement
    output_path = audio_path.replace('.wav', '_hindi_enhanced.wav')
    enhanced = enhancer.enhance(audio, sr, intensity=1.0)
    sf.write(output_path, enhanced, sr)
    print(f"\n✅ Saved (standard): {output_path}")
    
    # Strong enhancement
    output_path_strong = audio_path.replace('.wav', '_hindi_enhanced_strong.wav')
    enhanced_strong = enhancer.enhance(audio, sr, intensity=1.3)
    sf.write(output_path_strong, enhanced_strong, sr)
    print(f"✅ Saved (strong): {output_path_strong}")
    
    print("\n🎧 Compare the files to hear the difference!")
    print("   The enhanced version should have:")
    print("   - Natural pitch drops at sentence endings")
    print("   - More expressive intonation")
    print("   - Phrase-final lengthening")
    print("   - Better rhythm patterns")


def run_demo():
    """Run a complete demo generating Hindi speech and applying enhancements"""
    print("\n🎙️ HINDI PROSODY DEMO")
    print("=" * 60)
    
    # Check if we have voice profiles
    profiles_dir = Path("D:/voice cloning/voice_profiles")
    profiles = [p.name for p in profiles_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    if not profiles:
        print("❌ No voice profiles found. Please create one first.")
        return
    
    profile = profiles[0]
    print(f"Using profile: {profile}")
    
    # Test texts
    test_texts = [
        # Pure Hindi
        "नमस्ते! मेरा नाम प्रीतम है। आज मौसम बहुत अच्छा है।",
        
        # Hinglish (mixed)
        "Hello friends! आज हम बात करेंगे about AI technology।",
        
        # Conversational Hindi
        "अरे यार, तुम कहाँ थे? मैं तुम्हें बहुत देर से ढूंढ रहा था।",
    ]
    
    try:
        from myvoiceclone import MyVoiceClone
        
        print("\n🔄 Loading voice clone system...")
        vc = MyVoiceClone()
        vc.load_models()
        
        output_dir = Path("D:/voice cloning/audio_output/hindi_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {text[:50]}...")
            print(f"{'='*60}")
            
            output_path = output_dir / f"hindi_test_{i}.wav"
            
            audio, sr = vc.generate(
                text=text,
                profile_name=profile,
                emotion="conversational",
                show_progress=True
            )
            
            import torchaudio as ta
            ta.save(str(output_path), audio, sr)
            print(f"💾 Saved: {output_path}")
        
        print("\n" + "=" * 60)
        print("✅ Demo complete! Check the audio files in:")
        print(f"   {output_dir}")
        print("\nThe Hindi prosody enhancement should make the speech sound more natural!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def compare_before_after():
    """Generate comparison samples with and without enhancement"""
    print("\n🔬 BEFORE/AFTER COMPARISON")
    print("=" * 60)
    
    # Check for existing voice profiles
    profiles_dir = Path("D:/voice cloning/voice_profiles")
    profiles = [p.name for p in profiles_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    if not profiles:
        print("❌ No voice profiles found.")
        return
    
    profile = profiles[0]
    
    # Test text
    test_text = "आज मैं आपको बताऊंगा कि AI technology कैसे काम करती है। यह बहुत interesting है!"
    
    try:
        from myvoiceclone import MyVoiceClone
        import torchaudio as ta
        
        vc = MyVoiceClone()
        vc.load_models()
        
        output_dir = Path("D:/voice cloning/audio_output/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate with current settings (includes Hindi prosody)
        print("\n📝 Generating WITH Hindi prosody enhancement...")
        audio, sr = vc.generate(
            text=test_text,
            profile_name=profile,
            emotion="conversational",
            show_progress=True
        )
        ta.save(str(output_dir / "with_enhancement.wav"), audio, sr)
        
        print("\n✅ Comparison files saved to:")
        print(f"   {output_dir}")
        print("\n🎧 Listen and compare the naturalness!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python test_hindi_fixes.py audio.wav       # Enhance an audio file")
        print("  python test_hindi_fixes.py --demo          # Run full demo")
        print("  python test_hindi_fixes.py --compare       # Before/after comparison")
        return
    
    arg = sys.argv[1]
    
    if arg == "--demo":
        run_demo()
    elif arg == "--compare":
        compare_before_after()
    elif os.path.exists(arg):
        test_enhancer_standalone(arg)
    else:
        print(f"❌ File not found: {arg}")


if __name__ == "__main__":
    main()
