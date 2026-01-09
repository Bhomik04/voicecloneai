"""Test VibeVoice integration with unified TTS system"""
import os
os.environ['MODEL_CACHE_ROOT'] = 'D:/voice cloning/models_cache'

from tts_compatibility import UnifiedTTS, TTSConfig, TTSEngine

def test_english():
    """Test English generation with Chatterbox"""
    print("\n" + "="*60)
    print("TEST 1: English with Chatterbox")
    print("="*60)
    
    config = TTSConfig(default_engine=TTSEngine.CHATTERBOX)
    tts = UnifiedTTS(config=config)
    result = tts.generate(
        text="Hello! This is a test of the English voice cloning system.",
        profile="pritam",
        emotion="conversational"
    )
    
    if result:
        print(f"✅ Generated {result.duration:.2f}s audio")
        print(f"   Engine: {result.engine_used}")
        print(f"   RTF: {result.rtf:.2f}x")
        result.save("test_english_chatterbox.wav")
        print(f"   Saved: test_english_chatterbox.wav")
    else:
        print("❌ Generation failed")

def test_hindi_chatterbox():
    """Test Hindi generation with Chatterbox (with prosody fixes)"""
    print("\n" + "="*60)
    print("TEST 2: Hindi with Chatterbox (Enhanced)")
    print("="*60)
    
    config = TTSConfig(default_engine=TTSEngine.CHATTERBOX)
    tts = UnifiedTTS(config=config)
    result = tts.generate(
        text="नमस्ते, मेरा नाम प्रितम है। आप कैसे हैं?",
        profile="pritam",
        emotion="conversational",
        language="hi-IN"
    )
    
    if result:
        print(f"✅ Generated {result.duration:.2f}s audio")
        print(f"   Engine: {result.engine_used}")
        print(f"   RTF: {result.rtf:.2f}x")
        result.save("test_hindi_chatterbox.wav")
        print(f"   Saved: test_hindi_chatterbox.wav")
    else:
        print("❌ Generation failed")

def test_hinglish():
    """Test Hinglish (mixed) generation"""
    print("\n" + "="*60)
    print("TEST 3: Hinglish with Chatterbox")
    print("="*60)
    
    config = TTSConfig(default_engine=TTSEngine.CHATTERBOX)
    tts = UnifiedTTS(config=config)
    result = tts.generate(
        text="Hello, मेरा naam Pritam है। Main India se hoon.",
        profile="pritam",
        emotion="conversational",
        language="hinglish"
    )
    
    if result:
        print(f"✅ Generated {result.duration:.2f}s audio")
        print(f"   Engine: {result.engine_used}")
        print(f"   RTF: {result.rtf:.2f}x")
        result.save("test_hinglish_chatterbox.wav")
        print(f"   Saved: test_hinglish_chatterbox.wav")
    else:
        print("❌ Generation failed")

def test_hindi_vibevoice():
    """Test Hindi generation with VibeVoice"""
    print("\n" + "="*60)
    print("TEST 4: Hindi with VibeVoice-Hindi-7B")
    print("="*60)
    
    config = TTSConfig(default_engine=TTSEngine.VIBEVOICE)
    tts = UnifiedTTS(config=config)
    result = tts.generate(
        text="नमस्ते, मेरा नाम प्रितम है। आप कैसे हैं?",
        profile="pritam",
        emotion="conversational",
        language="hi-IN"
    )
    
    if result:
        print(f"✅ Generated {result.duration:.2f}s audio")
        print(f"   Engine: {result.engine_used}")
        print(f"   RTF: {result.rtf:.2f}x")
        result.save("test_hindi_vibevoice.wav")
        print(f"   Saved: test_hindi_vibevoice.wav")
    else:
        print("❌ Generation failed")

def test_auto_selection():
    """Test automatic engine selection"""
    print("\n" + "="*60)
    print("TEST 5: Auto-selection (Hindi should pick VibeVoice)")
    print("="*60)
    
    config = TTSConfig(default_engine=TTSEngine.AUTO)
    tts = UnifiedTTS(config=config)
    
    # Should auto-select VibeVoice for Hindi
    result = tts.generate(
        text="यह हिंदी में एक परीक्षण है।",
        profile="pritam",
        emotion="conversational"
    )
    
    if result:
        print(f"✅ Generated {result.duration:.2f}s audio")
        print(f"   Auto-selected engine: {result.engine_used}")
        print(f"   RTF: {result.rtf:.2f}x")
        result.save("test_auto_hindi.wav")
        print(f"   Saved: test_auto_hindi.wav")
    else:
        print("❌ Generation failed")

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 VIBEVOICE INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: English with Chatterbox (baseline)
        test_english()
        
        # Test 2: Hindi with Chatterbox (with prosody enhancements)
        test_hindi_chatterbox()
        
        # Test 3: Hinglish
        test_hinglish()
        
        # Test 4: Hindi with VibeVoice (native Hindi model)
        test_hindi_vibevoice()
        
        # Test 5: Auto-selection
        test_auto_selection()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print("\n📂 Output files:")
        print("   - test_english_chatterbox.wav (Chatterbox English)")
        print("   - test_hindi_chatterbox.wav (Chatterbox Hindi + prosody fixes)")
        print("   - test_hinglish_chatterbox.wav (Chatterbox Hinglish)")
        print("   - test_hindi_vibevoice.wav (VibeVoice-Hindi-7B)")
        print("   - test_auto_hindi.wav (Auto-selected engine)")
        print("\n💡 Compare the Hindi files to evaluate quality differences!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
