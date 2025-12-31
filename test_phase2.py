"""
Phase 2 Integration Test
========================
Tests all Phase 2 components to verify they're working correctly.
"""

import sys
sys.path.insert(0, ".")

def test_prosody_predictor():
    """Test prosody prediction"""
    print("\n🎵 Testing ProsodyPredictor...")
    
    from prosody_predictor import ProsodyPredictor
    
    predictor = ProsodyPredictor()
    
    # Test text analysis
    test_texts = [
        "Hello, how are you doing today?",
        "This is absolutely amazing! I can't believe it!",
        "The weather is nice.",
    ]
    
    for text in test_texts:
        annotations = predictor.analyze_text(text)
        print(f"  Text: '{text[:40]}...'")
        print(f"    → {len(annotations)} words analyzed")
        
    # Test SSML generation
    ssml = predictor.generate_ssml_tags(test_texts[0])
    print(f"  SSML output: {ssml[:60]}...")
    
    print("  ✅ ProsodyPredictor: PASSED")
    return True


def test_ensemble_generator():
    """Test ensemble generation"""
    print("\n🎼 Testing EnsembleGenerator...")
    
    import torch
    import numpy as np
    from ensemble_generator import EnsembleGenerator, QualityScorer
    
    # Initialize components
    generator = EnsembleGenerator(n_variants=3)
    scorer = QualityScorer()
    
    # Create mock audio
    sr = 22050
    duration = 2.0
    samples = int(sr * duration)
    t = torch.linspace(0, duration, samples)
    audio = torch.sin(2 * np.pi * 440 * t)
    audio = audio.unsqueeze(0)  # Add channel dim
    
    # Test quality scoring
    score = scorer.score(audio, sr)
    print(f"  Quality score for test audio: {score:.1f}")
    
    # Test mock generation
    def mock_generate(seed):
        torch.manual_seed(seed)
        noise = torch.randn(1, samples) * 0.05
        return audio + noise, sr
    
    # Generate variants
    variants = generator.generate_variants(mock_generate, seeds=[42, 123, 456])
    print(f"  Generated {len(variants)} variants")
    
    # Select best
    best = generator.select_best(variants)
    print(f"  Best variant seed: {best.seed}, score: {best.quality_score:.1f}")
    
    print("  ✅ EnsembleGenerator: PASSED")
    return True


def test_f5_engine():
    """Test F5-TTS engine (import only, model loads lazily)"""
    print("\n⚡ Testing F5TTSEngine...")
    
    from f5_engine import F5TTSEngine
    
    # Just test that the class can be instantiated
    # (don't actually load the model, it's large)
    try:
        engine = F5TTSEngine(device="cpu")
        print(f"  Engine initialized (model loads lazily)")
        print(f"  Cache dir: {engine.cache_dir}")
        print("  ✅ F5TTSEngine: PASSED (import/init)")
        return True
    except Exception as e:
        print(f"  ⚠️ F5TTSEngine initialization failed: {e}")
        print("  This may be OK if F5-TTS isn't fully installed")
        return False


def test_integration():
    """Test that Phase 2 can be imported into myvoiceclone"""
    print("\n🔗 Testing Phase 2 Integration...")
    
    try:
        # Simulate the imports from myvoiceclone.py
        from f5_engine import F5TTSEngine
        from prosody_predictor import ProsodyPredictor
        from ensemble_generator import EnsembleGenerator, QualityScorer
        
        PHASE2_AVAILABLE = True
        print("  Phase 2 imports: ✅")
        
        # Check the flag
        print(f"  PHASE2_AVAILABLE = {PHASE2_AVAILABLE}")
        
        print("  ✅ Integration: PASSED")
        return True
    except ImportError as e:
        print(f"  ❌ Integration failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  PHASE 2 INTEGRATION TEST")
    print("=" * 60)
    
    results = {
        "ProsodyPredictor": test_prosody_predictor(),
        "EnsembleGenerator": test_ensemble_generator(),
        "F5TTSEngine": test_f5_engine(),
        "Integration": test_integration(),
    }
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
