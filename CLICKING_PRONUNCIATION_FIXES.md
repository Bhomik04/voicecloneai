# Clicking Sound & Pronunciation Fixes Applied

## Summary of Changes

I've applied **4 major fixes** to eliminate clicking sounds and improve pronunciation:

---

## 🔧 Fix 1: Lowered cfg_weight for Better Pronunciation

**Problem**: cfg_weight was 0.4-0.7, causing syllable splitting (e.g., "ho-nest-ly")

**Solution**: Reduced all emotion presets to 0.3-0.4:
- `neutral`: 0.6 → **0.3**
- `excited`: 0.5 → **0.35**
- `calm`: 0.7 → **0.4**
- `dramatic`: 0.4 → **0.35**
- `conversational`: 0.5 → **0.3**
- `storytelling`: 0.5 → **0.35**

**Why**: Lower cfg_weight gives the model more freedom for natural pronunciation while still matching your voice.

---

## 🔧 Fix 2: Added 100ms Silence Padding Between Chunks

**Problem**: Audio chunks were being crossfaded immediately, causing overlap clicks

**Solution**: Insert 100ms of silence between each chunk BEFORE crossfading

**Code location**: Line ~1417 in `myvoiceclone.py`

```python
# Add 100ms silence padding to prevent clicks
silence_ms = 100
silence_samples = int(sample_rate * silence_ms / 1000)
silence = torch.zeros((1, silence_samples), dtype=wav.dtype, device=wav.device)
audio_segments.append(silence)
```

**Why**: Creates breathing room between chunks, preventing audio overlap that causes clicks.

---

## 🔧 Fix 3: High-Pass Filter at 80Hz

**Problem**: Low-frequency clicking artifacts from chunk stitching

**Solution**: Applied high-pass filter to remove frequencies below 80Hz (removes clicks without affecting voice)

**Code location**: Line ~1437 in `myvoiceclone.py`

```python
# High-pass filter at 80Hz to remove clicks/pops without affecting voice
full_audio = F.highpass_biquad(full_audio, sample_rate, cutoff_freq=80)
```

**Why**: Most clicking sounds are in the low-frequency range (20-100Hz), while human voice is 85Hz-255Hz. The 80Hz cutoff removes clicks without affecting voice quality.

---

## 🔧 Fix 4: Skip Chunking for Short Texts

**Problem**: Even short texts were being chunked, creating unnecessary stitching points

**Solution**: For texts under 200 characters OR less than 3 sentences, generate as single chunk

**Code location**: Line ~1365 in `myvoiceclone.py`

```python
if text_length < 200 or sentence_count < 3:
    # Generate as single chunk - no stitching = no clicks!
    chunks = [(text.strip(), TextProcessor.detect_language(text))]
```

**Why**: No chunking = no stitching = zero chance of clicks for short texts!

---

## ✅ Combined Effect

With all 4 fixes together:

1. **Silence padding** creates space between chunks
2. **300ms crossfade** smoothly blends the transition
3. **High-pass filter** removes residual clicking artifacts
4. **Single-chunk mode** eliminates stitching for short texts
5. **Lower cfg_weight** improves pronunciation clarity

---

## 🎯 How to Test

1. **IMPORTANT**: Close any running voice clone scripts
2. Restart your Python script
3. Try generating:
   - Short text (< 200 chars) → Should use single-chunk mode (no clicks!)
   - Long text with multiple sentences → Uses all 4 fixes

**Test sentences**:
```
Short: "This is a quick test to check pronunciation honestly."
Long: "This is a longer test. I want to check if honestly the clicking sounds are completely gone now. The pronunciation should also be much better with these new settings applied."
```

---

## 📊 What's Already Working

✅ **Studio Audio Quality**: 48kHz upsampling, -14 LUFS normalization, broadcast mastering  
✅ **300ms Crossfade**: Smooth cosine fading with DC offset removal  
✅ **Platform Presets**: Instagram/TikTok/YouTube optimized output

---

## ⚠️ Important Notes

### About Pronunciation
- ChatterBox TTS tokenizer has inherent limitations
- Some words may still split syllables occasionally
- Lower cfg_weight (0.3-0.4) helps but doesn't guarantee 100% perfection
- If pronunciation is critical, you may need to:
  - Use phonetic spelling (e.g., "onestly" instead of "honestly")
  - Try different emotion presets
  - Split problematic words into multiple generations

### About Clicking
- With all 4 fixes, clicking should be **drastically reduced** or **eliminated**
- If you still hear clicks:
  1. Make sure you restarted the script (old code may still be loaded)
  2. Try increasing silence padding to 150ms (line ~1419)
  3. Check if clicks occur at specific words (might be model artifact, not stitching)

---

## 🔍 Technical Details

**Crossfade Duration**: 300ms (increased from 100ms → 200ms → 300ms)  
**Silence Padding**: 100ms between chunks  
**High-Pass Filter**: 80Hz cutoff (Biquad filter)  
**cfg_weight Range**: 0.3-0.4 (optimal for natural pronunciation)  
**Single-Chunk Threshold**: < 200 characters OR < 3 sentences  

---

## 📝 Files Modified

1. **myvoiceclone.py**:
   - Lines 205-240: Emotion presets (cfg_weight lowered)
   - Lines 1365-1376: Single-chunk logic for short texts
   - Lines 1417-1428: Silence padding between chunks
   - Lines 1437-1445: High-pass filter application

2. **studio_audio_processor.py**: (Already working - no changes needed)

---

## 🚀 Next Steps

1. **Close all Python windows/terminals**
2. **Restart your voice cloning script**
3. **Test with both short and long text**
4. **Listen for clicking and pronunciation**
5. **Report results**

If issues persist after restart, we can:
- Increase silence padding to 150-200ms
- Adjust high-pass filter cutoff
- Use phonetic spelling for problematic words
- Consider alternative TTS models with better pronunciation

---

**Generated**: After applying all 4 fixes to eliminate clicking and improve pronunciation  
**Status**: ✅ Ready for testing - Please restart script and test!
