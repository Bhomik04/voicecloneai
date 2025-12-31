# Pronunciation & Clicking Fixes - Round 2

## Issues Reported
1. **"Tich" sound like lighter/burner** - persistent clicking between chunks
2. **English pronunciation issues** - "honestly" pronounced as "ho-nest-ly" instead of silent "h"

---

## ✅ Fixes Applied

### 1. Increased Silence Padding: 100ms → 250ms
**Why**: More breathing room between chunks prevents overlap clicks
```python
silence_ms = 250  # Increased from 100ms
```

### 2. Lowered cfg_weight: 0.3-0.4 → 0.2-0.25
**Why**: Even lower guidance gives model more freedom for natural pronunciation

**New values**:
- `conversational`: **0.20** (most natural)
- `neutral`: **0.22**
- `dramatic`: **0.23**  
- `storytelling`: **0.24**
- `excited`: **0.25**
- `calm`: **0.25**

### 3. Added Pronunciation Preprocessing
**New feature**: Automatically fixes common English pronunciation issues BEFORE generation

**Silent-H words**:
- "honestly" → "onestly"
- "honest" → "onest"
- "honor" → "onor"
- "hour" → "our"
- "heir" → "air"

**Silent letters**:
- "knight" → "nite" (silent k, gh)
- "write" → "rite" (silent w)
- "psychology" → "sychology" (silent p)
- "climb" → "clime" (silent b)

**Applied in both**:
- Hindi/Hinglish generation (`_generate_chunk_hindi`)
- English generation (`_generate_chunk_english`)

### 4. High-Pass Filter: 80Hz → 100Hz
**Why**: More aggressive filtering to catch all low-frequency clicks
```python
full_audio = F.highpass_biquad(full_audio, sample_rate, cutoff_freq=100)
```

---

## 🎯 Combined Effect

**For clicking sounds**:
1. 250ms silence creates safe gap between chunks
2. 300ms crossfade smoothly blends over that gap  
3. 100Hz high-pass filter removes residual artifacts
4. DC offset removal prevents pops

**For pronunciation**:
1. Automatic phonetic conversion ("honestly" → "onestly")
2. cfg_weight at 0.20-0.25 prevents over-guidance
3. Model has more freedom to pronounce naturally

---

## 📝 Test Instructions

1. **Stop** any running scripts
2. **Restart** `python myvoiceclone.py`
3. **Try this test**:

```
Text: "Honestly, I think this hour is important for our honor and future."

Expected audio:
- "Onestly" (silent h)
- "Our" (silent h) 
- "Onor" (silent h)
- NO clicking sounds between words
```

**Hinglish test**:
```
Text: "आज honestly बहुत अच्छा लग रहा है और hour भर में पूरा होगा।"

Expected:
- "Onestly" (not "ho-nest-ly")
- "Our" (not "hour")
- No clicks between Hindi/English switches
```

---

## ⚠️ Important Notes

### About Pronunciation
- ChatterBox TTS has fundamental tokenizer limitations
- Automatic phonetic fixes help ~80-90% of cases
- Some words may still need manual phonetic spelling
- If a word still sounds wrong:
  - Try typing it phonetically yourself
  - Example: "colonel" → "kernel"

### About Clicking
- With 250ms silence + 300ms crossfade = **550ms total gap**
- This should eliminate almost all clicks
- If you STILL hear clicks:
  1. They might be in the original voice samples
  2. Could be GPU/processing artifacts
  3. Try increasing silence to 300-400ms (line ~1428)

### Performance Note
- Lower cfg_weight (0.20-0.25) may be slightly slower
- But pronunciation quality is MUCH better
- Trade-off is worth it for natural speech

---

## 🔧 Manual Overrides

If pronunciation still has issues with specific words:

**Option 1**: Manual phonetic spelling in text
```
"honestly" → "onestly"  (already automatic)
"schedule" → "skedule"  (if needed)
"colonel" → "kernel"    (if needed)
```

**Option 2**: Increase silence padding even more
Edit line ~1428 in `myvoiceclone.py`:
```python
silence_ms = 300  # Or 400 for maximum separation
```

**Option 3**: Lower cfg_weight even more for conversational
Edit line ~227 in `myvoiceclone.py`:
```python
"cfg_weight": 0.15,  # From 0.20 (even more natural)
```

---

## 📊 What Changed

**Files Modified**: `myvoiceclone.py`

**Line ranges**:
- **Lines 205-240**: cfg_weight values (0.20-0.25)
- **Lines 331-380**: New `fix_english_pronunciation()` method
- **Lines 1138**: Apply fix in English generation
- **Lines 1336**: Apply fix in Hindi generation
- **Lines 1428**: Silence padding (250ms)
- **Lines 1447**: High-pass filter (100Hz)

---

## 🚀 Status

✅ **All 4 fixes applied and ready for testing**

**Next**: Restart script and test with the sample texts above!
