# 🎙️ VoiceCloneAI - Ultimate Voice Cloning System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

**Professional-grade voice cloning with multi-engine support, Hindi/English TTS, and studio-quality audio enhancement**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-demo)

</div>

---

## 🌟 Overview

VoiceCloneAI is a state-of-the-art voice cloning system that combines multiple TTS engines with advanced audio processing to deliver broadcast-quality voice synthesis. Built for both English and Hindi languages, it features emotion-aware generation, prosody enhancement, and professional audio post-processing.

### Why VoiceCloneAI?

- 🎯 **Multi-Engine Architecture**: Chatterbox TTS + VibeVoice-Hindi-7B support
- 🌍 **True Multilingual**: Native English, Hindi, and Hinglish support
- 🎭 **Emotion-Aware**: 8 emotion presets with intelligent auto-detection
- 🎚️ **Studio Quality**: Professional audio enhancement pipeline
- 🇮🇳 **Hindi Prosody Fix**: Authentic Hindi accent with natural pitch patterns
- 💻 **Production Ready**: Web UI + CLI + Python API
- ⚡ **GPU Accelerated**: Optimized for NVIDIA GPUs (4GB+ VRAM)

---

## ✨ Features

### 🎤 Voice Cloning
- **Few-Shot Learning**: Clone any voice with 10-30 seconds of audio
- **Voice Profiles**: Save and reuse voice embeddings
- **Reference Audio**: Automatic sample selection from profile directories
- **Cross-Lingual**: Clone voices across English and Hindi

### 🗣️ Text-to-Speech Engines

#### Chatterbox TTS (Primary)
- **English Generation**: High-quality neural TTS
- **Multilingual Mode**: Hindi and Hinglish support
- **Emotion Presets**: conversational, excited, serious, professional, friendly, warm, dramatic, news_anchor
- **Real-time Capable**: Fast inference with FP16 support

#### VibeVoice-Hindi-7B (Optional)
- **7B Parameters**: Qwen2.5-7B backbone + 600M diffusion head
- **Native Hindi**: Specialized for authentic Hindi speech
- **Long-Form**: Up to 32K tokens (~45 minutes)
- **Zero-Shot**: Works with minimal reference audio

### 🎨 Audio Enhancement Pipeline

#### Phase 1: Studio Quality Processing
- **Adobe Podcast Enhancer**: Professional noise reduction
- **Studio Audio Processor**: Broadcast-quality EQ and dynamics
- **Spectral Denoising**: Advanced noise removal
- **Dynamic Range Control**: Consistent audio levels

#### Phase 2: Prosody & Naturalness
- **Hindi Prosody Enhancer** ⭐ **NEW**
  - Phrase-final pitch drops (~4 semitones)
  - Authentic Hindi intonation patterns
  - Micro-pitch variations for natural quality
  - Syllable-timed rhythm correction
  - Fixes "foreigner accent" problem in Hindi TTS

- **Emotion Analyzer**: Rule-based emotion detection
- **Prosody Processor**: Natural pauses and breath points
- **Word Emphasis**: Intelligent stress placement

### 🎮 Interfaces

#### Web UI (Gradio)
```bash
python myvoiceclone.py
# Access at http://localhost:7860
```
- Intuitive interface
- Real-time generation
- Audio playback and download
- Profile management

#### CLI Mode
```bash
python myvoiceclone.py --cli
```
- Quick generation from terminal
- Batch processing capable
- Script automation support

#### Python API
```python
from tts_compatibility import UnifiedTTS

tts = UnifiedTTS()
audio = tts.generate(
    text="Hello world!",
    profile="my_voice",
    emotion="conversational"
)
audio.save("output.wav")
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9-3.12
- NVIDIA GPU with 4GB+ VRAM (recommended)
- CUDA 11.8+ / ROCm 5.6+
- Windows/Linux/macOS

### Step 1: Clone Repository
```bash
git clone https://github.com/Bhomik04/voicecloneai.git
cd voicecloneai
```

### Step 2: Install Dependencies
```bash
# Create virtual environment (recommended)
conda create -n voiceclone python=3.12
conda activate voiceclone

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install requirements
pip install -r requirements.txt

# Install Chatterbox TTS
cd chatterbox
pip install -e .
cd ..
```

### Step 3: Download Models

Models will auto-download on first run, or you can pre-download:

```bash
# Chatterbox models (auto-downloads to models_cache/)
python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained()"

# Optional: VibeVoice-Hindi-7B (~30GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tarun7r/vibevoice-hindi-7b', 
                  local_dir='models_cache/vibevoice-hindi-7b')
"
```

### Step 4: Verify Installation
```bash
python myvoiceclone.py --help
```

---

## 🎯 Quick Start

### 1. Create a Voice Profile

Place 10-30 seconds of clean audio samples in `voice_profiles/your_name/samples/`:

```bash
voice_profiles/
└── john_doe/
    ├── samples/
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── sample_003.wav
    └── metadata.json
```

**Audio Requirements:**
- Format: WAV, MP3, or FLAC
- Sample Rate: 16kHz or higher
- Duration: 3-10 seconds per sample
- Quality: Clear speech, minimal background noise
- Total: 10-30 seconds across all samples

### 2. Generate Speech

#### Using Web UI:
```bash
python myvoiceclone.py
```
1. Select voice profile
2. Enter text
3. Choose emotion
4. Click Generate
5. Download output

#### Using CLI:
```bash
python myvoiceclone.py --cli

# Follow prompts:
# Profile: john_doe
# Text: Hello, this is a test.
# Emotion: conversational
```

#### Using Python API:
```python
from tts_compatibility import UnifiedTTS

tts = UnifiedTTS()
result = tts.generate(
    text="नमस्ते, आप कैसे हैं?",  # Hindi text
    profile="john_doe",
    emotion="friendly"
)
result.save("output.wav")
print(f"Duration: {result.duration}s, RTF: {result.rtf}x")
```

### 3. Hindi Generation with Enhanced Prosody

```python
# Automatic Hindi prosody enhancement
result = tts.generate(
    text="मेरा नाम जॉन है। मैं भारत से हूं।",
    profile="john_doe",
    language="hi-IN"  # Forces Hindi mode
)
result.save("hindi_output.wav")
```

The Hindi Prosody Enhancer automatically:
- ✅ Adds phrase-final pitch drops
- ✅ Applies authentic Hindi intonation
- ✅ Adjusts rhythm to syllable-timing
- ✅ Adds micro-variations for naturalness

---

## 📖 Documentation

### Emotion Presets

| Emotion | Use Case | Characteristics |
|---------|----------|-----------------|
| `conversational` | Default, chat | Natural, friendly tone |
| `excited` | Announcements | High energy, enthusiastic |
| `serious` | Professional | Authoritative, formal |
| `professional` | Business | Clear, confident |
| `friendly` | Casual | Warm, approachable |
| `warm` | Personal | Gentle, caring |
| `dramatic` | Storytelling | Expressive, theatrical |
| `news_anchor` | Broadcasting | Clear, measured |

### Configuration

Edit `model_paths.py` to customize cache locations:

```python
MODEL_CACHE_ROOT = "D:/your_path/models_cache"  # Model storage
HF_HOME = f"{MODEL_CACHE_ROOT}/huggingface"     # Hugging Face cache
TORCH_HOME = f"{MODEL_CACHE_ROOT}/torch"        # PyTorch models
```

### Advanced Usage

#### Batch Processing
```python
texts = ["Hello world", "नमस्ते दुनिया", "Testing voice"]
for i, text in enumerate(texts):
    result = tts.generate(text, profile="john_doe")
    result.save(f"output_{i}.wav")
```

#### Custom Enhancement Settings
```python
from hindi_prosody_enhancer import HindiProsodyEnhancer

enhancer = HindiProsodyEnhancer()
enhanced_audio = enhancer.enhance(
    audio_array, 
    sample_rate,
    intensity=1.0  # 0.0-1.5, higher = stronger effect
)
```

#### Voice Profile Management
```python
from tts_compatibility import VoiceProfileManager

manager = VoiceProfileManager()
profiles = manager.list_profiles()
profile = manager.get_profile("john_doe")
best_ref = manager.get_best_reference("john_doe")
```

---

## 🏗️ Architecture

```
VoiceCloneAI
│
├── Core TTS Engines
│   ├── Chatterbox (English/Hindi/Hinglish)
│   └── VibeVoice-Hindi-7B (Native Hindi)
│
├── Audio Enhancement Pipeline
│   ├── Phase 1: Studio Quality
│   │   ├── Adobe Podcast Enhancer
│   │   ├── Studio Audio Processor
│   │   └── Spectral Denoising
│   │
│   └── Phase 2: Prosody & Naturalness
│       ├── Hindi Prosody Enhancer ⭐
│       ├── Emotion Analyzer
│       ├── Prosody Processor
│       └── Audio Post-Processing
│
├── Voice Management
│   ├── Profile System
│   ├── Embedding Cache
│   └── Sample Management
│
└── Interfaces
    ├── Gradio Web UI
    ├── CLI
    └── Python API
```

---

## 🎪 Demo

### Sample Outputs

**English:**
> "Welcome to VoiceCloneAI, the ultimate voice cloning system with professional-grade audio quality."

**Hindi:**
> "नमस्ते, वॉइस क्लोन एआई में आपका स्वागत है। यह एक उन्नत वॉइस क्लोनिंग सिस्टम है।"

**Hinglish:**
> "Hello दोस्तों, aaj hum dekhenge ki kaise VoiceCloneAI use करते हैं।"

---

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

#### Out of Memory (OOM)
- Reduce batch size
- Use CPU mode: `device="cpu"`
- Enable low VRAM mode in config

#### Hindi Sounds Like "Foreigner"
The Hindi Prosody Enhancer should fix this! If issues persist:
```python
# Increase enhancement intensity
enhancer.enhance(audio, sr, intensity=1.5)  # Max: 1.5
```

#### Model Download Slow
```bash
# Use mirror or manual download
# Models stored in: models_cache/
# See VIBEVOICE_INTEGRATION_STATUS.md for details
```

### Dependency Conflicts

Chatterbox and VibeVoice require different transformers versions:
- **Chatterbox**: transformers==4.46.3 ✅ (Recommended)
- **VibeVoice**: transformers==4.51.3 ⚠️ (Conflicts)

**Solution**: Use separate conda environments (see `VIBEVOICE_INTEGRATION_STATUS.md`)

---

## 📊 Performance

### Benchmarks (NVIDIA Quadro T2000, 4GB VRAM)

| Engine | Language | RTF | Quality | VRAM |
|--------|----------|-----|---------|------|
| Chatterbox | English | 20x | Excellent | 2.5GB |
| Chatterbox | Hindi | 18x | Very Good* | 2.8GB |
| VibeVoice | Hindi | 12x | Excellent | 3.8GB |

*With Hindi Prosody Enhancer applied

**RTF (Real-Time Factor)**: 20x = generates 20 seconds of audio per 1 second of compute time

---

## 🗺️ Roadmap

- [x] Multi-engine support (Chatterbox + VibeVoice)
- [x] Hindi prosody enhancement
- [x] Studio-quality audio processing
- [x] Emotion-aware generation
- [x] Web UI and CLI
- [ ] Real-time streaming mode
- [ ] Voice conversion (voice-to-voice)
- [ ] Multi-speaker conversations
- [ ] Fine-tuning on custom datasets
- [ ] API server with REST endpoints
- [ ] Mobile app support
- [ ] Additional language support (Tamil, Telugu, Bengali)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

### Development Setup
```bash
git clone https://github.com/Bhomik04/voicecloneai.git
cd voicecloneai
pip install -e .
pip install -r requirements-dev.txt  # Additional dev tools
```

### Code Style
- Follow PEP 8
- Add type hints
- Document functions with docstrings
- Run tests before committing

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **Chatterbox TTS**: MIT License
- **VibeVoice**: Microsoft Research License
- **Transformers**: Apache 2.0 License

---

## 🙏 Acknowledgments

- **Chatterbox Team** for the excellent TTS engine
- **Microsoft Research** for VibeVoice-Hindi-7B
- **Hugging Face** for transformers library
- **PyTorch Team** for the deep learning framework
- **Gradio** for the web interface framework

---

## 📧 Contact

- **Author**: Bhomik Pal
- **GitHub**: [@Bhomik04](https://github.com/Bhomik04)
- **Repository**: [voicecloneai](https://github.com/Bhomik04/voicecloneai)
- **Issues**: [Report Bug](https://github.com/Bhomik04/voicecloneai/issues)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ by [Bhomik04](https://github.com/Bhomik04)**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Bhomik04.voicecloneai)

</div>
