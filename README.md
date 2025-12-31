# 🎙️ VoiceCloneAI - Advanced Voice Cloning System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced, production-ready voice cloning system featuring emotional TTS, multi-language support, studio-quality audio processing, and real-time voice conversion. Built on top of state-of-the-art models including Chatterbox-Turbo and F5-TTS.

## ✨ Key Features

### 🎯 Core Capabilities
- **Multi-Emotion Voice Cloning**: Generate speech with various emotions (excited, calm, dramatic, conversational)
- **Bilingual Support**: Clone voices in both English and Hindi
- **Long-Form Generation**: Create extended audio (1+ minutes) with perfect quality
- **Studio-Quality Processing**: Professional audio enhancement and artifact removal
- **Real-Time Voice Conversion**: Convert your voice to the cloned voice in real-time
- **Zero-Shot Cloning**: Clone any voice with just a few seconds of audio

### 🔧 Advanced Features
- **Intelligent Audio Chunking**: Smart text segmentation with smooth crossfade transitions
- **Prosody Control**: Fine-tune pitch, speed, and rhythm for natural-sounding speech
- **Emotion Analysis**: Automatic detection and replication of emotional characteristics
- **Voice Similarity Analysis**: Measure and optimize voice similarity scores
- **Audio Enhancement Pipeline**: Multi-stage processing including noise reduction, normalization, and artifact removal
- **Profile Management**: Save and reuse voice profiles with metadata

### 🎨 User Interface
- **Gradio Web Interface**: Easy-to-use web UI for voice cloning
- **Multiple TTS Modes**: Standard TTS, Turbo TTS, and Voice Conversion
- **Batch Processing**: Generate multiple outputs efficiently
- **Real-Time Preview**: Listen to samples before final generation

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with 4GB+ VRAM (tested on T2000 4GB)
- 32GB RAM recommended
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/pritam-ray/voicecloneai.git
cd voicecloneai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure model paths** (for Windows users with D: drive)
```bash
set_model_paths.bat
```

4. **Download required models** (automatic on first run)
```bash
python myvoiceclone.py
```

### Basic Usage

#### Option 1: Web Interface (Recommended)
```bash
python myvoiceclone.py
```
Then open your browser to `http://localhost:7860`

#### Option 2: Quick Start Script
```bash
python phase3_quickstart.py
```

#### Option 3: Chatterbox Models
```bash
cd chatterbox
python gradio_tts_app.py        # Standard TTS
python gradio_tts_turbo_app.py  # Turbo TTS (faster)
python gradio_vc_app.py         # Voice Conversion
```

## 📖 Documentation

### Core Modules

| Module | Description | File |
|--------|-------------|------|
| **Voice Cloning** | Main voice cloning interface | `myvoiceclone.py` |
| **Enhanced Cloning** | Advanced features with emotion control | `enhanced_voice_clone.py` |
| **Audio Processing** | Professional audio enhancement | `studio_audio_processor.py` |
| **F5 Engine** | F5-TTS integration | `f5_engine.py` |
| **Emotion Analysis** | Emotional characteristic detection | `emotion_analyzer.py` |
| **Voice Analysis** | Similarity scoring and optimization | `voice_similarity_analyzer.py` |

### Guides

- 📘 [Complete Fix Guide](COMPLETE_FIX_GUIDE.md) - Comprehensive troubleshooting
- 🎯 [Enhanced Voice Guide](ENHANCED_VOICE_GUIDE.md) - Advanced features walkthrough
- 🎨 [Studio Quality Guide](STUDIO_QUALITY_GUIDE.md) - Professional audio processing
- 🔧 [Profile System Guide](PROFILE_SYSTEM_GUIDE.md) - Voice profile management
- ⚡ [Quick Start Enhanced](QUICK_START_ENHANCED.md) - Fast setup guide

## 🎯 Use Cases

### 1. Voice Cloning
Clone your voice or any reference voice with just 10-15 seconds of audio:
```python
from myvoiceclone import VoiceCloner

cloner = VoiceCloner()
cloner.create_profile(
    name="my_voice",
    samples_dir="path/to/samples",
    emotions=["neutral", "excited", "calm"]
)
```

### 2. Text-to-Speech Generation
```python
output = cloner.generate_speech(
    text="Hello! This is my cloned voice.",
    emotion="excited",
    language="en"
)
```

### 3. Long-Form Content
```python
output = cloner.generate_long_form(
    text=long_script,
    chunk_size=200,
    crossfade_duration=0.5
)
```

### 4. Voice Conversion
```python
converted = cloner.convert_voice(
    source_audio="my_recording.wav",
    target_voice="profile_name"
)
```

## 🏗️ Project Structure

```
voicecloneai/
├── myvoiceclone.py              # Main voice cloning system
├── enhanced_voice_clone.py       # Enhanced features
├── f5_engine.py                  # F5-TTS engine
├── studio_audio_processor.py     # Audio enhancement
├── emotion_analyzer.py           # Emotion detection
├── voice_similarity_analyzer.py  # Similarity analysis
├── prosody_processor.py          # Prosody control
├── dataset_manager.py            # Dataset management
├── model_paths.py                # Model configuration
├── chatterbox/                   # Chatterbox TTS integration
│   ├── gradio_tts_app.py        # Standard TTS interface
│   ├── gradio_tts_turbo_app.py  # Turbo TTS interface
│   ├── gradio_vc_app.py         # Voice conversion interface
│   └── src/chatterbox/          # Core Chatterbox modules
├── audio/                        # Input audio samples
├── audio_output/                 # Generated audio outputs
├── voice_profiles/              # Saved voice profiles
├── models_cache/                # Cached models
└── datasets/                    # Training datasets
```

## 🔧 Configuration

### Hardware Optimization
The system automatically optimizes for your hardware. For NVIDIA T2000 (4GB VRAM):
- Enables TF32 for faster operations
- Uses mixed precision training
- Implements gradient checkpointing
- Optimizes batch sizes

### Model Paths
Configure custom model cache locations in `model_paths.py`:
```python
CACHE_BASE = "D:/models_cache"  # Change to your preferred location
```

## 🎨 Features in Detail

### Emotion Control
- **Excited**: High energy, enthusiastic delivery
- **Calm**: Peaceful, soothing tone
- **Dramatic**: Emphasized, theatrical delivery
- **Conversational**: Natural, casual speaking style

### Audio Enhancement Pipeline
1. **Noise Reduction**: AI-powered denoising
2. **Normalization**: Consistent volume levels
3. **Artifact Removal**: TTS artifact detection and removal
4. **Studio Processing**: Professional-grade enhancement
5. **Crossfade Blending**: Smooth transitions for long-form content

### Voice Profile System
Save and manage multiple voice profiles:
- Automatic embedding generation
- Metadata tracking
- Quality reports
- Sample organization

## 📊 Performance

- **Generation Speed**: ~2-5x real-time (depending on hardware)
- **Memory Usage**: 4-8GB VRAM for standard models
- **Audio Quality**: Studio-grade (24-bit, 24kHz default)
- **Similarity Score**: 85-95% on average

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size in configuration
- Use gradient checkpointing
- Close other GPU applications

**Audio Quality Issues**
- Use studio audio processing pipeline
- Ensure clean reference samples
- Check sample rate consistency

**Model Download Failures**
- Check internet connection
- Verify cache directory permissions
- Use manual download from HuggingFace

For detailed troubleshooting, see [COMPLETE_FIX_GUIDE.md](COMPLETE_FIX_GUIDE.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon several excellent open-source projects:
- [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI - State-of-the-art TTS models
- [F5-TTS](https://github.com/SWivid/F5-TTS) - Advanced text-to-speech system
- PyTorch, Transformers, and the broader ML community

## 📧 Contact

Pritam Ray - [@pritam-ray](https://github.com/pritam-ray)

Project Link: [https://github.com/pritam-ray/voicecloneai](https://github.com/pritam-ray/voicecloneai)

## 🗺️ Roadmap

- [ ] Real-time streaming TTS
- [ ] Multi-speaker conversation generation
- [ ] Fine-tuning interface for custom models
- [ ] API server for production deployment
- [ ] Mobile app integration
- [ ] Additional language support (Spanish, French, German)
- [ ] Voice style transfer capabilities
- [ ] Advanced prosody controls

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ by [Pritam Ray](https://github.com/pritam-ray)**
