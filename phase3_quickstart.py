"""
Phase 3 Quick Start - Dataset & Fine-Tuning Guide
==================================================

This script helps you get started with Phase 3:
1. Download datasets from Kaggle/HuggingFace
2. Prepare training data
3. Fine-tune your model

Run: python phase3_quickstart.py
"""

import os
import sys
from pathlib import Path

# Set environment for D: drive
os.environ['HF_HOME'] = 'D:\\voice cloning\\models_cache\\huggingface'
os.environ['TORCH_HOME'] = 'D:\\voice cloning\\models_cache\\torch'


def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           📂 PHASE 3: DATASETS & FINE-TUNING                     ║
║           Quick Start Guide                                      ║
╚══════════════════════════════════════════════════════════════════╝
""")


def check_kaggle_setup():
    """Check if Kaggle API is configured"""
    print("\n🔐 Checking Kaggle API setup...")
    
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("   ✅ Kaggle API configured")
        return True
    else:
        print("   ⚠️ Kaggle API not configured")
        print("""
   📋 To set up Kaggle API:
   1. Go to: https://www.kaggle.com/settings
   2. Scroll to "API" section
   3. Click "Create New Token"
   4. Save kaggle.json to: C:\\Users\\<username>\\.kaggle\\kaggle.json
""")
        return False


def show_recommended_datasets():
    """Show recommended datasets for different use cases"""
    print("\n📊 RECOMMENDED DATASETS BY USE CASE:")
    print("=" * 60)
    
    print("""
🎯 FOR QUICK EXPERIMENTS (< 2GB):
   • cmu_arctic (1.2 GB, 7 speakers, 7 hours)
     Command: manager.download('cmu_arctic')

🎯 FOR ENGLISH VOICE CLONING (2-10 GB):
   • ljspeech (2.6 GB, 1 speaker, 24 hours)
     Best for: Single-speaker fine-tuning
     Command: manager.download('ljspeech')
   
   • libritts_clean (5.6 GB, 251 speakers, 100 hours)
     Best for: Multi-speaker voice cloning
     Command: manager.download('libritts_clean')

🎯 FOR HINDI VOICE CLONING:
   • hindi_tts_kaggle (7.0 GB, 10 speakers, 50 hours)
     Best for: Hindi TTS fine-tuning
     Command: manager.download('hindi_tts_kaggle')
   
   • common_voice_hi (3.0 GB, 5000 speakers, 100 hours)
     Best for: Hindi ASR/TTS
     Command: manager.download('common_voice_hi')

🎯 FOR PRODUCTION (10+ GB):
   • vctk (10.9 GB, 110 speakers, 44 hours)
     Best for: Multi-accent English
     Command: manager.download('vctk')
""")


def interactive_download():
    """Interactive dataset download"""
    from dataset_manager import DatasetManager, DATASETS_CATALOG
    
    print("\n📥 DATASET DOWNLOAD")
    print("=" * 60)
    
    manager = DatasetManager()
    
    print("\n📋 Available datasets:")
    for i, (name, info) in enumerate(DATASETS_CATALOG.items(), 1):
        status = "✅" if name in manager.downloaded_datasets else "⬜"
        print(f"   {i}. {status} {name} ({info.size_gb} GB, {info.languages})")
    
    print("\n   0. Skip download")
    
    choice = input("\n👉 Enter number to download (or 0 to skip): ").strip()
    
    if choice == "0":
        return
    
    try:
        idx = int(choice) - 1
        dataset_name = list(DATASETS_CATALOG.keys())[idx]
        
        print(f"\n📥 Downloading {dataset_name}...")
        manager.download(dataset_name)
        
    except (ValueError, IndexError):
        print("❌ Invalid choice")


def prepare_training_data():
    """Guide for preparing training data"""
    print("\n📦 PREPARE TRAINING DATA")
    print("=" * 60)
    
    print("""
To prepare downloaded datasets for training:

```python
from prepare_dataset import DatasetPreparer

# Initialize preparer
preparer = DatasetPreparer("D:/voice cloning/training_data")

# Add dataset(s) - choose based on what you downloaded:

# Option 1: LJSpeech (single speaker)
preparer.add_ljspeech("D:/voice cloning/datasets/ljspeech")

# Option 2: VCTK (multi-speaker)
preparer.add_vctk("D:/voice cloning/datasets/vctk", max_speakers=20)

# Option 3: Your own recordings
preparer.add_custom_folder(
    audio_dir="your_audio_folder",
    transcripts_file="your_transcripts.csv",  # filename|text
    speaker_id="my_voice",
    language="en"  # or "hi" for Hindi
)

# Prepare final dataset
train_meta, val_meta = preparer.prepare(split_ratio=0.95)
```

Output will be in: D:/voice cloning/training_data/
   ├── wavs/          (resampled audio files)
   ├── metadata.csv   (all samples)
   ├── metadata_train.csv
   ├── metadata_val.csv
   └── speakers.json
""")


def show_fine_tuning_guide():
    """Guide for fine-tuning"""
    print("\n🔧 FINE-TUNING GUIDE")
    print("=" * 60)
    
    print("""
After preparing your dataset, run fine-tuning:

```bash
# Basic fine-tuning (50 epochs)
python fine_tuner.py --dataset "D:/voice cloning/training_data" --epochs 50

# With custom settings
python fine_tuner.py \\
    --dataset "D:/voice cloning/training_data" \\
    --epochs 100 \\
    --batch-size 1 \\
    --lr 1e-5 \\
    --output "D:/voice cloning/fine_tuned_models"
```

⚡ OPTIMIZATION FOR T2000 4GB VRAM:
   • Batch size: 1 (automatically set)
   • FP16 enabled (saves 50% VRAM)
   • Gradient checkpointing enabled
   • Gradient accumulation: 4 steps

📊 Expected Training Time:
   • 1000 samples, 50 epochs: ~2-4 hours
   • 5000 samples, 50 epochs: ~8-12 hours
   • 10000 samples, 50 epochs: ~20-30 hours

💾 Checkpoints saved to:
   D:/voice cloning/fine_tuned_models/
   ├── epoch_10/
   ├── epoch_20/
   ├── ...
   ├── best_model/
   └── final/
""")


def show_custom_recording_guide():
    """Guide for recording your own voice"""
    print("\n🎤 RECORD YOUR OWN VOICE DATASET")
    print("=" * 60)
    
    print("""
For best voice cloning results, record your own voice:

📋 RECORDING REQUIREMENTS:
   • 30-60 minutes of audio (minimum 100 samples)
   • Clear, quiet environment
   • Consistent microphone distance
   • Various emotions and tones
   • Both English and Hindi (if bilingual)

📝 TRANSCRIPT FORMAT (transcripts.csv):
   filename|text
   sample_001|Hello, my name is Bhomik.
   sample_002|This is a test recording.
   sample_003|नमस्ते, मेरा नाम भोमिक है।
   ...

📁 FOLDER STRUCTURE:
   my_recordings/
   ├── sample_001.wav
   ├── sample_002.wav
   ├── sample_003.wav
   └── transcripts.csv

🎙️ RECOMMENDED RECORDING SCRIPTS:

ENGLISH (30 sentences):
1. "Hello, my name is [name]. Nice to meet you."
2. "The weather today is absolutely beautiful."
3. "I can't believe how amazing this technology is!"
4. "Let me tell you a story about my childhood."
5. "Please speak slowly and clearly."
... (continue with various emotions)

HINDI (30 sentences):
1. "नमस्ते, मेरा नाम [नाम] है।"
2. "आज का मौसम बहुत अच्छा है।"
3. "यह तकनीक कितनी अद्भुत है!"
4. "मैं आपको अपने बचपन की कहानी सुनाता हूं।"
5. "कृपया धीरे और स्पष्ट बोलें।"
... (continue with various emotions)

💡 TIPS:
   • Record in WAV format (44.1kHz or 22.05kHz)
   • Each sample: 5-15 seconds
   • Include: neutral, excited, calm, dramatic tones
   • Avoid background noise and echo
""")


def main():
    print_header()
    
    while True:
        print("\n📋 PHASE 3 OPTIONS:")
        print("   1. Check Kaggle API setup")
        print("   2. View recommended datasets")
        print("   3. Download a dataset")
        print("   4. View data preparation guide")
        print("   5. View fine-tuning guide")
        print("   6. View custom recording guide")
        print("   7. Exit")
        
        choice = input("\n👉 Enter choice (1-7): ").strip()
        
        if choice == "1":
            check_kaggle_setup()
        elif choice == "2":
            show_recommended_datasets()
        elif choice == "3":
            try:
                interactive_download()
            except Exception as e:
                print(f"❌ Error: {e}")
                print("   Make sure Kaggle API is configured for Kaggle datasets")
        elif choice == "4":
            prepare_training_data()
        elif choice == "5":
            show_fine_tuning_guide()
        elif choice == "6":
            show_custom_recording_guide()
        elif choice == "7":
            print("\n👋 Goodbye! Happy voice cloning!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
