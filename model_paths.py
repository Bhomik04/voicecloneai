"""
Model Path Configuration for D: Drive Storage
Import this module at the start of your scripts to ensure all models download to D: drive
"""

import os

# Set all cache directories to D: drive BEFORE importing any ML libraries
MODEL_CACHE_ROOT = r"D:\voice cloning\models_cache"

# HuggingFace (transformers, datasets, WavLM, etc.)
os.environ['HF_HOME'] = os.path.join(MODEL_CACHE_ROOT, 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(MODEL_CACHE_ROOT, 'huggingface', 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(MODEL_CACHE_ROOT, 'huggingface', 'transformers')

# PyTorch hub
os.environ['TORCH_HOME'] = os.path.join(MODEL_CACHE_ROOT, 'torch')

# General XDG cache
os.environ['XDG_CACHE_HOME'] = MODEL_CACHE_ROOT

# F5-TTS specific
os.environ['F5TTS_CACHE'] = os.path.join(MODEL_CACHE_ROOT, 'f5tts')

# Vocos (used by F5-TTS)
os.environ['VOCOS_CACHE'] = os.path.join(MODEL_CACHE_ROOT, 'vocos')

# VibeVoice Hindi-7B model path
VIBEVOICE_MODEL_PATH = os.path.join(MODEL_CACHE_ROOT, 'vibevoice-hindi-7b')
os.environ['VIBEVOICE_MODEL_PATH'] = VIBEVOICE_MODEL_PATH

# Create directories if they don't exist
for subdir in ['huggingface', 'huggingface/hub', 'huggingface/transformers', 
               'torch', 'f5tts', 'vocos', 'vibevoice-hindi-7b']:
    path = os.path.join(MODEL_CACHE_ROOT, subdir)
    os.makedirs(path, exist_ok=True)

print(f"✅ Model cache configured: {MODEL_CACHE_ROOT}")


def get_cache_info():
    """Print current cache configuration"""
    print("\n📁 Model Cache Configuration:")
    print(f"   Root: {MODEL_CACHE_ROOT}")
    print(f"   HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"   TORCH_HOME: {os.environ.get('TORCH_HOME', 'Not set')}")
    print(f"   F5TTS_CACHE: {os.environ.get('F5TTS_CACHE', 'Not set')}")
    print(f"   VIBEVOICE_MODEL_PATH: {os.environ.get('VIBEVOICE_MODEL_PATH', 'Not set')}")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("D:\\")
        print(f"\n💾 D: Drive Space:")
        print(f"   Total: {total // (2**30)} GB")
        print(f"   Used: {used // (2**30)} GB")
        print(f"   Free: {free // (2**30)} GB")
    except:
        pass


if __name__ == "__main__":
    get_cache_info()
