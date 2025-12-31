@echo off
REM ============================================
REM Set Model Cache Paths to D: Drive
REM Run this before using voice cloning system
REM ============================================

REM HuggingFace models (transformers, WavLM, etc.)
set HF_HOME=D:\voice cloning\models_cache\huggingface
set HUGGINGFACE_HUB_CACHE=D:\voice cloning\models_cache\huggingface\hub
set TRANSFORMERS_CACHE=D:\voice cloning\models_cache\huggingface\transformers

REM PyTorch models (torch.hub downloads)
set TORCH_HOME=D:\voice cloning\models_cache\torch

REM General cache
set XDG_CACHE_HOME=D:\voice cloning\models_cache

REM F5-TTS specific
set F5TTS_CACHE=D:\voice cloning\models_cache\f5tts

echo ✅ All model paths set to D: drive
echo HF_HOME: %HF_HOME%
echo TORCH_HOME: %TORCH_HOME%
