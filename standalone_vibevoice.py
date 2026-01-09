"""
Standalone VibeVoice TTS Script
Requires: transformers==4.51.3 (conflicts with Chatterbox)

Usage:
    python standalone_vibevoice.py --text "नमस्ते" --profile pritam --output output.wav
"""
import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path

# Set model cache
os.environ['MODEL_CACHE_ROOT'] = 'D:/voice cloning/models_cache'
os.environ['HF_HOME'] = 'D:/voice cloning/models_cache/huggingface'
os.environ['TORCH_HOME'] = 'D:/voice cloning/models_cache/torch'

MODEL_PATH = "D:/voice cloning/models_cache/vibevoice-hindi-7b"
PROFILES_DIR = "D:/voice cloning/voice_profiles"

def load_vibevoice_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load VibeVoice model"""
    print("🔄 Loading VibeVoice-Hindi-7B...")
    
    # Check transformers version
    import transformers
    if transformers.__version__ != "4.51.3":
        print(f"⚠️ Warning: transformers version is {transformers.__version__}, expected 4.51.3")
        print(f"   Install correct version: pip install transformers==4.51.3")
        print(f"   Note: This will break Chatterbox compatibility!")
    
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    
    # Load processor and model
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager"  # Use for 4GB VRAM
    )
    
    print(f"✅ VibeVoice loaded on {device}")
    return model, processor

def get_reference_audio(profile_name):
    """Get reference audio from profile"""
    profile_path = Path(PROFILES_DIR) / profile_name
    
    # Try enhanced samples first
    enhanced_dir = profile_path / "samples_enhanced"
    if enhanced_dir.exists():
        samples = list(enhanced_dir.glob("*.wav"))
        if samples:
            return str(samples[0])
    
    # Fall back to regular samples
    samples_dir = profile_path / "samples"
    if samples_dir.exists():
        samples = list(samples_dir.glob("*.wav"))
        if samples:
            return str(samples[0])
    
    raise FileNotFoundError(f"No audio samples found for profile: {profile_name}")

def generate_speech(model, processor, text, reference_audio, device="cuda"):
    """Generate speech with VibeVoice"""
    print(f"\n📝 Text: {text}")
    print(f"🎵 Reference: {reference_audio}")
    
    # Load reference audio
    ref_wav, ref_sr = torchaudio.load(reference_audio)
    if ref_sr != 16000:
        ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 16000)
    
    # Process inputs
    inputs = processor(
        text=text,
        audio=ref_wav.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    print("🎙️ Generating...")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    
    # Extract audio
    audio = output.audio_values.squeeze().cpu()
    sample_rate = 16000
    
    print(f"✅ Generated {len(audio) / sample_rate:.2f}s audio")
    
    return audio, sample_rate

def main():
    parser = argparse.ArgumentParser(description="VibeVoice-Hindi-7B Standalone TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--profile", type=str, default="pritam", help="Voice profile name")
    parser.add_argument("--output", type=str, default="vibevoice_output.wav", help="Output audio file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print(f"   Download it first using:")
        print(f"   python -c \"from huggingface_hub import snapshot_download; snapshot_download('tarun7r/vibevoice-hindi-7b', local_dir='{MODEL_PATH}')\"")
        sys.exit(1)
    
    # Determine device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, processor = load_vibevoice_model(device)
    
    # Get reference audio
    reference_audio = get_reference_audio(args.profile)
    
    # Generate speech
    audio, sample_rate = generate_speech(model, processor, args.text, reference_audio, device)
    
    # Save output
    torchaudio.save(args.output, audio.unsqueeze(0), sample_rate)
    print(f"💾 Saved: {args.output}")

if __name__ == "__main__":
    main()
