"""
Recreate voice profile using enhanced samples
This ensures the model learns from clean, processed audio
"""

import sys
from pathlib import Path
from myvoiceclone import MyVoiceClone
import torch

def recreate_profile(profile_name: str, enhanced_samples_folder: str):
    """
    Recreate a voice profile using enhanced samples
    
    Args:
        profile_name: Name for the new profile (e.g., "pritam_clean")
        enhanced_samples_folder: Path to folder with enhanced WAV files
    """
    print("\n🎙️  RECREATING VOICE PROFILE WITH ENHANCED SAMPLES")
    print("="*70)
    
    # Initialize voice clone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clone = MyVoiceClone(device=device)
    
    # Load models
    print("\n📦 Loading models...")
    clone.load_models(load_multilingual=True, load_english=False)
    
    # Delete old profile if exists
    if clone.get_profile(profile_name):
        print(f"\n🗑️  Deleting old profile '{profile_name}'...")
        clone.profile_manager.delete_profile(profile_name)
    
    # Create new profile
    print(f"\n✨ Creating new profile '{profile_name}'...")
    clone.create_profile(profile_name)
    
    # Find all enhanced WAV files
    samples_path = Path(enhanced_samples_folder)
    wav_files = list(samples_path.glob("*.wav"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {enhanced_samples_folder}")
        return False
    
    print(f"\n📁 Found {len(wav_files)} enhanced sample(s)")
    print("="*70)
    
    # Add each sample
    for i, wav_file in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] Adding: {wav_file.name}")
        
        # Assume multilingual (handles both English and Hindi)
        clone.add_voice_sample(
            profile_name=profile_name,
            audio_path=str(wav_file),
            language="multilingual"
        )
    
    print("\n" + "="*70)
    print(f"✅ Profile '{profile_name}' created with {len(wav_files)} enhanced samples!")
    print("\n💡 Now use this profile for generation with better quality and pronunciation")
    
    return True


def main():
    if len(sys.argv) < 3:
        print("\n❌ Usage: python recreate_profile_enhanced.py <profile_name> <enhanced_samples_folder>")
        print("\nExample:")
        print("  python recreate_profile_enhanced.py pritam_clean \"voice_profiles\\pritam\\samples_enhanced\"")
        print("\nThis will:")
        print("  1. Delete old 'pritam_clean' profile (if exists)")
        print("  2. Create new profile using enhanced samples")
        print("  3. Compute fresh embeddings from clean audio")
        return
    
    profile_name = sys.argv[1]
    samples_folder = sys.argv[2]
    
    recreate_profile(profile_name, samples_folder)


if __name__ == "__main__":
    main()
