"""
Fix Audio Quality - Enhance existing TTS output files
Makes them sound natural with proper dynamics and pacing
"""

import sys
from pathlib import Path
from advanced_audio_processor import NaturalSpeechProcessor

def main():
    print("\n🎙️  AUDIO QUALITY FIXER")
    print("="*70)
    print("This will make your TTS output sound natural with:")
    print("  ✓ Natural volume dynamics (loud → soft at sentence end)")
    print("  ✓ Proper sentence pacing and pauses")
    print("  ✓ Aggressive noise reduction")
    print("  ✓ De-essing (reduce harsh S sounds)")
    print("  ✓ Warmth and body (less robotic)")
    print("  ✓ Broadcast-quality limiting")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n❌ Please provide an audio file to enhance")
        print("\nUsage:")
        print("  python fix_audio_quality.py your_audio.wav")
        print("  python fix_audio_quality.py your_audio.wav output_enhanced.wav")
        print("\nOr process a folder:")
        print("  python fix_audio_quality.py audio_output/")
        return
    
    input_path = Path(sys.argv[1])
    
    # Initialize processor
    processor = NaturalSpeechProcessor()
    
    # Process single file
    if input_path.is_file():
        output_path = sys.argv[2] if len(sys.argv) > 2 else str(input_path).replace('.wav', '_enhanced.wav')
        
        print(f"\n📁 Input:  {input_path}")
        print(f"📁 Output: {output_path}")
        print("\n" + "="*70)
        
        success = processor.process_file(str(input_path), output_path)
        
        if success:
            print("\n✅ Enhancement complete!")
            print(f"📁 Enhanced file saved to: {output_path}")
        else:
            print("\n❌ Enhancement failed!")
    
    # Process folder
    elif input_path.is_dir():
        wav_files = list(input_path.glob("*.wav"))
        
        if not wav_files:
            print(f"\n❌ No WAV files found in {input_path}")
            return
        
        print(f"\n📁 Found {len(wav_files)} WAV file(s) in {input_path}")
        print("="*70)
        
        success_count = 0
        for i, wav_file in enumerate(wav_files, 1):
            # Skip already enhanced files
            if '_enhanced' in wav_file.stem:
                print(f"\n[{i}/{len(wav_files)}] Skipping {wav_file.name} (already enhanced)")
                continue
            
            output_path = str(wav_file).replace('.wav', '_enhanced.wav')
            
            print(f"\n[{i}/{len(wav_files)}] Processing: {wav_file.name}")
            
            if processor.process_file(str(wav_file), output_path):
                success_count += 1
                print(f"   ✅ Saved: {Path(output_path).name}")
            else:
                print(f"   ❌ Failed")
        
        print("\n" + "="*70)
        print(f"✅ Enhanced {success_count}/{len(wav_files)} files")
    
    else:
        print(f"\n❌ {input_path} not found!")


if __name__ == "__main__":
    main()
