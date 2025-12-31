"""
Video to MP3 Converter
Extracts audio from all video files in a folder and saves as MP3
"""

import os
from pathlib import Path
from moviepy import VideoFileClip

def convert_videos_to_mp3(
    input_folder: str,
    output_folder: str,
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
):
    """
    Convert all video files in input folder to MP3 files in output folder.
    
    Args:
        input_folder: Path to folder containing video files
        output_folder: Path to folder where MP3 files will be saved
        video_extensions: Tuple of video file extensions to process
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_folder)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
        video_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"❌ No video files found in {input_folder}")
        return
    
    print(f"📹 Found {len(video_files)} video file(s)")
    print(f"📁 Input:  {input_folder}")
    print(f"📁 Output: {output_folder}")
    print("="*60)
    
    success_count = 0
    failed_files = []
    
    for i, video_file in enumerate(video_files, 1):
        try:
            # Generate output filename
            output_filename = video_file.stem + '.mp3'
            output_file = output_path / output_filename
            
            print(f"\n[{i}/{len(video_files)}] Converting: {video_file.name}")
            
            # Load video and extract audio
            video = VideoFileClip(str(video_file))
            
            # Check if video has audio
            if video.audio is None:
                print(f"   ⚠️  Warning: No audio track found, skipping...")
                video.close()
                failed_files.append((video_file.name, "No audio track"))
                continue
            
            # Extract and save audio as MP3
            video.audio.write_audiofile(
                str(output_file),
                codec='mp3',
                bitrate='192k'
            )
            
            # Close video to free memory
            video.close()
            
            # Get file size
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved: {output_filename} ({size_mb:.2f} MB)")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            failed_files.append((video_file.name, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print(f"✨ Conversion Complete!")
    print(f"   ✅ Successful: {success_count}/{len(video_files)}")
    
    if failed_files:
        print(f"   ❌ Failed: {len(failed_files)}")
        print("\nFailed files:")
        for filename, error in failed_files:
            print(f"   • {filename}: {error}")


if __name__ == "__main__":
    # Example usage - modify these paths
    INPUT_FOLDER = "D:\\voice cloning\\videos"           # Folder with MP4/video files
    OUTPUT_FOLDER = "D:\\voice cloning\\audio"    # Folder for MP3 files
    
    print("🎵 Video to MP3 Converter")
    print("="*60)
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        print(f"📝 Creating folder: {INPUT_FOLDER}")
        os.makedirs(INPUT_FOLDER)
        print(f"✅ Please add video files to '{INPUT_FOLDER}' and run again.")
    else:
        # Run conversion
        convert_videos_to_mp3(INPUT_FOLDER, OUTPUT_FOLDER)
        print(f"\n✅ All MP3 files saved to: {OUTPUT_FOLDER}")
