"""
🎙️ ENHANCED VOICE CLONE - Podcast Quality Voice Cloning
========================================================

Features:
✅ Automatic audio preprocessing for training samples
✅ Post-processing for podcast-quality output
✅ Better multilingual (English/Hindi) pronunciation
✅ Professional audio enhancement pipeline
✅ Noise reduction and audio cleanup

Usage:
    python enhanced_voice_clone.py
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import gradio as gr

# Import audio processor
from audio_processor import PodcastAudioProcessor
from advanced_audio_processor import NaturalSpeechProcessor

# Import existing voice cloning system
from myvoiceclone import MyVoiceClone, ProfileManager


class EnhancedVoiceClone:
    """
    Enhanced voice cloning with professional audio processing
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize enhanced voice cloning system
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        print("\n🎙️ Initializing Enhanced Voice Clone System...")
        print("="*70)
        
        # Initialize audio processor
        print("📊 Loading audio processor...")
        self.audio_processor = PodcastAudioProcessor(target_sr=24000)
        self.natural_speech_processor = NaturalSpeechProcessor(target_sr=24000)
        print("   ✅ Audio processor ready")
        
        # Initialize voice cloning system
        print("🤖 Loading voice cloning models...")
        self.voice_clone = MyVoiceClone(device=device)
        print("   ✅ Voice cloning ready")
        
        print("\n✨ Enhanced Voice Clone System Ready!")
        print("="*70)
    
    def preprocess_samples(self, 
                          input_folder: str,
                          output_folder: Optional[str] = None,
                          ) -> str:
        """
        Preprocess audio samples for better voice cloning
        
        Args:
            input_folder: Folder containing raw audio samples
            output_folder: Folder to save enhanced samples (auto-generated if None)
            
        Returns:
            Path to enhanced samples folder
        """
        if output_folder is None:
            output_folder = input_folder + "_enhanced"
        
        print(f"\n🎙️  PREPROCESSING AUDIO SAMPLES")
        print("="*70)
        print("This will:")
        print("  ✓ Remove background noise and hiss")
        print("  ✓ Normalize audio levels")
        print("  ✓ Remove silence at start/end")
        print("  ✓ Apply professional enhancement")
        print("="*70)
        
        from audio_processor import process_folder
        process_folder(input_folder, output_folder, self.audio_processor)
        
        return output_folder
    
    def create_profile_enhanced(self,
                               profile_name: str,
                               raw_samples_folder: str,
                               preprocess: bool = True) -> bool:
        """
        Create voice profile with optional audio preprocessing
        
        Args:
            profile_name: Name for the voice profile
            raw_samples_folder: Folder containing voice samples
            preprocess: Whether to preprocess samples first
            
        Returns:
            Success status
        """
        samples_folder = raw_samples_folder
        
        # Preprocess if requested
        if preprocess:
            print(f"\n📊 Preprocessing samples for profile '{profile_name}'...")
            samples_folder = self.preprocess_samples(raw_samples_folder)
            print(f"✅ Samples enhanced and saved to: {samples_folder}")
        
        # Get all audio files
        audio_files = []
        samples_path = Path(samples_folder)
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            audio_files.extend(list(samples_path.glob(f'*{ext}')))
        
        if not audio_files:
            print(f"❌ No audio files found in {samples_folder}")
            return False
        
        print(f"\n📁 Found {len(audio_files)} sample(s)")
        
        # Create profile using voice cloning system
        audio_paths = [str(f) for f in audio_files]
        
        # Assume multilingual (Hindi + English) - let the system handle it
        languages = ["multilingual"] * len(audio_paths)
        
        # Create profile
        success = self.voice_clone.base.profile_manager.create_profile_gradio(
            profile_name=profile_name,
            audio_files=audio_paths,
            languages=languages
        )
        
        if success:
            print(f"✅ Profile '{profile_name}' created successfully!")
        else:
            print(f"❌ Failed to create profile '{profile_name}'")
        
        return success
    
    def generate_enhanced(self,
                         text: str,
                         profile_name: str,
                         emotion: str = "neutral",
                         language: str = "auto",
                         output_path: Optional[str] = None,
                         apply_post_processing: bool = True):
        """
        Generate speech with post-processing enhancement
        
        Args:
            text: Text to synthesize
            profile_name: Voice profile to use
            emotion: Emotion style
            language: Language ('auto', 'en', 'hi', 'multilingual')
            output_path: Path to save output (auto-generated if None)
            apply_post_processing: Apply podcast-quality enhancement
            
        Returns:
            Tuple of (audio_array, sample_rate, output_path)
        """
        from datetime import datetime
        
        print(f"\n🎙️  GENERATING ENHANCED VOICE")
        print("="*70)
        print(f"📝 Text: {text[:100]}...")
        print(f"👤 Profile: {profile_name}")
        print(f"😊 Emotion: {emotion}")
        print(f"🌐 Language: {language}")
        print("="*70)
        
        # Generate audio using voice cloning
        audio, sr = self.voice_clone.generate(
            text=text,
            profile_name=profile_name,
            emotion=emotion,
            language=language,
            apply_audio_enhancement=True  # Use built-in enhancement
        )
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = Path("D:/voice cloning/audio_output")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"{profile_name}_{emotion}_{timestamp}.wav")
        
        # Save raw TTS output
        temp_path = output_path.replace(".wav", "_raw.wav")
        sf.write(temp_path, audio, sr)
        
        # Apply post-processing if requested
        if apply_post_processing:
            print("\n⚡ Applying podcast-quality post-processing...")
            
            # First pass: Basic cleanup
            temp_cleaned = temp_path.replace("_raw.wav", "_cleaned.wav")
            success = self.audio_processor.process_tts_output(temp_path, temp_cleaned)
            
            if success:
                # Second pass: Add natural dynamics and prosody
                print("\n🎭 Adding natural speech dynamics and pacing...")
                audio_clean, sr_clean = sf.read(temp_cleaned)
                audio_natural = self.natural_speech_processor.process_tts_output(audio_clean, sr_clean)
                
                # Save final enhanced audio
                sf.write(output_path, audio_natural, sr_clean)
                print(f"   ✅ Enhanced audio saved to: {output_path}")
                
                # Load enhanced audio to return
                audio, sr = audio_natural, sr_clean
                
                # Clean up temp files
                os.remove(temp_path)
                os.remove(temp_cleaned)
            else:
                print(f"   ⚠️  Post-processing failed, using raw output")
                os.rename(temp_path, output_path)
        else:
            os.rename(temp_path, output_path)
        
        print(f"\n✅ Generation complete!")
        print(f"📁 Saved to: {output_path}")
        
        return audio, sr, output_path


def create_enhanced_interface(enhanced_clone: EnhancedVoiceClone):
    """Create Gradio interface for enhanced voice cloning"""
    
    with gr.Blocks(title="Enhanced Voice Clone - Podcast Quality", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ Enhanced Voice Clone - Podcast Quality
        
        Professional voice cloning with automatic audio enhancement for crystal-clear, podcast-quality output.
        
        ### 🌟 Features:
        - 🎯 **Multilingual Support**: Perfect English & Hindi pronunciation
        - 🎚️ **Professional Audio**: Noise reduction, normalization, compression
        - 🎭 **Emotion Control**: Multiple emotional styles
        - 🎧 **Podcast Quality**: Studio-grade output processing
        """)
        
        with gr.Tab("🎤 Create Voice Profile"):
            gr.Markdown("### Step 1: Create Your Voice Profile")
            
            with gr.Row():
                with gr.Column():
                    profile_name_input = gr.Textbox(
                        label="Profile Name",
                        placeholder="e.g., pritam",
                        info="Choose a unique name for this voice profile"
                    )
                    
                    samples_folder_input = gr.Textbox(
                        label="Samples Folder Path",
                        placeholder="D:\\voice cloning\\voice_profiles\\pritam\\samples",
                        info="Folder containing your voice samples (MP3, WAV, etc.)"
                    )
                    
                    preprocess_checkbox = gr.Checkbox(
                        label="Preprocess Samples (Recommended)",
                        value=True,
                        info="Apply noise reduction and enhancement to samples"
                    )
                    
                    create_btn = gr.Button("🎙️ Create Profile", variant="primary", size="lg")
                
                with gr.Column():
                    create_status = gr.Textbox(
                        label="Status",
                        lines=10,
                        interactive=False
                    )
            
            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - Use 10-30 samples of 5-15 seconds each
            - Include samples in both English and Hindi (if multilingual)
            - Vary emotions and speaking styles
            - Record in a quiet environment
            - Preprocessing is **highly recommended** for better quality
            """)
        
        with gr.Tab("🎙️ Generate Speech"):
            gr.Markdown("### Step 2: Generate Podcast-Quality Speech")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        lines=5,
                        placeholder="Enter text in English, Hindi, or mixed...\n\nExample:\nHello दोस्तों! आज हम बात करेंगे AI voice cloning के बारे में.",
                        info="Supports English, Hindi, and code-switching"
                    )
                    
                    profile_select = gr.Dropdown(
                        label="Voice Profile",
                        choices=[],  # Will be populated dynamically
                        info="Select which voice to use"
                    )
                    
                    emotion_select = gr.Dropdown(
                        label="Emotion/Style",
                        choices=["neutral", "happy", "excited", "calm", "dramatic", "conversational"],
                        value="neutral",
                        info="Choose speaking style"
                    )
                    
                    language_select = gr.Dropdown(
                        label="Language Mode",
                        choices=["auto", "en", "hi", "multilingual"],
                        value="auto",
                        info="Auto-detect recommended for mixed content"
                    )
                    
                    postprocess_checkbox = gr.Checkbox(
                        label="Apply Podcast Enhancement (Recommended)",
                        value=True,
                        info="Professional audio processing for studio quality"
                    )
                    
                    generate_btn = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")
                
                with gr.Column():
                    audio_output = gr.Audio(
                        label="Generated Speech",
                        type="numpy"
                    )
                    
                    output_path_display = gr.Textbox(
                        label="Saved to",
                        interactive=False
                    )
            
            gr.Markdown("""
            ### 🎧 Audio Enhancement Features:
            - **Noise Reduction**: Removes background hiss and hum
            - **Compression**: Consistent volume throughout
            - **EQ Enhancement**: Boosts vocal clarity and presence
            - **Normalization**: Optimal loudness levels
            - **Limiting**: Prevents clipping and distortion
            """)
        
        # Event handlers
        def create_profile_handler(profile_name, samples_folder, preprocess):
            try:
                import sys
                from io import StringIO
                
                # Capture output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                # Create profile
                success = enhanced_clone.create_profile_enhanced(
                    profile_name=profile_name,
                    raw_samples_folder=samples_folder,
                    preprocess=preprocess
                )
                
                # Get output
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                if success:
                    output += f"\n\n✅ Profile '{profile_name}' created successfully!"
                else:
                    output += f"\n\n❌ Failed to create profile '{profile_name}'"
                
                # Update profile dropdown
                profiles = list(enhanced_clone.voice_clone.base.profile_manager.profiles.keys())
                
                return output, gr.update(choices=profiles)
                
            except Exception as e:
                sys.stdout = old_stdout
                return f"❌ Error: {str(e)}", gr.update()
        
        def generate_handler(text, profile, emotion, language, postprocess):
            try:
                audio, sr, path = enhanced_clone.generate_enhanced(
                    text=text,
                    profile_name=profile,
                    emotion=emotion,
                    language=language,
                    apply_post_processing=postprocess
                )
                
                return (sr, audio), path
                
            except Exception as e:
                return None, f"❌ Error: {str(e)}"
        
        # Connect handlers
        create_btn.click(
            fn=create_profile_handler,
            inputs=[profile_name_input, samples_folder_input, preprocess_checkbox],
            outputs=[create_status, profile_select]
        )
        
        generate_btn.click(
            fn=generate_handler,
            inputs=[text_input, profile_select, emotion_select, language_select, postprocess_checkbox],
            outputs=[audio_output, output_path_display]
        )
        
        # Load existing profiles on startup
        def load_profiles():
            profiles = list(enhanced_clone.voice_clone.base.profile_manager.profiles.keys())
            return gr.update(choices=profiles)
        
        demo.load(fn=load_profiles, outputs=[profile_select])
    
    return demo


def main():
    """Main entry point"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Enhanced Voice Clone - Podcast Quality")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--preprocess-only", type=str, help="Only preprocess samples in specified folder")
    
    args = parser.parse_args()
    
    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhanced_clone = EnhancedVoiceClone(device=device)
    
    # Preprocess-only mode
    if args.preprocess_only:
        print("\n🎙️  PREPROCESSING MODE")
        enhanced_clone.preprocess_samples(args.preprocess_only)
        print("\n✅ Preprocessing complete!")
        return
    
    # Web interface
    print("\n🚀 Starting Enhanced Voice Clone Web Interface...")
    demo = create_enhanced_interface(enhanced_clone)
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
