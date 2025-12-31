"""
Dataset Preparation - Convert Datasets to Training Format
=========================================================

Converts various TTS dataset formats to unified format for fine-tuning:
- LJSpeech format → Training format
- VCTK format → Training format
- Common Voice → Training format
- Custom audio + text → Training format

Output format:
  dataset/
  ├── wavs/
  │   ├── 000001.wav
  │   └── ...
  ├── metadata.csv  (filename|text)
  └── speakers.json (speaker info)
"""

import os
import csv
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torchaudio as ta
import torch
from tqdm import tqdm


@dataclass
class AudioSample:
    """Single audio sample"""
    audio_path: str
    text: str
    speaker_id: str = "default"
    language: str = "en"
    duration: float = 0.0


class DatasetPreparer:
    """
    Prepares and converts TTS datasets to unified training format
    
    Target format (LJSpeech-style):
    - wavs/ folder with numbered WAV files
    - metadata.csv with filename|text
    - All audio resampled to 22050Hz mono
    """
    
    TARGET_SAMPLE_RATE = 22050
    
    def __init__(self, output_dir: str = "D:/voice cloning/training_data"):
        """
        Initialize dataset preparer
        
        Args:
            output_dir: Output directory for prepared dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(exist_ok=True)
        
        self.samples: List[AudioSample] = []
        self.speakers: Dict[str, Dict] = {}
        
        print(f"📂 Dataset Preparer initialized")
        print(f"   Output: {self.output_dir}")
        
    def add_ljspeech(self, dataset_path: str) -> int:
        """
        Add LJSpeech format dataset
        
        Expected structure:
        dataset/
        ├── wavs/
        │   ├── LJ001-0001.wav
        │   └── ...
        └── metadata.csv (filename|raw_text|normalized_text)
        """
        dataset_path = Path(dataset_path)
        metadata_file = dataset_path / "metadata.csv"
        wavs_path = dataset_path / "wavs"
        
        if not metadata_file.exists():
            # Try alternate location
            metadata_file = dataset_path / "LJSpeech-1.1" / "metadata.csv"
            wavs_path = dataset_path / "LJSpeech-1.1" / "wavs"
        
        if not metadata_file.exists():
            print(f"❌ metadata.csv not found in {dataset_path}")
            return 0
        
        print(f"📥 Loading LJSpeech from {dataset_path}")
        
        count = 0
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    filename = row[0]
                    text = row[-1]  # Use normalized text if available
                    
                    audio_file = wavs_path / f"{filename}.wav"
                    if audio_file.exists():
                        self.samples.append(AudioSample(
                            audio_path=str(audio_file),
                            text=text,
                            speaker_id="ljspeech",
                            language="en"
                        ))
                        count += 1
        
        self.speakers["ljspeech"] = {"name": "LJSpeech", "language": "en"}
        print(f"   ✅ Loaded {count} samples")
        return count
    
    def add_vctk(self, dataset_path: str, max_speakers: int = 20) -> int:
        """
        Add VCTK format dataset
        
        Expected structure:
        dataset/
        ├── wav48_silence_trimmed/
        │   ├── p225/
        │   │   ├── p225_001_mic1.flac
        │   │   └── ...
        │   └── ...
        └── txt/
            ├── p225/
            │   ├── p225_001.txt
            │   └── ...
            └── ...
        """
        dataset_path = Path(dataset_path)
        
        # Find audio directory
        audio_dirs = ["wav48_silence_trimmed", "wav48", "wav"]
        audio_path = None
        for d in audio_dirs:
            if (dataset_path / d).exists():
                audio_path = dataset_path / d
                break
        
        if audio_path is None:
            print(f"❌ Audio directory not found in {dataset_path}")
            return 0
        
        txt_path = dataset_path / "txt"
        
        print(f"📥 Loading VCTK from {dataset_path}")
        print(f"   Audio: {audio_path}")
        print(f"   Max speakers: {max_speakers}")
        
        count = 0
        speaker_dirs = sorted(list(audio_path.iterdir()))[:max_speakers]
        
        for speaker_dir in speaker_dirs:
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            txt_speaker_dir = txt_path / speaker_id
            
            for audio_file in speaker_dir.glob("*_mic1.*"):
                # Get base name (without mic and extension)
                base_name = audio_file.stem.replace("_mic1", "")
                
                # Find text file
                txt_file = txt_speaker_dir / f"{base_name}.txt"
                if not txt_file.exists():
                    continue
                
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                self.samples.append(AudioSample(
                    audio_path=str(audio_file),
                    text=text,
                    speaker_id=speaker_id,
                    language="en"
                ))
                count += 1
            
            self.speakers[speaker_id] = {"name": speaker_id, "language": "en"}
        
        print(f"   ✅ Loaded {count} samples from {len(speaker_dirs)} speakers")
        return count
    
    def add_common_voice(self, dataset_path: str, language: str = "en", max_samples: int = 10000) -> int:
        """
        Add Common Voice format dataset
        
        Can be HuggingFace format or TSV format
        """
        dataset_path = Path(dataset_path)
        
        # Check for HuggingFace arrow format
        if (dataset_path / "dataset_dict.json").exists():
            return self._add_common_voice_hf(dataset_path, language, max_samples)
        
        # Check for TSV format
        tsv_files = list(dataset_path.glob("*.tsv"))
        if tsv_files:
            return self._add_common_voice_tsv(dataset_path, language, max_samples)
        
        print(f"❌ Unknown Common Voice format in {dataset_path}")
        return 0
    
    def _add_common_voice_hf(self, dataset_path: Path, language: str, max_samples: int) -> int:
        """Add Common Voice from HuggingFace format"""
        try:
            from datasets import load_from_disk
            
            print(f"📥 Loading Common Voice (HF) from {dataset_path}")
            
            dataset = load_from_disk(str(dataset_path))
            
            # Get train split
            if "train" in dataset:
                data = dataset["train"]
            else:
                data = dataset
            
            count = 0
            for i, sample in enumerate(data):
                if count >= max_samples:
                    break
                
                audio_path = sample.get("path") or sample.get("audio", {}).get("path")
                text = sample.get("sentence") or sample.get("text", "")
                speaker = sample.get("client_id", f"cv_{i}")
                
                if audio_path and text:
                    self.samples.append(AudioSample(
                        audio_path=str(audio_path),
                        text=text,
                        speaker_id=speaker[:10],  # Truncate long IDs
                        language=language
                    ))
                    count += 1
            
            print(f"   ✅ Loaded {count} samples")
            return count
            
        except Exception as e:
            print(f"❌ Failed to load HF dataset: {e}")
            return 0
    
    def _add_common_voice_tsv(self, dataset_path: Path, language: str, max_samples: int) -> int:
        """Add Common Voice from TSV format"""
        print(f"📥 Loading Common Voice (TSV) from {dataset_path}")
        
        # Find clips directory
        clips_dir = dataset_path / "clips"
        if not clips_dir.exists():
            clips_dir = dataset_path
        
        # Find validated.tsv or train.tsv
        tsv_file = None
        for name in ["validated.tsv", "train.tsv", "other.tsv"]:
            if (dataset_path / name).exists():
                tsv_file = dataset_path / name
                break
        
        if not tsv_file:
            print(f"❌ No TSV file found")
            return 0
        
        count = 0
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if count >= max_samples:
                    break
                
                audio_path = clips_dir / row.get("path", "")
                text = row.get("sentence", "")
                speaker = row.get("client_id", "unknown")
                
                if audio_path.exists() and text:
                    self.samples.append(AudioSample(
                        audio_path=str(audio_path),
                        text=text,
                        speaker_id=speaker[:10],
                        language=language
                    ))
                    count += 1
        
        print(f"   ✅ Loaded {count} samples")
        return count
    
    def add_custom_folder(
        self,
        audio_dir: str,
        transcripts_file: Optional[str] = None,
        speaker_id: str = "custom",
        language: str = "en"
    ) -> int:
        """
        Add custom audio folder with optional transcripts
        
        Args:
            audio_dir: Directory with audio files
            transcripts_file: Optional CSV/TXT with filename|text
            speaker_id: Speaker identifier
            language: Language code
        """
        audio_dir = Path(audio_dir)
        
        print(f"📥 Loading custom folder: {audio_dir}")
        
        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"❌ No audio files found")
            return 0
        
        # Load transcripts if provided
        transcripts = {}
        if transcripts_file and Path(transcripts_file).exists():
            with open(transcripts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        transcripts[parts[0]] = parts[1]
        
        count = 0
        for audio_file in audio_files:
            # Get transcript
            text = transcripts.get(audio_file.stem, "")
            
            # If no transcript, use filename as placeholder
            if not text:
                text = audio_file.stem.replace("_", " ").replace("-", " ")
            
            self.samples.append(AudioSample(
                audio_path=str(audio_file),
                text=text,
                speaker_id=speaker_id,
                language=language
            ))
            count += 1
        
        self.speakers[speaker_id] = {"name": speaker_id, "language": language}
        print(f"   ✅ Loaded {count} audio files")
        
        if not transcripts:
            print(f"   ⚠️ No transcripts provided - using filenames as text")
            print(f"   💡 Create a transcripts.csv with: filename|text")
        
        return count
    
    def prepare(
        self,
        split_ratio: float = 0.95,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        shuffle: bool = True
    ) -> Tuple[Path, Path]:
        """
        Prepare final dataset with train/val split
        
        Args:
            split_ratio: Train/validation split (0.95 = 95% train)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            shuffle: Shuffle samples before split
            
        Returns:
            Tuple of (train_metadata_path, val_metadata_path)
        """
        if not self.samples:
            raise ValueError("No samples loaded! Add datasets first.")
        
        print(f"\n{'='*60}")
        print(f"📦 Preparing dataset: {len(self.samples)} samples")
        print(f"   Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(self.samples)
        
        # Process samples
        processed = []
        
        for i, sample in enumerate(tqdm(self.samples, desc="Processing")):
            try:
                # Load audio
                audio, sr = ta.load(sample.audio_path)
                
                # Convert to mono
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                
                # Calculate duration
                duration = audio.shape[1] / sr
                
                # Skip if too short or too long
                if duration < min_duration or duration > max_duration:
                    continue
                
                # Resample if needed
                if sr != self.TARGET_SAMPLE_RATE:
                    resampler = ta.transforms.Resample(sr, self.TARGET_SAMPLE_RATE)
                    audio = resampler(audio)
                
                # Generate output filename
                output_filename = f"{i:06d}.wav"
                output_path = self.wavs_dir / output_filename
                
                # Save
                ta.save(output_path, audio, self.TARGET_SAMPLE_RATE)
                
                processed.append({
                    'filename': output_filename.replace('.wav', ''),
                    'text': sample.text,
                    'speaker': sample.speaker_id,
                    'language': sample.language,
                    'duration': duration
                })
                
            except Exception as e:
                print(f"\n   ⚠️ Failed to process {sample.audio_path}: {e}")
                continue
        
        print(f"\n✅ Processed {len(processed)} / {len(self.samples)} samples")
        
        # Split into train/val
        split_idx = int(len(processed) * split_ratio)
        train_samples = processed[:split_idx]
        val_samples = processed[split_idx:]
        
        # Write metadata files
        train_meta = self.output_dir / "metadata_train.csv"
        val_meta = self.output_dir / "metadata_val.csv"
        
        def write_metadata(samples, path):
            with open(path, 'w', encoding='utf-8', newline='') as f:
                for s in samples:
                    f.write(f"{s['filename']}|{s['text']}\n")
        
        write_metadata(train_samples, train_meta)
        write_metadata(val_samples, val_meta)
        
        # Also write combined metadata
        combined_meta = self.output_dir / "metadata.csv"
        write_metadata(processed, combined_meta)
        
        # Write speakers info
        speakers_file = self.output_dir / "speakers.json"
        with open(speakers_file, 'w') as f:
            json.dump(self.speakers, f, indent=2)
        
        # Write dataset info
        info = {
            'total_samples': len(processed),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'speakers': len(self.speakers),
            'total_duration_hours': sum(s['duration'] for s in processed) / 3600,
            'sample_rate': self.TARGET_SAMPLE_RATE,
            'created_at': str(Path.cwd()),
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total: {info['total_samples']} samples")
        print(f"   Train: {info['train_samples']} samples")
        print(f"   Val: {info['val_samples']} samples")
        print(f"   Speakers: {info['speakers']}")
        print(f"   Duration: {info['total_duration_hours']:.2f} hours")
        
        print(f"\n📁 Output Files:")
        print(f"   {train_meta}")
        print(f"   {val_meta}")
        print(f"   {speakers_file}")
        
        return train_meta, val_meta


def main():
    """Example usage"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           📂 DATASET PREPARATION TOOL                            ║
║           Convert TTS Datasets to Training Format                ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    preparer = DatasetPreparer("D:/voice cloning/training_data")
    
    print("\n📋 USAGE EXAMPLES:")
    print("-" * 60)
    
    print("""
# Add LJSpeech dataset:
preparer.add_ljspeech("D:/voice cloning/datasets/ljspeech")

# Add VCTK dataset (first 20 speakers):
preparer.add_vctk("D:/voice cloning/datasets/vctk", max_speakers=20)

# Add Common Voice:
preparer.add_common_voice("D:/voice cloning/datasets/common_voice", "en", max_samples=5000)

# Add your own audio folder:
preparer.add_custom_folder(
    audio_dir="D:/voice cloning/my_recordings",
    transcripts_file="D:/voice cloning/my_recordings/transcripts.csv",
    speaker_id="bhomik",
    language="hi"
)

# Prepare final dataset:
train_meta, val_meta = preparer.prepare(
    split_ratio=0.95,
    min_duration=1.0,
    max_duration=15.0
)
""")


if __name__ == "__main__":
    main()
