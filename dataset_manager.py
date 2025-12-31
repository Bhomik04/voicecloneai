"""
Dataset Manager - Voice Cloning Dataset Downloader & Manager
=============================================================

Manages TTS/Voice Cloning datasets from multiple sources:
- Kaggle (LJSpeech, VCTK, Hindi TTS, Common Voice)
- HuggingFace (LibriTTS, LJSpeech, VCTK)
- Direct downloads (LJSpeech, VCTK)

All datasets stored on D: drive to save C: drive space.
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure paths to D: drive
DATASET_DIR = Path("D:/voice cloning/datasets")
CACHE_DIR = Path("D:/voice cloning/models_cache")

# Ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetInfo:
    """Information about a TTS dataset"""
    name: str
    description: str
    source: str  # 'kaggle', 'huggingface', 'direct'
    source_id: str  # Kaggle slug, HF repo, or URL
    size_gb: float
    languages: List[str]
    speakers: int
    hours: float
    format: str  # 'ljspeech', 'vctk', 'common_voice'
    license: str


# Available datasets catalog
DATASETS_CATALOG = {
    # ============== ENGLISH DATASETS ==============
    "ljspeech": DatasetInfo(
        name="LJSpeech",
        description="Single female speaker, high-quality audiobook recordings. Best for single-speaker TTS.",
        source="huggingface",
        source_id="keithito/lj_speech",
        size_gb=2.6,
        languages=["en"],
        speakers=1,
        hours=24,
        format="ljspeech",
        license="Public Domain",
    ),
    
    "vctk": DatasetInfo(
        name="VCTK Corpus",
        description="110 English speakers with various accents. Great for multi-speaker voice cloning.",
        source="kaggle",
        source_id="mfekadu/english-multispeaker-corpus-for-voice-cloning",
        size_gb=10.9,
        languages=["en"],
        speakers=110,
        hours=44,
        format="vctk",
        license="CC BY 4.0",
    ),
    
    "libritts": DatasetInfo(
        name="LibriTTS",
        description="Large multi-speaker corpus from audiobooks. Clean subset recommended.",
        source="huggingface",
        source_id="openslr/librispeech_asr",
        size_gb=60.0,
        languages=["en"],
        speakers=2456,
        hours=585,
        format="libritts",
        license="CC BY 4.0",
    ),
    
    "libritts_clean": DatasetInfo(
        name="LibriTTS Clean-100",
        description="Clean subset of LibriTTS. 100 hours, high quality.",
        source="direct",
        source_id="https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        size_gb=5.6,
        languages=["en"],
        speakers=251,
        hours=100,
        format="libritts",
        license="CC BY 4.0",
    ),
    
    "common_voice_en": DatasetInfo(
        name="Common Voice English",
        description="Mozilla's crowdsourced voice dataset. Diverse speakers.",
        source="huggingface",
        source_id="mozilla-foundation/common_voice_16_0",
        size_gb=85.0,
        languages=["en"],
        speakers=75000,
        hours=2500,
        format="common_voice",
        license="CC0",
    ),
    
    # ============== HINDI DATASETS ==============
    "hindi_tts_kaggle": DatasetInfo(
        name="Hindi Speech TTS (Orpheus)",
        description="Hindi TTS dataset for Orpheus model fine-tuning. 7GB.",
        source="kaggle",
        source_id="ashutoshanand1/hindi-speech-tts-dataset-orpheus-fine-tuning",
        size_gb=7.0,
        languages=["hi"],
        speakers=10,
        hours=50,
        format="ljspeech",
        license="CC BY 4.0",
    ),
    
    "indicvoices": DatasetInfo(
        name="IndicVoices",
        description="Multi-speaker Hindi and Indian language TTS dataset.",
        source="huggingface",
        source_id="ai4bharat/indicvoices",
        size_gb=15.0,
        languages=["hi", "ta", "te", "mr", "bn"],
        speakers=1000,
        hours=200,
        format="custom",
        license="CC BY 4.0",
    ),
    
    "common_voice_hi": DatasetInfo(
        name="Common Voice Hindi",
        description="Mozilla's Hindi voice dataset.",
        source="huggingface",
        source_id="mozilla-foundation/common_voice_16_0",
        size_gb=3.0,
        languages=["hi"],
        speakers=5000,
        hours=100,
        format="common_voice",
        license="CC0",
    ),
    
    # ============== SMALL/QUICK DATASETS ==============
    "cmu_arctic": DatasetInfo(
        name="CMU ARCTIC",
        description="Small but high-quality dataset. Good for quick experiments.",
        source="kaggle",
        source_id="bryanpark/cmu-arctic-speech-corpus",
        size_gb=1.2,
        languages=["en"],
        speakers=7,
        hours=7,
        format="ljspeech",
        license="Free",
    ),
}


class DatasetManager:
    """
    Manages TTS datasets for voice cloning fine-tuning
    
    Features:
    - Download from Kaggle, HuggingFace, or direct URLs
    - Automatic format detection and conversion
    - Metadata management
    - Storage on D: drive
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize dataset manager
        
        Args:
            base_dir: Base directory for datasets (default: D:/voice cloning/datasets)
        """
        self.base_dir = base_dir or DATASET_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.base_dir / "datasets_metadata.json"
        self.downloaded_datasets = self._load_metadata()
        
        print(f"📂 Dataset Manager initialized")
        print(f"   Base directory: {self.base_dir}")
        print(f"   Downloaded: {len(self.downloaded_datasets)} datasets")
        
    def _load_metadata(self) -> Dict:
        """Load downloaded datasets metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save downloaded datasets metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.downloaded_datasets, f, indent=2)
            
    def list_available(self) -> List[DatasetInfo]:
        """List all available datasets in catalog"""
        return list(DATASETS_CATALOG.values())
    
    def list_downloaded(self) -> List[str]:
        """List downloaded datasets"""
        return list(self.downloaded_datasets.keys())
    
    def get_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset"""
        return DATASETS_CATALOG.get(dataset_name.lower())
    
    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """Get path to downloaded dataset"""
        if dataset_name in self.downloaded_datasets:
            return Path(self.downloaded_datasets[dataset_name]['path'])
        return None
    
    def download_kaggle(self, dataset_slug: str, dest_dir: Path) -> bool:
        """
        Download dataset from Kaggle
        
        Requires: 
        - pip install kaggle
        - ~/.kaggle/kaggle.json with API credentials
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print(f"📥 Downloading from Kaggle: {dataset_slug}")
            
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_files(
                dataset_slug,
                path=str(dest_dir),
                unzip=True
            )
            
            print(f"   ✅ Downloaded to {dest_dir}")
            return True
            
        except ImportError:
            print("❌ Kaggle not installed. Run: pip install kaggle")
            print("   Then configure: https://www.kaggle.com/docs/api")
            return False
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            return False
    
    def download_huggingface(self, repo_id: str, dest_dir: Path, subset: str = None) -> bool:
        """
        Download dataset from HuggingFace
        
        Args:
            repo_id: HuggingFace dataset ID
            dest_dir: Destination directory
            subset: Optional subset (e.g., 'hi' for Hindi)
        """
        try:
            from datasets import load_dataset
            
            print(f"📥 Downloading from HuggingFace: {repo_id}")
            if subset:
                print(f"   Subset: {subset}")
            
            # Load dataset (will cache to HF_HOME on D: drive)
            if subset:
                dataset = load_dataset(repo_id, subset, trust_remote_code=True)
            else:
                dataset = load_dataset(repo_id, trust_remote_code=True)
            
            # Save to our directory
            dataset.save_to_disk(str(dest_dir))
            
            print(f"   ✅ Downloaded to {dest_dir}")
            return True
            
        except ImportError:
            print("❌ datasets library not installed. Run: pip install datasets")
            return False
        except Exception as e:
            print(f"❌ HuggingFace download failed: {e}")
            return False
    
    def download_direct(self, url: str, dest_dir: Path) -> bool:
        """
        Download dataset from direct URL
        
        Handles: .zip, .tar.gz, .tar.bz2
        """
        import urllib.request
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        filename = url.split('/')[-1]
        download_path = dest_dir / filename
        
        print(f"📥 Downloading: {filename}")
        print(f"   URL: {url[:60]}...")
        
        try:
            # Download with progress
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r   Progress: {percent}%")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, download_path, progress_hook)
            print()  # Newline after progress
            
            # Extract if archive
            if filename.endswith('.zip'):
                print(f"   📦 Extracting ZIP...")
                with zipfile.ZipFile(download_path, 'r') as zf:
                    zf.extractall(dest_dir)
                download_path.unlink()  # Remove ZIP after extraction
                
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                print(f"   📦 Extracting TAR.GZ...")
                with tarfile.open(download_path, 'r:gz') as tf:
                    tf.extractall(dest_dir)
                download_path.unlink()
                
            elif filename.endswith('.tar.bz2'):
                print(f"   📦 Extracting TAR.BZ2...")
                with tarfile.open(download_path, 'r:bz2') as tf:
                    tf.extractall(dest_dir)
                download_path.unlink()
            
            print(f"   ✅ Downloaded to {dest_dir}")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False
    
    def download(self, dataset_name: str, force: bool = False) -> Optional[Path]:
        """
        Download a dataset by name
        
        Args:
            dataset_name: Name from catalog (e.g., 'ljspeech', 'vctk')
            force: Force re-download even if exists
            
        Returns:
            Path to downloaded dataset, or None if failed
        """
        dataset_name = dataset_name.lower()
        
        # Check if already downloaded
        if dataset_name in self.downloaded_datasets and not force:
            existing_path = Path(self.downloaded_datasets[dataset_name]['path'])
            if existing_path.exists():
                print(f"✅ {dataset_name} already downloaded at {existing_path}")
                return existing_path
        
        # Get dataset info
        info = DATASETS_CATALOG.get(dataset_name)
        if info is None:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"   Available: {', '.join(DATASETS_CATALOG.keys())}")
            return None
        
        # Create destination directory
        dest_dir = self.base_dir / dataset_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {info.name}")
        print(f"   Description: {info.description}")
        print(f"   Size: {info.size_gb:.1f} GB")
        print(f"   Languages: {', '.join(info.languages)}")
        print(f"   Speakers: {info.speakers}")
        print(f"   Hours: {info.hours}")
        print(f"{'='*60}\n")
        
        # Download based on source
        success = False
        
        if info.source == "kaggle":
            success = self.download_kaggle(info.source_id, dest_dir)
            
        elif info.source == "huggingface":
            # For Common Voice, need to specify language subset
            subset = None
            if "common_voice" in info.source_id:
                subset = info.languages[0]
            success = self.download_huggingface(info.source_id, dest_dir, subset)
            
        elif info.source == "direct":
            success = self.download_direct(info.source_id, dest_dir)
        
        if success:
            # Update metadata
            self.downloaded_datasets[dataset_name] = {
                'path': str(dest_dir),
                'downloaded_at': datetime.now().isoformat(),
                'info': {
                    'name': info.name,
                    'languages': info.languages,
                    'speakers': info.speakers,
                    'hours': info.hours,
                    'format': info.format,
                }
            }
            self._save_metadata()
            
            return dest_dir
        
        return None
    
    def show_catalog(self):
        """Pretty print available datasets"""
        print("\n" + "="*80)
        print(" 📚 AVAILABLE TTS DATASETS FOR VOICE CLONING")
        print("="*80)
        
        # Group by language
        english = []
        hindi = []
        other = []
        
        for name, info in DATASETS_CATALOG.items():
            if 'en' in info.languages and len(info.languages) == 1:
                english.append((name, info))
            elif 'hi' in info.languages:
                hindi.append((name, info))
            else:
                other.append((name, info))
        
        def print_dataset(name, info, downloaded):
            status = "✅" if name in downloaded else "⬜"
            print(f"\n  {status} {name}")
            print(f"     📝 {info.description[:60]}...")
            print(f"     📊 {info.size_gb:.1f}GB | {info.speakers} speakers | {info.hours}h")
            print(f"     🔗 Source: {info.source} ({info.source_id[:40]}...)")
        
        downloaded = self.downloaded_datasets
        
        print("\n🇺🇸 ENGLISH DATASETS:")
        print("-"*40)
        for name, info in english:
            print_dataset(name, info, downloaded)
        
        print("\n\n🇮🇳 HINDI DATASETS:")
        print("-"*40)
        for name, info in hindi:
            print_dataset(name, info, downloaded)
        
        if other:
            print("\n\n🌍 MULTILINGUAL DATASETS:")
            print("-"*40)
            for name, info in other:
                print_dataset(name, info, downloaded)
        
        print("\n" + "="*80)
        print(" 💡 Usage: manager.download('ljspeech') or manager.download('hindi_tts_kaggle')")
        print("="*80 + "\n")


def setup_kaggle_credentials():
    """Help user set up Kaggle API credentials"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  🔐 KAGGLE API SETUP                             ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Go to: https://www.kaggle.com/settings                       ║
║  2. Scroll to "API" section                                      ║
║  3. Click "Create New Token"                                     ║
║  4. Save kaggle.json to:                                         ║
║     - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json        ║
║     - Linux/Mac: ~/.kaggle/kaggle.json                           ║
║  5. Run: pip install kaggle                                      ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    """Interactive dataset manager"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           📂 VOICE CLONING DATASET MANAGER                       ║
║           Download & Manage TTS Training Datasets                ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    manager = DatasetManager()
    
    while True:
        print("\n📋 OPTIONS:")
        print("  1. Show available datasets")
        print("  2. Download a dataset")
        print("  3. List downloaded datasets")
        print("  4. Setup Kaggle API")
        print("  5. Exit")
        
        choice = input("\n👉 Enter choice (1-5): ").strip()
        
        if choice == "1":
            manager.show_catalog()
            
        elif choice == "2":
            print("\n📝 Available datasets:")
            for name in DATASETS_CATALOG.keys():
                print(f"   - {name}")
            dataset = input("\n👉 Enter dataset name: ").strip().lower()
            if dataset:
                manager.download(dataset)
                
        elif choice == "3":
            downloaded = manager.list_downloaded()
            if downloaded:
                print("\n✅ Downloaded datasets:")
                for name in downloaded:
                    path = manager.get_dataset_path(name)
                    print(f"   - {name}: {path}")
            else:
                print("\n⚠️ No datasets downloaded yet")
                
        elif choice == "4":
            setup_kaggle_credentials()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
