"""
Advanced Voice Encoder using Microsoft WavLM
Provides 768-dimensional voice embeddings (vs 256D from LSTM)

This is a key component for achieving ElevenLabs-level quality
WavLM was trained on 94k hours of speech data
"""

# Configure model paths to D: drive BEFORE importing transformers
import model_paths

import torch
import torch.nn as nn
import torchaudio as ta
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from typing import List, Union
import numpy as np


class AdvancedVoiceEncoder:
    """
    WavLM-based voice encoder for high-quality speaker embeddings
    
    Key advantages over LSTM encoder:
    - 768D embeddings (vs 256D) = 3x more speaker information
    - Trained on massive dataset (94k hours)
    - Better speaker discrimination
    - More robust to noise and variations
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "microsoft/wavlm-large",
        use_cache: bool = True,
    ):
        """
        Initialize WavLM encoder
        
        Args:
            device: 'cuda' or 'cpu'
            model_name: HuggingFace model name
            use_cache: Cache loaded model in memory
        """
        self.device = device
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Model components (lazy loaded)
        self.model = None
        self.feature_extractor = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def _load_model(self):
        """Lazy load the WavLM model (only when first needed)"""
        if self.model is not None:
            return
            
        print(f"🔄 Loading WavLM model: {self.model_name}")
        print("   (First time will download ~1.5GB model...)")
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name
        )
        
        # Load model
        self.model = WavLMModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        print("✅ WavLM model loaded successfully")
        
    def encode_voice(
        self,
        audio_path: str,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a single voice sample to 768D embedding
        
        Args:
            audio_path: Path to audio file (.wav)
            return_numpy: Return numpy array instead of torch tensor
            
        Returns:
            Voice embedding [1, 768] or [768]
        """
        # Check cache
        if self.use_cache and audio_path in self.embedding_cache:
            emb = self.embedding_cache[audio_path]
            return emb.cpu().numpy() if return_numpy else emb
            
        # Load model if needed
        self._load_model()
        
        # Load and preprocess audio
        audio, sr = ta.load(audio_path)
        
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        # Resample to 16kHz (WavLM requirement)
        if sr != 16000:
            resampler = ta.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            
        # Extract features
        inputs = self.feature_extractor(
            audio.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling over time dimension
            # outputs.last_hidden_state: [batch, time, 768]
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
            
        # Cache result
        if self.use_cache:
            self.embedding_cache[audio_path] = embeddings
            
        if return_numpy:
            return embeddings.cpu().numpy()
        else:
            return embeddings
    
    def encode_multiple_samples(
        self,
        audio_paths: List[str],
        average: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode multiple voice samples and optionally average them
        
        This is the recommended approach for creating voice profiles
        Multiple samples provide more robust speaker representation
        
        Args:
            audio_paths: List of paths to audio files
            average: Average the embeddings (recommended)
            return_numpy: Return numpy array
            
        Returns:
            If average=True: [1, 768] averaged embedding
            If average=False: [num_samples, 768] stacked embeddings
        """
        print(f"🎙️ Encoding {len(audio_paths)} voice samples with WavLM...")
        
        embeddings = []
        
        for i, path in enumerate(audio_paths, 1):
            print(f"   [{i}/{len(audio_paths)}] {path.split('/')[-1]}")
            emb = self.encode_voice(path, return_numpy=False)
            embeddings.append(emb)
            
        # Stack embeddings
        stacked = torch.cat(embeddings, dim=0)  # [num_samples, 768]
        
        if average:
            # Average across samples
            result = stacked.mean(dim=0, keepdim=True)  # [1, 768]
            print(f"✅ Averaged embedding shape: {result.shape}")
        else:
            result = stacked
            print(f"✅ Stacked embeddings shape: {result.shape}")
            
        if return_numpy:
            return result.cpu().numpy()
        else:
            return result
    
    def compare_voices(
        self,
        audio_path1: str,
        audio_path2: str,
    ) -> float:
        """
        Compare two voice samples using cosine similarity
        
        Returns similarity score (0-1, higher = more similar)
        """
        emb1 = self.encode_voice(audio_path1)
        emb2 = self.encode_voice(audio_path2)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1, emb2, dim=-1
        )
        
        return similarity.item()
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.embedding_cache.clear()
        print("✅ Embedding cache cleared")


def test_advanced_encoder():
    """Test the WavLM encoder"""
    import os
    import glob
    
    print("🎙️ Testing Advanced Voice Encoder (WavLM)\n")
    
    # Find test samples
    profile_dir = "voice_profiles/pritam/samples"
    
    if not os.path.exists(profile_dir):
        print(f"❌ Profile directory not found: {profile_dir}")
        print("   Create a profile first or update the test path")
        return
        
    samples = glob.glob(os.path.join(profile_dir, "*.wav"))
    
    if len(samples) == 0:
        print(f"❌ No .wav files found in {profile_dir}")
        return
        
    print(f"📂 Found {len(samples)} voice samples")
    for i, s in enumerate(samples[:5], 1):  # Show first 5
        print(f"   {i}. {os.path.basename(s)}")
    
    if len(samples) > 5:
        print(f"   ... and {len(samples) - 5} more")
        
    # Create encoder
    print("\n⚡ Initializing WavLM encoder...")
    encoder = AdvancedVoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test single sample encoding
    print(f"\n🔬 Test 1: Encode single sample")
    emb1 = encoder.encode_voice(samples[0])
    print(f"   Embedding shape: {emb1.shape}")
    print(f"   Embedding mean: {emb1.mean():.4f}")
    print(f"   Embedding std: {emb1.std():.4f}")
    
    # Test multiple sample encoding
    print(f"\n🔬 Test 2: Encode multiple samples (averaging)")
    test_samples = samples[:min(3, len(samples))]
    emb_avg = encoder.encode_multiple_samples(test_samples, average=True)
    print(f"   Averaged embedding shape: {emb_avg.shape}")
    
    # Test voice comparison
    if len(samples) >= 2:
        print(f"\n🔬 Test 3: Compare two voice samples")
        similarity = encoder.compare_voices(samples[0], samples[1])
        print(f"   Similarity: {similarity:.4f}")
        print(f"   (Should be high ~0.8-0.95 for same speaker)")
        
    # Compare with different samples if available
    if len(samples) >= 3:
        sim12 = encoder.compare_voices(samples[0], samples[1])
        sim13 = encoder.compare_voices(samples[0], samples[2])
        
        print(f"\n📊 Pairwise similarities:")
        print(f"   Sample 1 vs 2: {sim12:.4f}")
        print(f"   Sample 1 vs 3: {sim13:.4f}")
        print(f"   Average: {(sim12 + sim13) / 2:.4f}")
        
    print("\n✅ WavLM encoder test complete!")
    print("\n💡 Next steps:")
    print("   1. This encoder provides 768D embeddings (vs 256D LSTM)")
    print("   2. Integrate into MyVoiceClone for better voice quality")
    print("   3. Compare generated audio with/without WavLM")


if __name__ == "__main__":
    test_advanced_encoder()
