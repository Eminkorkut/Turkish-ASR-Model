"""
ASR Dataset module with modern preprocessing and data augmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import glob
import random
from typing import List, Tuple, Optional, Dict, Any

from data.preprocessing import (
    AudioPreprocessor,
    SpecAugment,
    SpeedPerturbation,
    NoisePerturbation,
    TARGET_SAMPLE_RATE
)


class ASRDataset(Dataset):
    """
    ASR Dataset with modern torchaudio preprocessing and augmentation support.
    
    Features:
    - GPU-accelerated feature extraction via torchaudio
    - Speed perturbation
    - Noise injection
    - SpecAugment (applied in training)
    """
    
    def __init__(
        self,
        file_pairs: List[Tuple[str, str]],
        tokenizer,
        n_mel_channels: int = 80,
        augment: bool = False,
        speed_perturb: bool = False,
        noise_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            file_pairs: List of (wav_path, txt_path) tuples
            tokenizer: Text tokenizer
            n_mel_channels: Number of mel filterbanks
            augment: Whether to apply SpecAugment
            speed_perturb: Whether to apply speed perturbation
            noise_dir: Directory with noise files for noise injection
            device: Device for preprocessing ("cpu" or "cuda")
        """
        self.file_pairs = file_pairs
        self.tokenizer = tokenizer
        self.n_mel_channels = n_mel_channels
        self.augment = augment
        self.device = device
        
        # Initialize preprocessor
        # Force CPU for data loading to avoid CUDA initialization errors in workers
        self.preprocessor = AudioPreprocessor(
            n_mels=n_mel_channels,
            normalize=True,
            device="cpu"
        )
        
        # Augmentations
        self.speed_perturb = SpeedPerturbation() if speed_perturb else None
        self.noise_perturb = NoisePerturbation(noise_dir=noise_dir) if noise_dir else None
        self.spec_augment = SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            n_freq_masks=2,
            n_time_masks=2
        ) if augment else None
        
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_path, txt_path = self.file_pairs[idx]
        
        try:
            # 1. Load Audio
            waveform, sr = self.preprocessor.load_audio(wav_path)
            
            # 2. Apply waveform-level augmentations
            if self.training and self.speed_perturb:
                waveform = self.speed_perturb(waveform, sr)
                
            if self.training and self.noise_perturb:
                waveform = self.noise_perturb(waveform, sr)
            
            # 3. Extract Features
            features = self.preprocessor.extract_features(waveform)
            
            # 4. Apply SpecAugment (on features)
            if self.training and self.spec_augment:
                features = self.spec_augment(features)
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))
        
        # 5. Load and tokenize transcript
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        except FileNotFoundError:
            transcript = ""
            
        target = torch.LongTensor(self.tokenizer.encode(transcript))
        
        return features.cpu(), target
    
    @property
    def training(self) -> bool:
        """Check if dataset is in training mode (for augmentation)."""
        return self.augment


class BucketingSampler(Sampler):
    """
    Bucket sampler that groups samples by length for efficient batching.
    Reduces padding overhead by ensuring similar-length samples are batched together.
    """
    
    def __init__(
        self,
        data_source: ASRDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Pre-compute lengths (file sizes as proxy for audio length)
        self.lengths = []
        for wav_path, _ in data_source.file_pairs:
            try:
                self.lengths.append(os.path.getsize(wav_path))
            except OSError:
                self.lengths.append(0)
                
    def __iter__(self):
        # Sort indices by length
        indices = list(range(len(self.data_source)))
        indices = sorted(indices, key=lambda i: self.lengths[i])
        
        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
                
        # Shuffle batches (not within batches)
        if self.shuffle:
            random.shuffle(batches)
            
        # Flatten
        for batch in batches:
            yield from batch
            
    def __len__(self) -> int:
        if self.drop_last:
            return (len(self.data_source) // self.batch_size) * self.batch_size
        return len(self.data_source)


def find_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    Find all .wav files and their corresponding .txt transcripts.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of (wav_path, txt_path) tuples
    """
    wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    pairs = []
    
    for wav_path in wav_files:
        txt_path = wav_path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            pairs.append((wav_path, txt_path))
            
    return pairs


def create_datasets(
    config,
    tokenizer,
    augment_train: bool = True
) -> Tuple[Optional[ASRDataset], Optional[ASRDataset], Optional[ASRDataset]]:
    """
    Create train, validation, and test datasets from configuration.
    
    Args:
        config: Configuration object with data paths
        tokenizer: Text tokenizer
        augment_train: Whether to apply augmentation to training set
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    train_pairs = []
    valid_pairs = []
    test_pairs = []
    
    # 1. Use specific paths if provided
    if config.train_path:
        print(f"Loading training data from: {config.train_path}")
        train_pairs = find_files(config.train_path)
    
    if config.valid_path:
        print(f"Loading validation data from: {config.valid_path}")
        valid_pairs = find_files(config.valid_path)
        
    if config.test_path:
        print(f"Loading test data from: {config.test_path}")
        test_pairs = find_files(config.test_path)
        
    # 2. If no specific paths, split from main data_path
    if not train_pairs and config.data_path:
        print(f"Loading and splitting data from: {config.data_path}")
        all_pairs = find_files(config.data_path)
        total_count = len(all_pairs)
        
        if total_count == 0:
            raise ValueError(f"No data found in: {config.data_path}")
            
        # Calculate split sizes
        test_size = int(total_count * config.test_split)
        valid_size = int(total_count * config.val_split)
        train_size = total_count - test_size - valid_size
        
        # Shuffle with fixed seed for reproducibility
        random.seed(config.seed)
        random.shuffle(all_pairs)
        
        train_pairs = all_pairs[:train_size]
        valid_pairs = all_pairs[train_size:train_size + valid_size]
        test_pairs = all_pairs[train_size + valid_size:]
        
        print(f"Split: Train={len(train_pairs)}, Valid={len(valid_pairs)}, Test={len(test_pairs)}")

    # Get noise directory if specified
    noise_dir = getattr(config, 'noise_dir', None)
    
    # Create Dataset objects
    train_dataset = ASRDataset(
        train_pairs, 
        tokenizer, 
        config.n_mel_channels,
        augment=augment_train,
        speed_perturb=augment_train,
        noise_dir=noise_dir
    ) if train_pairs else None
    
    valid_dataset = ASRDataset(
        valid_pairs, 
        tokenizer, 
        config.n_mel_channels,
        augment=False
    ) if valid_pairs else None
    
    test_dataset = ASRDataset(
        test_pairs, 
        tokenizer, 
        config.n_mel_channels,
        augment=False
    ) if test_pairs else None
    
    return train_dataset, valid_dataset, test_dataset


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Collate function for DataLoader.
    Pads features and targets to batch max length.
    
    Args:
        batch: List of (features, target) tuples
        
    Returns:
        Tuple of (features_padded, targets_padded, input_lengths, target_lengths)
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None and item[0] is not None]
    
    if len(batch) == 0:
        return None, None, None, None
        
    features, targets = zip(*batch)
    
    # Compute lengths before padding
    input_lengths = torch.LongTensor([f.size(0) for f in features])
    target_lengths = torch.LongTensor([len(t) for t in targets])
    
    # Pad sequences
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return features_padded, targets_padded, input_lengths, target_lengths
