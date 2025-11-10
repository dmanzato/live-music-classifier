import os
from typing import List, Optional, Tuple
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from transforms.audio import get_mel_transform, wav_to_logmel

class GTZAN(torch.utils.data.Dataset):
    """GTZAN dataset loader for music genre classification.
    
    Expects directory layout like:
      root/
        blues/
          *.wav
        classical/
          *.wav
        country/
          *.wav
        disco/
          *.wav
        hiphop/
          *.wav
        jazz/
          *.wav
        metal/
          *.wav
        pop/
          *.wav
        reggae/
          *.wav
        rock/
          *.wav
    
    Args:
        root: dataset root directory
        split: 'train', 'val', 'test', or 'all' to include all files
        train_ratio: proportion of data for training (default: 0.8)
        val_ratio: proportion of data for validation (default: 0.1)
        target_sr: resample audio to this sample rate
        duration: seconds to which clips are trimmed/padded (default: 30.0, full GTZAN clip length)
        n_mels, n_fft, hop_length: spectrogram params
        augment: optional callable (e.g., SpecAugment) applied on spectrogram tensors [B,1,H,W] or per-sample [1,1,H,W]
        seed: random seed for reproducible train/val/test splits
    """
    # Standard GTZAN genres
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def __init__(
        self,
        root: str,
        split: str = 'all',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        target_sr: int = 16_000,
        duration: float = 30.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        
        if not self.root.exists():
            raise FileNotFoundError(f"GTZAN root directory not found: {root}")
        
        # Collect all audio files by genre
        self.genre_to_files = {}
        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.exists():
                raise FileNotFoundError(f"Genre directory not found: {genre_dir}")
            
            wav_files = sorted(list(genre_dir.glob("*.wav")))
            if not wav_files:
                raise ValueError(f"No WAV files found in {genre_dir}")
            
            self.genre_to_files[genre] = wav_files
        
        # Create file list with labels, validating files can be loaded
        self.filepaths = []
        self.labels = []
        
        for genre_idx, genre in enumerate(self.GENRES):
            for filepath in self.genre_to_files[genre]:
                # Validate file can be loaded (skip corrupted files)
                if self._validate_audio_file(filepath):
                    self.filepaths.append(filepath)
                    self.labels.append(genre_idx)
                else:
                    # Log warning but continue
                    import warnings
                    warnings.warn(f"Skipping corrupted or unreadable file: {filepath}", UserWarning)
        
        # Split dataset if needed
        if split != 'all':
            import random
            random.seed(seed)
            
            # Create indices for each genre to ensure balanced splits
            genre_indices = {i: [] for i in range(len(self.GENRES))}
            for idx, label in enumerate(self.labels):
                genre_indices[label].append(idx)
            
            # Shuffle indices within each genre
            for genre_idx in genre_indices:
                random.shuffle(genre_indices[genre_idx])
            
            # Calculate split sizes per genre
            train_indices = []
            val_indices = []
            test_indices = []
            
            for genre_idx in range(len(self.GENRES)):
                genre_list = genre_indices[genre_idx]
                n = len(genre_list)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_indices.extend(genre_list[:n_train])
                val_indices.extend(genre_list[n_train:n_train + n_val])
                test_indices.extend(genre_list[n_train + n_val:])
            
            # Select indices based on split
            if split == 'train':
                selected_indices = train_indices
            elif split == 'val':
                selected_indices = val_indices
            elif split == 'test':
                selected_indices = test_indices
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'")
            
            # Filter filepaths and labels
            self.filepaths = [self.filepaths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]
        
        # Create class mappings
        self.class_ids = list(range(len(self.GENRES)))
        self.id2idx = {cid: cid for cid in self.class_ids}  # For compatibility
        self.idx2name = {i: genre for i, genre in enumerate(self.GENRES)}
        
        # Prepare mel transform
        self.mel_t = get_mel_transform(
            sample_rate=self.target_sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        
        # Pre-compute number of samples per clip
        self.num_samples = int(self.duration * self.target_sr)
    
    def __len__(self) -> int:
        return len(self.filepaths)
    
    def _validate_audio_file(self, path: Path) -> bool:
        """
        Validate that an audio file can be loaded.
        Returns True if file is valid, False otherwise.
        """
        if not path.exists():
            return False
        
        # Try to read the file header to check if it's a valid audio file
        try:
            # Quick check: try to read first few bytes
            with open(path, 'rb') as f:
                header = f.read(12)
                # Check for WAV file signature (RIFF...WAVE)
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    return True
                # If not WAV, try to load with soundfile or scipy
                # This is a quick validation - actual loading happens in _load_wave_fixed
                try:
                    # Try a quick read with soundfile
                    sf.read(str(path), frames=1, dtype='float32')
                    return True
                except Exception:
                    # Try scipy as fallback
                    if HAS_SCIPY:
                        try:
                            wavfile.read(str(path))
                            return True
                        except Exception:
                            return False
                    return False
        except Exception:
            return False
    
    def _load_wave_fixed(self, path: Path) -> torch.Tensor:
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Try multiple methods to load the audio file
        wav = None
        sr = None
        
        # Method 1: Try soundfile (preferred, most reliable)
        try:
            wav, sr = sf.read(str(path), always_2d=True, dtype='float32')
            wav = torch.from_numpy(wav).T  # Convert to [C, T] format
        except Exception as e1:
            # Method 2: Try scipy.io.wavfile (fallback for WAV files)
            if HAS_SCIPY:
                try:
                    sr, wav = wavfile.read(str(path))
                    # Convert to float32 and normalize to [-1, 1]
                    if wav.dtype == np.int16:
                        wav = wav.astype(np.float32) / 32768.0
                    elif wav.dtype == np.int32:
                        wav = wav.astype(np.float32) / 2147483648.0
                    else:
                        wav = wav.astype(np.float32)
                    
                    # Ensure 2D shape [C, T]
                    if wav.ndim == 1:
                        wav = wav[np.newaxis, :]  # [1, T]
                    elif wav.ndim == 2:
                        wav = wav.T  # [C, T]
                    
                    wav = torch.from_numpy(wav)
                except Exception as e2:
                    # If both methods fail, raise informative error
                    raise RuntimeError(
                        f"Failed to load audio file: {path}\n"
                        f"soundfile error: {e1}\n"
                        f"scipy.io.wavfile error: {e2}\n"
                        f"Make sure the file exists, is readable, and is a valid WAV file.\n"
                        f"If this persists, the file may be corrupted or in an unsupported format."
                    ) from e2
            else:
                # If scipy is not available, just report the soundfile error
                raise RuntimeError(
                    f"Failed to load audio file with soundfile: {path}\n"
                    f"Error: {e1}\n"
                    f"Make sure the file exists, is readable, and is a valid audio file.\n"
                    f"If this persists, check that soundfile/libsndfile is properly installed.\n"
                    f"Alternatively, install scipy for an additional fallback: pip install scipy"
                ) from e1
        # To mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)
            sr = self.target_sr
        # Pad/trim to fixed duration
        T = wav.size(1)
        if T < self.num_samples:
            pad = self.num_samples - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif T > self.num_samples:
            wav = wav[:, :self.num_samples]
        return wav  # [1, num_samples]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.filepaths[idx]
        y = self.labels[idx]
        
        wav = self._load_wave_fixed(path)  # [1, T]
        log_mel = wav_to_logmel(wav, sr=self.target_sr, mel_transform=self.mel_t)  # [1, n_mels, time]
        
        # Add channel dimension for Conv2d: [1, 1, n_mels, time]
        x = log_mel.unsqueeze(0)
        
        # Optional per-sample augment (e.g., SpecAugment expects [B,1,H,W])
        if self.augment is not None:
            x = self.augment(x)
        # Remove batch dim back: [1, n_mels, time]
        x = x.squeeze(0)
        
        return x, y

