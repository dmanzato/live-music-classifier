# datasets/gtzan.py
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from transforms.audio import get_mel_transform, wav_to_logmel

# Canonical GTZAN genre ordering (10 classes)
GENRES: List[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


@dataclass(frozen=True)
class Item:
    path: Path
    label: int  # class index
    genre: str  # class name


class GTZAN(Dataset):
    """
    GTZAN dataset loader with deterministic, stratified train/val/test splits.

    Folder layout expected (common GTZAN "genres_original"):
        root/
          blues/*.wav
          classical/*.wav
          ...
          rock/*.wav

    Returns:
        x:  torch.FloatTensor [1, n_mels, time]  (log-mel spectrogram)
        y:  int (class index)
        meta: dict with keys {"path", "genre"}
    """

    # expose for external consumers
    GENRES = GENRES

    def __init__(
        self,
        root: str,
        split: str = "train",              # {"train","val","test"}
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        target_sr: int = 22050,
        duration: float = 7.5,            # seconds
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: int = 1337,                 # deterministic split seed
    ):
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        assert 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0, "Invalid ratios"
        assert (train_ratio + val_ratio) < 1.0, "train_ratio + val_ratio must be < 1.0"

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"GTZAN root not found: {self.root}")

        # class maps
        self.name2idx: Dict[str, int] = {g: i for i, g in enumerate(GENRES)}
        self.idx2name: Dict[int, str] = {i: g for g, i in self.name2idx.items()}

        self.split = split
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.target_sr = int(target_sr)
        self.duration = float(duration)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.augment = augment
        self.seed = int(seed)

        # Build deterministic, per-genre split
        all_items: List[Item] = []
        for g in GENRES:
            g_dir = self.root / g
            if not g_dir.exists():
                # Skip missing genres; keep class map stable
                continue
            files = sorted([p for p in g_dir.glob("*.wav") if p.is_file()])
            rng = random.Random(self.seed)  # deterministic per genre
            rng.shuffle(files)

            n = len(files)
            n_train = int(round(self.train_ratio * n))
            n_val = int(round(self.val_ratio * n))
            # ensure we don't exceed
            n_train = min(n_train, n)
            n_val = min(n_val, max(0, n - n_train))
            n_test = max(0, n - n_train - n_val)

            if self.split == "train":
                split_files = files[:n_train]
            elif self.split == "val":
                split_files = files[n_train:n_train + n_val]
            else:
                split_files = files[n_train + n_val:n_train + n_val + n_test]

            label = self.name2idx[g]
            all_items.extend([Item(path=f, label=label, genre=g) for f in split_files])

        # Keep a stable order for reproducibility
        self.items: List[Item] = sorted(all_items, key=lambda it: (it.label, it.path.name))

        # Prepare mel transform ON CPU (safer with MPS / CUDA STFT windowing)
        self._mel_t = get_mel_transform(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # precompute crop strategy
        self._random_crop = (self.split == "train")
        self._num_samples = int(round(self.duration * self.target_sr))

    # ------------ public utilities ------------

    def __len__(self) -> int:
        return len(self.items)

    def classes(self) -> Sequence[str]:
        return [self.idx2name[i] for i in range(len(self.idx2name))]

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {g: 0 for g in GENRES}
        for it in self.items:
            counts[it.genre] = counts.get(it.genre, 0) + 1
        return counts

    # ------------ core I/O & transforms ------------

    @staticmethod
    def _center_or_random_crop(x: np.ndarray, N: int, random_crop: bool, rng: Optional[random.Random] = None) -> np.ndarray:
        L = len(x)
        if L == N:
            return x
        if L < N:
            pad = N - L
            left = pad // 2
            right = pad - left
            return np.pad(x, (left, right), mode="constant")
        # L > N
        if random_crop:
            r = rng if rng is not None else random
            start = r.randint(0, L - N)
        else:
            start = max(0, (L - N) // 2)
        return x[start:start + N]

    def _load_mono(self, path: Path) -> Tuple[np.ndarray, int]:
        # Always_2d gives [T, C] -> we average channels
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        else:
            wav = wav[:, 0]
        return wav, int(sr)

    def _resample_if_needed(self, x: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.target_sr:
            return x
        # high-quality rational resampling
        return resample_poly(x, self.target_sr, sr).astype(np.float32)

    def _wav_to_logmel(self, x: np.ndarray) -> torch.Tensor:
        # x: mono float32 [T] at target_sr
        with torch.no_grad():
            wav_t = torch.from_numpy(x).unsqueeze(0)  # [1, T] CPU
            logmel = wav_to_logmel(wav_t, sr=self.target_sr, mel_transform=self._mel_t)  # [1, n_mels, time]
            # Per-example standardization helps optimization
            logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        return logmel

    # ------------ dataset API ------------

    def __getitem__(self, index: int):
        it = self.items[index]

        # Load & preprocess audio
        wav, sr = self._load_mono(it.path)
        wav = self._resample_if_needed(wav, sr)

        # Crop/pad duration
        rng = random.Random(self.seed + index) if self._random_crop else None
        wav = self._center_or_random_crop(wav, self._num_samples, self._random_crop, rng)

        # To log-mel
        logmel = self._wav_to_logmel(wav)  # [1, n_mels, time]

        # Optional SpecAugment (or any augment callable expecting [1,n_mels,time])
        if self.augment is not None:
            logmel = self.augment(logmel)

        y = it.label
        meta = {"path": str(it.path), "genre": it.genre}
        return logmel, y, meta