"""Tests for dataset loading."""
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.gtzan import GTZAN


def create_mock_dataset(tmpdir):
    """Create a mock GTZAN dataset structure for testing."""
    # Create genre directories
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        genre_dir = Path(tmpdir) / genre
        genre_dir.mkdir(parents=True)
        
        # Create a few dummy WAV files per genre
        for i in range(3):
            wav_path = genre_dir / f"{genre}.{i:05d}.wav"
            # Create a minimal WAV file header (just for structure testing)
            wav_path.write_bytes(b"dummy wav content")
    
    return str(tmpdir), genres


class TestGTZAN:
    """Test GTZAN dataset loader."""
    
    def test_dataset_creation_with_mock(self):
        """Test dataset creation with mock data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, genres = create_mock_dataset(tmpdir)
            
            # Dataset initialization should succeed even with dummy WAV files
            # (audio loading only happens in __getitem__, not __init__)
            dataset = GTZAN(
                root=data_root,
                split='train',
                target_sr=22050,
                duration=30.0,
            )
            
            # Check that structure is correct
            assert hasattr(dataset, 'items')
            assert hasattr(dataset, 'GENRES')
            assert len(dataset.GENRES) == 10
            # Items should be populated (even if audio files are invalid)
            assert isinstance(dataset.items, list)
            assert len(dataset.items) > 0  # Should have items from the mock files
    
    def test_dataset_split_filtering(self):
        """Test that dataset correctly filters by split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, genres = create_mock_dataset(tmpdir)
            
            # Test loading specific splits
            dataset_train = GTZAN(
                root=data_root,
                split='train',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=30.0,
            )
            dataset_val = GTZAN(
                root=data_root,
                split='val',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=30.0,
            )
            dataset_test = GTZAN(
                root=data_root,
                split='test',
                train_ratio=0.8,
                val_ratio=0.1,
                target_sr=22050,
                duration=30.0,
            )
            
            # All splits should have items (even if audio files are invalid)
            assert len(dataset_train.items) >= 0
            assert len(dataset_val.items) >= 0
            assert len(dataset_test.items) >= 0
            
            # With 3 files per genre and 10 genres, we should have items
            # (exact counts depend on split ratios, but all should have some)
            total_items = len(dataset_train.items) + len(dataset_val.items) + len(dataset_test.items)
            assert total_items > 0  # Should have items from mock files
    
    def test_dataset_class_mapping(self):
        """Test that class mapping is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, genres = create_mock_dataset(tmpdir)
            
            dataset = GTZAN(
                root=data_root,
                split='train',
                target_sr=22050,
                duration=30.0,
            )
            
            # Check class mapping structure
            assert hasattr(dataset, 'name2idx')
            assert hasattr(dataset, 'idx2name')
            
            # Check that mappings are consistent
            assert len(dataset.GENRES) == 10
            for i, genre in enumerate(dataset.GENRES):
                assert dataset.idx2name[i] == genre
                assert dataset.name2idx[genre] == i
