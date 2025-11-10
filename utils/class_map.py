"""Shared class map loading and saving utilities for live-music-classifier."""
import json
from pathlib import Path
from typing import List
from utils.logging import get_logger

logger = get_logger("class_map")

# Standard GTZAN genres
GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_class_map(data_root: Path, artifacts_dir: Path) -> List[str]:
    """
    Load class names in index order.
    
    Priority:
    1. artifacts/class_map.json with key 'idx2name' (list or dict)
    2. GTZAN genre folders to derive class names (standard GTZAN genres)
    
    Args:
        data_root: Path to GTZAN root directory.
        artifacts_dir: Path to artifacts directory (where class_map.json would be).
    
    Returns:
        List of class names in index order (idx 0, 1, 2, ...).
    
    Raises:
        FileNotFoundError: If neither class_map.json nor GTZAN genre folders are found.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import load_class_map
        >>> idx2name = load_class_map(Path("/path/to/GTZAN"), Path("artifacts"))
        >>> print(idx2name[0])  # First class name
    """
    # Try to load from class_map.json first
    cm_path = artifacts_dir / "class_map.json"
    if cm_path.exists():
        try:
            with open(cm_path, "r") as f:
                data = json.load(f)
            if isinstance(data.get("idx2name"), list):
                logger.debug(f"Loaded class map from {cm_path} (list format)")
                return data["idx2name"]
            elif isinstance(data.get("idx2name"), dict):
                # Convert dict to list, sorted by key
                idx2name = [v for k, v in sorted(data["idx2name"].items(), key=lambda kv: int(kv[0]))]
                logger.debug(f"Loaded class map from {cm_path} (dict format)")
                return idx2name
        except Exception as e:
            logger.warning(f"Failed to load class_map.json: {e}, falling back to GTZAN genres")
    
    # Fallback: use standard GTZAN genres or derive from folder structure
    if data_root.exists():
        # Check if it's a GTZAN-style directory with genre folders
        found_genres = []
        for genre in GTZAN_GENRES:
            genre_dir = data_root / genre
            if genre_dir.exists() and genre_dir.is_dir():
                found_genres.append(genre)
        
        if found_genres:
            # Use found genres, sorted to match standard order
            idx2name = [g for g in GTZAN_GENRES if g in found_genres]
            logger.debug(f"Loaded class map from GTZAN genre folders: {len(idx2name)} genres")
            return idx2name
    
    # Final fallback: use standard GTZAN genres
    logger.warning(f"Could not find class_map.json at {cm_path} or GTZAN genre folders in {data_root}. Using standard GTZAN genres.")
    return GTZAN_GENRES.copy()


def save_class_map(artifacts_dir: Path, idx2name: List[str]) -> None:
    """
    Save class map to artifacts/class_map.json.
    
    Args:
        artifacts_dir: Path to artifacts directory.
        idx2name: List of class names in index order.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import save_class_map
        >>> save_class_map(Path("artifacts"), ["air_conditioner", "car_horn", ...])
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm_path = artifacts_dir / "class_map.json"
    
    data = {
        "idx2name": idx2name,
        "num_classes": len(idx2name)
    }
    
    with open(cm_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved class map to {cm_path} ({len(idx2name)} classes)")

