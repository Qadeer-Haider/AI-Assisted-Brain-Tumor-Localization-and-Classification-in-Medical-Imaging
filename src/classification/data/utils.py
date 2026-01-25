"""
Data utilities for classification task.

Provides functions for building dataframes from file paths,
stratified train/val splitting, and class weight computation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from ...utils.constants import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    CLASSIFICATION_DATA_DIR,
    DEFAULT_VAL_SPLIT,
    RANDOM_SEED,
)


def build_dataframe(
    data_dir: Optional[Path] = None,
    split: str = "train",
) -> pd.DataFrame:
    """
    Build a DataFrame from image file paths.
    
    Parses filenames following the BRISC2025 naming convention:
    brisc2025_<split>_<index>_<tumor>_<view>_<sequence>.jpg
    
    Args:
        data_dir: Path to the data directory. If None, uses default classification dir.
        split: Dataset split ('train' or 'test').
        
    Returns:
        DataFrame with columns: path, class, plane, strata
        
    Example:
        >>> df = build_dataframe(split="train")
        >>> print(df.head())
    """
    if data_dir is None:
        data_dir = CLASSIFICATION_DATA_DIR / split
    else:
        data_dir = Path(data_dir)
    
    # Find all images recursively
    all_paths = list(data_dir.rglob("*.jpg"))
    
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {data_dir}")
    
    records = []
    for p in all_paths:
        fname = p.name
        # Parse filename: brisc2025_train_00001_gl_ax_t1.jpg
        parts = fname.replace(".jpg", "").split("_")
        
        if len(parts) >= 5:
            tumor_class = _parse_tumor_class(parts[3])
            plane = parts[4]
        else:
            # Fallback: use parent directory name as class
            tumor_class = p.parent.name
            plane = "unknown"
        
        records.append({
            "path": str(p),
            "class": tumor_class,
            "plane": plane,
        })
    
    df = pd.DataFrame(records)
    
    # Create stratification column for balanced splitting
    df["strata"] = df["class"] + "_" + df["plane"]
    
    return df


def _parse_tumor_class(abbrev: str) -> str:
    """
    Parse tumor abbreviation to full class name.
    
    Args:
        abbrev: Abbreviation (gl, me, nt, pi).
        
    Returns:
        Full class name.
    """
    mapping = {
        "gl": "glioma",
        "me": "meningioma",
        "nt": "no_tumor",
        "no": "no_tumor",  # Alternative abbreviation
        "pi": "pituitary",
    }
    return mapping.get(abbrev, abbrev)


def get_stratified_split(
    df: pd.DataFrame,
    val_split: float = DEFAULT_VAL_SPLIT,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and validation sets using stratified sampling.
    
    Stratifies by both class and plane to ensure balanced representation.
    
    Args:
        df: Input DataFrame with 'strata' column.
        val_split: Fraction of data for validation.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df).
        
    Example:
        >>> df = build_dataframe(split="train")
        >>> train_df, val_df = get_stratified_split(df, val_split=0.2)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=random_state,
    )
    
    train_idx, val_idx = next(splitter.split(df, df["strata"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df


def compute_class_weights(
    df: pd.DataFrame,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced dataset handling.
    
    Uses sklearn's 'balanced' strategy to compute weights inversely
    proportional to class frequencies.
    
    Args:
        df: DataFrame with 'class' column.
        class_to_idx: Mapping from class names to indices.
        
    Returns:
        Dictionary mapping class indices to weights.
        
    Example:
        >>> weights = compute_class_weights(train_df)
        >>> print(weights)
        {0: 1.2, 1: 0.9, 2: 1.1, 3: 0.8}
    """
    if class_to_idx is None:
        class_to_idx = CLASS_TO_IDX
    
    # Map class names to indices
    labels = df["class"].map(class_to_idx)
    
    # Check for unmapped classes (NaN values)
    if labels.isna().any():
        unmapped = df[labels.isna()]["class"].unique()
        raise ValueError(
            f"Found unmapped class names: {unmapped}. "
            f"Expected classes: {list(class_to_idx.keys())}"
        )
    
    # Convert to numpy array
    labels = labels.values.astype(int)
    num_classes = len(class_to_idx)
    
    # Get unique classes actually present in the labels
    unique_classes = np.unique(labels)
    
    # Compute weights only for classes present in the dataset
    weights_array = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    
    # Create full weight dict with all classes
    # Classes not in training data get weight 1.0
    weight_dict = {i: 1.0 for i in range(num_classes)}
    for cls, weight in zip(unique_classes, weights_array):
        weight_dict[int(cls)] = float(weight)
    
    return weight_dict


def print_distribution(df: pd.DataFrame, split_name: str = "Split") -> None:
    """
    Print class and plane distribution for a DataFrame.
    
    Args:
        df: DataFrame with 'class' and 'plane' columns.
        split_name: Name of the split for display.
    """
    print(f"\n✅ {split_name} class distribution:")
    print(df["class"].value_counts())
    
    print(f"\n✅ {split_name} plane distribution:")
    print(df["plane"].value_counts())
    
    print(f"\n✅ {split_name} class-plane groups:")
    print(df.groupby(["class", "plane"]).size())
