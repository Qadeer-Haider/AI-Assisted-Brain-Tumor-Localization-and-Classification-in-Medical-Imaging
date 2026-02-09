"""
TensorFlow Dataset creation for segmentation.

Provides efficient tf.data.Dataset pipelines for image-mask pairs
used in brain tumor segmentation training.
"""

import glob
import os
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ...utils.constants import SEGMENTATION_DATA_DIR, IMG_SIZE


def parse_filename(filepath: str) -> str:
    """
    Parse BRICS 2025 filename to extract metadata for stratification.
    
    File naming: brisc2025_{split}_{index}_{tumor}_{view}_{sequence}
    - tumor: gl (glioma), me (meningioma), pi (pituitary)
    - view: ax (axial), co (coronal), sa (sagittal)
    - sequence: t1 (T1-weighted)
    
    Args:
        filepath: Path to the image file.
        
    Returns:
        Stratification key (tumor_view_sequence).
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    parts = name.split('_')
    
    if len(parts) >= 6:
        tumor = parts[3]    # gl, me, pi
        view = parts[4]     # ax, co, sa
        sequence = parts[5] # t1
        return f"{tumor}_{view}_{sequence}"
    return "unknown"


def get_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    """
    Get matching image-mask pairs from directories.
    
    Searches for common image extensions and finds corresponding masks.
    
    Args:
        images_dir: Directory containing images.
        masks_dir: Directory containing masks.
        
    Returns:
        List of (image_path, mask_path) tuples.
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_paths = sorted(set(image_paths))
    
    pairs = []
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Try common mask extensions
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        for mask_ext in mask_extensions:
            mask_path = os.path.join(masks_dir, name_without_ext + mask_ext)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
    
    return pairs


def load_and_preprocess(
    img_path: tf.Tensor,
    mask_path: tf.Tensor,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess an image-mask pair.
    
    Args:
        img_path: Path to the image file.
        mask_path: Path to the mask file.
        img_size: Target size for resizing.
        
    Returns:
        Tuple of (image, mask) tensors normalized to [0, 1].
    """
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    
    return img, mask


def make_segmentation_dataset(
    pairs: List[Tuple[str, str]],
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = 16,
    shuffle: bool = True,
    augmentation_fn: Optional[Callable] = None,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset for segmentation from image-mask pairs.
    
    Args:
        pairs: List of (image_path, mask_path) tuples.
        img_size: Target image size.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        augmentation_fn: Optional augmentation function to apply.
        prefetch: Whether to use prefetching.
        
    Returns:
        tf.data.Dataset yielding (image, mask) batches.
    """
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    # Load and preprocess
    def _load(img_path, mask_path):
        return load_and_preprocess(img_path, mask_path, img_size)
    
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if provided
    if augmentation_fn is not None:
        ds = ds.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    
    # Batch
    ds = ds.batch(batch_size)
    
    # Prefetch
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def create_segmentation_datasets(
    train_images_dir: str,
    train_masks_dir: str,
    test_images_dir: Optional[str] = None,
    test_masks_dir: Optional[str] = None,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
    use_augmentation: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Create train, validation, and optionally test datasets.
    
    Uses stratified splitting based on tumor type, view, and sequence.
    
    Args:
        train_images_dir: Directory containing training images.
        train_masks_dir: Directory containing training masks.
        test_images_dir: Optional directory containing test images.
        test_masks_dir: Optional directory containing test masks.
        img_size: Target image size.
        batch_size: Batch size.
        val_split: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        use_augmentation: Whether to apply augmentation to training data.
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds) or (train_ds, val_ds, None).
    """
    # Load training pairs
    train_all_pairs = get_image_mask_pairs(train_images_dir, train_masks_dir)
    print(f"📂 Found {len(train_all_pairs)} image-mask pairs in train folder")
    
    # Extract stratification labels
    stratify_labels = [parse_filename(p[0]) for p in train_all_pairs]
    
    # Print distribution
    label_counts = Counter(stratify_labels)
    print(f"\n📊 Category distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count} samples")
    
    # Stratified split
    train_pairs, val_pairs = train_test_split(
        train_all_pairs,
        test_size=val_split,
        random_state=random_state,
        stratify=stratify_labels,
    )
    
    print(f"\n✅ Split Summary:")
    print(f"   Train: {len(train_pairs)}")
    print(f"   Val:   {len(val_pairs)}")
    
    # Get augmentation function
    aug_fn = None
    if use_augmentation:
        from .preprocessing import get_augmentation_fn
        aug_fn = get_augmentation_fn(img_size)
    
    # Create datasets
    train_ds = make_segmentation_dataset(
        train_pairs,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        augmentation_fn=aug_fn,
    )
    
    val_ds = make_segmentation_dataset(
        val_pairs,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        augmentation_fn=None,
    )
    
    # Test dataset
    test_ds = None
    if test_images_dir and test_masks_dir:
        test_pairs = get_image_mask_pairs(test_images_dir, test_masks_dir)
        print(f"   Test:  {len(test_pairs)}")
        
        test_ds = make_segmentation_dataset(
            test_pairs,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            augmentation_fn=None,
        )
    
    return train_ds, val_ds, test_ds
