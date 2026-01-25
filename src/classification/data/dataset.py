"""
TensorFlow Dataset creation for classification.

Provides efficient tf.data.Dataset pipelines for training,
validation, and testing.
"""

from typing import Callable, Dict, Optional

import pandas as pd
import tensorflow as tf

from ...utils.constants import (
    CLASS_TO_IDX,
    DEFAULT_BATCH_SIZE,
    IMG_SIZE,
    NUM_CLASSES,
)
from .preprocessing import get_preprocessing_fn, get_train_augmentation


def make_dataset(
    df: pd.DataFrame,
    model_name: str = "ConvNeXtBase",
    shuffle: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augmentation: Optional[tf.keras.Sequential] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    img_size: tuple = IMG_SIZE,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from a DataFrame of image paths and labels.
    
    This is a production-ready data pipeline that:
    - Loads images efficiently using tf.data
    - Applies model-specific preprocessing
    - Optionally applies data augmentation
    - Uses prefetching and parallel processing for performance
    
    Args:
        df: DataFrame with 'path' and 'class' columns.
        model_name: Name of the backbone model (for preprocessing selection).
        shuffle: Whether to shuffle the dataset.
        batch_size: Batch size.
        augmentation: Optional augmentation Sequential model.
        class_to_idx: Mapping from class names to indices.
        img_size: Target image size (height, width).
        prefetch: Whether to use prefetching.
        
    Returns:
        tf.data.Dataset yielding (image, label) batches.
        
    Example:
        >>> train_ds = make_dataset(train_df, shuffle=True, augmentation=augs)
        >>> val_ds = make_dataset(val_df, shuffle=False)
    """
    if class_to_idx is None:
        class_to_idx = CLASS_TO_IDX
    
    num_classes = len(class_to_idx)
    
    # Get preprocessing function for this model
    preprocess_fn = get_preprocessing_fn(model_name)
    
    # Extract paths and labels
    paths = df["path"].values
    labels = df["class"].map(class_to_idx).values
    
    # Create base dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Define loading function
    def load_img(path, label):
        # 1. Read and decode
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # 2. Resize
        img = tf.image.resize(img, img_size)
        
        # 3. Apply model-specific preprocessing
        img = preprocess_fn(img)
        
        # 4. Apply augmentation ONLY if provided
        if augmentation is not None:
            img = augmentation(img, training=True)
        
        # 5. One-hot encode label
        label = tf.one_hot(label, num_classes)
        
        return img, label
    
    # Map with parallel calls for speed
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle only if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    
    # Batch
    ds = ds.batch(batch_size)
    
    # Prefetch for performance
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    model_name: str = "ConvNeXtBase",
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_augmentation: bool = True,
) -> tuple:
    """
    Create train, validation, and optionally test datasets.
    
    Convenience function that creates all datasets with appropriate settings:
    - Training: shuffled, with augmentation
    - Validation: not shuffled, no augmentation
    - Test: not shuffled, no augmentation
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Optional test DataFrame.
        model_name: Model name for preprocessing selection.
        batch_size: Batch size for all datasets.
        use_augmentation: Whether to use augmentation for training.
        
    Returns:
        Tuple of (train_ds, val_ds) or (train_ds, val_ds, test_ds).
        
    Example:
        >>> train_ds, val_ds, test_ds = create_datasets(train_df, val_df, test_df)
    """
    # Get augmentation for training
    train_aug = get_train_augmentation() if use_augmentation else None
    
    # Create datasets
    train_ds = make_dataset(
        train_df,
        model_name=model_name,
        shuffle=True,
        batch_size=batch_size,
        augmentation=train_aug,
    )
    
    val_ds = make_dataset(
        val_df,
        model_name=model_name,
        shuffle=False,
        batch_size=batch_size,
        augmentation=None,
    )
    
    if test_df is not None:
        test_ds = make_dataset(
            test_df,
            model_name=model_name,
            shuffle=False,
            batch_size=batch_size,
            augmentation=None,
        )
        return train_ds, val_ds, test_ds
    
    return train_ds, val_ds
