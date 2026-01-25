"""
TensorFlow Dataset creation for segmentation.

Provides efficient tf.data.Dataset pipelines for image-mask pairs
used in tumor segmentation training.

TODO: Implement when segmentation notebook is ready.
"""

from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

from ...utils.constants import SEGMENTATION_DATA_DIR, IMG_SIZE


def make_segmentation_dataset(
    data_dir: Optional[Path] = None,
    split: str = "train",
    batch_size: int = 16,
    img_size: Tuple[int, int] = IMG_SIZE,
    shuffle: bool = True,
    augmentation: Optional[tf.keras.Sequential] = None,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset for segmentation from image-mask pairs.
    
    Args:
        data_dir: Path to the data directory.
        split: Dataset split ('train' or 'test').
        batch_size: Batch size.
        img_size: Target image size.
        shuffle: Whether to shuffle the dataset.
        augmentation: Optional augmentation pipeline.
        
    Returns:
        tf.data.Dataset yielding (image, mask) batches.
        
    Note:
        This is a placeholder. Implementation pending.
    """
    raise NotImplementedError(
        "Segmentation dataset creation not yet implemented. "
        "This will be added when the segmentation notebook is ready."
    )


def load_image_mask_pair(
    image_path: str,
    mask_path: str,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load an image and its corresponding mask.
    
    Args:
        image_path: Path to the image file.
        mask_path: Path to the mask file.
        img_size: Target size for resizing.
        
    Returns:
        Tuple of (image, mask) tensors.
    """
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    
    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method="nearest")
    mask = mask / 255.0  # Binary mask
    
    return img, mask
