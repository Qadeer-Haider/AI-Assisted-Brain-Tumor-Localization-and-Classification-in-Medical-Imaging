"""
Preprocessing and augmentation for segmentation.

Provides Albumentations-based augmentation pipelines that apply 
identical spatial transformations to both images and masks.
"""

from typing import Callable, Tuple

import numpy as np
from tensorflow import keras
import tensorflow as tf

try:
    import albumentations as albu
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[WARNING] Albumentations not installed. Augmentation will be disabled.")


def get_training_augmentation() -> "albu.Compose":
    """
    Create advanced augmentation pipeline using Albumentations.
    
    All spatial transforms are applied consistently to both image and mask.
    Pixel-level transforms are applied to image only.
    
    Returns:
        Albumentations Compose pipeline.
        
    Raises:
        ImportError: If albumentations is not installed.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "Albumentations is required for augmentation. "
            "Install with: pip install albumentations"
        )
    
    return albu.Compose([
        # Spatial transforms (applied to both image and mask)
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.3),
        albu.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5),
        
        # Elastic/distortion transforms (applied to both image and mask)
        albu.OneOf([
            albu.ElasticTransform(alpha=1, sigma=50, p=1),
            albu.GridDistortion(p=1),
        ], p=1),
        
        # Pixel-level transforms (applied to image only, not mask)
        albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        albu.RandomGamma(gamma_limit=(80, 120), p=0.3),
        albu.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    augmentation: "albu.Compose",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Albumentations augmentation to image and mask pair.
    
    Args:
        image: Image tensor as numpy array (float32, [0,1]).
        mask: Mask tensor as numpy array (float32, [0,1]).
        augmentation: Albumentations Compose pipeline.
        
    Returns:
        Tuple of augmented (image, mask) as float32 arrays.
    """
    # Convert to uint8 for Albumentations
    image_uint8 = (image * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Apply augmentation
    augmented = augmentation(image=image_uint8, mask=mask_uint8)
    
    # Convert back to float32 [0, 1]
    aug_image = augmented['image'].astype(np.float32) / 255.0
    aug_mask = augmented['mask'].astype(np.float32) / 255.0
    
    # Ensure mask is binary
    aug_mask = np.where(aug_mask > 0.5, 1.0, 0.0).astype(np.float32)
    
    # Ensure proper dimensions
    if len(aug_mask.shape) == 2:
        aug_mask = np.expand_dims(aug_mask, axis=-1)
    
    return aug_image, aug_mask


def get_augmentation_fn(
    img_size: Tuple[int, int] = (256, 256),
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Create a TensorFlow-compatible augmentation function.
    
    Uses tf.py_function to wrap Albumentations pipeline.
    
    Args:
        img_size: Target image size (height, width).
        
    Returns:
        Function that augments (image, mask) tensor pairs.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        # Return identity function if albumentations not available
        def identity(image, mask):
            return image, mask
        return identity
    
    # Create augmentation pipeline
    augmentation = get_training_augmentation()
    
    def _apply_aug(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return apply_augmentation(image, mask, augmentation)
    
    def tf_augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """TensorFlow wrapper for Albumentations augmentation."""
        aug_img, aug_mask = tf.py_function(
            func=_apply_aug,
            inp=[image, mask],
            Tout=[tf.float32, tf.float32]
        )
        # Set shapes (lost during py_function)
        aug_img.set_shape([img_size[0], img_size[1], 3])
        aug_mask.set_shape([img_size[0], img_size[1], 1])
        return aug_img, aug_mask
    
    return tf_augment


class SegmentationAugmentation(keras.layers.Layer):
    """
    Custom Keras layer for segmentation augmentation.
    
    Applies identical random transformations to both
    image and mask tensors. Uses TensorFlow-native transforms
    as a fallback when Albumentations is not available.
    
    Example:
        >>> aug = SegmentationAugmentation()
        >>> augmented_img, augmented_mask = aug((image, mask), training=True)
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation_range: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
    
    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply augmentation to image-mask pair.
        
        Args:
            inputs: Tuple of (image, mask) tensors.
            training: Whether in training mode.
            
        Returns:
            Tuple of augmented (image, mask).
        """
        if not training:
            return inputs
        
        image, mask = inputs
        
        # Concatenate for identical transforms
        combined = tf.concat([image, mask], axis=-1)
        
        # Apply random flips
        if self.horizontal_flip:
            combined = tf.image.random_flip_left_right(combined)
        
        if self.vertical_flip:
            combined = tf.image.random_flip_up_down(combined)
        
        # Split back
        image = combined[..., :3]
        mask = combined[..., 3:]
        
        return image, mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'rotation_range': self.rotation_range,
        })
        return config


def get_segmentation_augmentation() -> keras.Sequential:
    """
    Create TensorFlow-native augmentation pipeline for segmentation.
    
    This is a simpler alternative when Albumentations is not available.
    Note: For best results, use get_augmentation_fn() with Albumentations.
    
    Returns:
        Keras Sequential model for augmentation.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
    ], name="segmentation_augmentation")
