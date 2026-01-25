"""
Preprocessing functions for classification.

Provides image preprocessing functions for different backbone architectures
and data augmentation pipelines for training.
"""

from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import layers

# Import preprocessing functions for each backbone
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.convnext import preprocess_input as convnext_prep

from ...utils.constants import IMG_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

PREPROCESSING_FUNCTIONS = {
    "ResNet152V2": resnet_prep,
    "DenseNet201": densenet_prep,
    "VGG16": vgg_prep,
    "EfficientNetV2S": efficientnet_prep,
    "ConvNeXtBase": convnext_prep,
}


def get_preprocessing_fn(model_name: str) -> Callable:
    """
    Get the appropriate preprocessing function for a model architecture.
    
    Each backbone requires its specific preprocessing to match ImageNet
    training conditions (normalization, scaling, etc.).
    
    Args:
        model_name: Name of the model architecture.
        
    Returns:
        Preprocessing function for the specified model.
        
    Raises:
        ValueError: If model_name is not recognized.
        
    Example:
        >>> preprocess = get_preprocessing_fn("ConvNeXtBase")
        >>> img = preprocess(img)
    """
    # Handle variants
    base_name = model_name.replace("_Attention", "")
    
    if base_name in PREPROCESSING_FUNCTIONS:
        return PREPROCESSING_FUNCTIONS[base_name]
    
    # Try partial match
    for key, fn in PREPROCESSING_FUNCTIONS.items():
        if key in model_name or model_name in key:
            return fn
    
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: {list(PREPROCESSING_FUNCTIONS.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_augmentation(
    horizontal_flip: bool = True,
    rotation_range: float = 0.028,
) -> tf.keras.Sequential:
    """
    Create a data augmentation pipeline for training.
    
    Uses conservative augmentations suitable for medical imaging,
    where excessive distortion could affect clinical relevance.
    
    Args:
        horizontal_flip: Whether to apply horizontal flipping.
        rotation_range: Rotation range as fraction of 2π.
        
    Returns:
        Keras Sequential model for augmentation.
        
    Example:
        >>> augment = get_train_augmentation()
        >>> augmented_img = augment(img, training=True)
    """
    augmentation_layers = []
    
    if horizontal_flip:
        augmentation_layers.append(layers.RandomFlip("horizontal"))
    
    if rotation_range > 0:
        augmentation_layers.append(layers.RandomRotation(rotation_range))
    
    return tf.keras.Sequential(augmentation_layers, name="train_augmentation")


def get_validation_augmentation() -> Optional[tf.keras.Sequential]:
    """
    Get augmentation for validation/test (None - no augmentation).
    
    Returns:
        None (no augmentation for validation/test).
    """
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(
    path: tf.Tensor,
    img_size: tuple = IMG_SIZE,
    preprocess_fn: Optional[Callable] = None,
) -> tf.Tensor:
    """
    Load and preprocess a single image from path.
    
    Args:
        path: Path to the image file.
        img_size: Target image size (height, width).
        preprocess_fn: Preprocessing function to apply.
        
    Returns:
        Preprocessed image tensor.
    """
    # Read and decode
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize
    img = tf.image.resize(img, img_size)
    
    # Apply model-specific preprocessing
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    
    return img
