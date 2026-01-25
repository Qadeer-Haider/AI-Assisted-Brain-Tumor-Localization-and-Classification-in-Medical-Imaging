"""
Backbone architectures for classification.

Provides pre-trained backbone models from ImageNet for transfer learning.
All backbones return feature maps suitable for the universal classification head.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet152V2,
    DenseNet201,
    VGG16,
    EfficientNetV2S,
    ConvNeXtBase,
)

from ...utils.constants import IMG_SIZE, INPUT_SHAPE


# ═══════════════════════════════════════════════════════════════════════════════
# AVAILABLE BACKBONES
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_BACKBONES = {
    # ResNet family
    "ResNet152V2": ResNet152V2,
    
    # DenseNet family
    "DenseNet201": DenseNet201,
    
    # VGG family
    "VGG16": VGG16,
    
    # EfficientNet V2
    "EfficientNetV2S": EfficientNetV2S,
    
    # ConvNeXt (modern standard)
    "ConvNeXtBase": ConvNeXtBase,
}


def get_backbone(
    model_name: str,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    weights: str = "imagenet",
    trainable: bool = False,
) -> tf.keras.Model:
    """
    Get a pre-trained backbone model for feature extraction.
    
    All backbones are loaded without the top classification layer,
    making them suitable for transfer learning with a custom head.
    
    Args:
        model_name: Name of the backbone architecture.
        input_shape: Input shape (height, width, channels).
        weights: Pre-trained weights to use ('imagenet' or None).
        trainable: Whether to make the backbone trainable.
        
    Returns:
        Keras Model for feature extraction.
        
    Raises:
        ValueError: If model_name is not recognized.
        
    Example:
        >>> backbone = get_backbone("ConvNeXtBase", trainable=False)
        >>> features = backbone(inputs, training=False)
    """
    # Handle attention variants
    base_name = model_name.replace("_Attention", "")
    
    if base_name not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Unknown backbone: {model_name}. "
            f"Available: {list(AVAILABLE_BACKBONES.keys())}"
        )
    
    backbone_class = AVAILABLE_BACKBONES[base_name]
    
    # Create backbone without top layer
    backbone = backbone_class(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    
    # Set trainability
    backbone.trainable = trainable
    
    return backbone


def get_backbone_output_shape(model_name: str, input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> Tuple[int, ...]:
    """
    Get the output shape of a backbone model.
    
    Useful for debugging and understanding feature dimensions.
    
    Args:
        model_name: Name of the backbone architecture.
        input_shape: Input shape (height, width, channels).
        
    Returns:
        Output shape tuple (excluding batch dimension).
    """
    backbone = get_backbone(model_name, input_shape)
    return backbone.output_shape[1:]


def list_available_backbones() -> list:
    """
    List all available backbone architectures.
    
    Returns:
        List of backbone names.
    """
    return list(AVAILABLE_BACKBONES.keys())
