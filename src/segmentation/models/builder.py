"""
Model builder for segmentation.

Factory function that creates segmentation models using keras-unet-collection.
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf

from ...utils.constants import INPUT_SHAPE
from .architectures import (
    ARCHITECTURE_BUILDERS,
    build_attention_unet,
    build_resunet_pp,
    build_swin_unet,
    build_unet,
)


# Available models for segmentation
AVAILABLE_SEGMENTATION_MODELS = [
    "UNet",
    "AttentionUNet", 
    "ResUNetPP",
    "SwinUNet",
]


def build_segmentation_model(
    model_name: str = "UNet",
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    backbone: str = "ResNet50",
    weights: str = "imagenet",
    freeze_backbone: bool = True,
    **kwargs,
) -> tf.keras.Model:
    """
    Build a segmentation model.
    
    Factory function that creates various segmentation architectures
    using keras-unet-collection library.
    
    Args:
        model_name: Name of the model architecture.
            Options: "UNet", "AttentionUNet", "ResUNetPP", "SwinUNet"
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes (1 for binary segmentation).
        backbone: Backbone architecture for encoder (e.g., "ResNet50").
        weights: Pre-trained weights ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights during training.
        **kwargs: Additional model-specific arguments.
        
    Returns:
        Keras Model.
        
    Example:
        >>> model = build_segmentation_model("UNet", backbone="ResNet50")
        >>> model.summary()
    """
    print(f"\n🏗️ Building {model_name} with {backbone} backbone...")
    
    if model_name not in AVAILABLE_SEGMENTATION_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {AVAILABLE_SEGMENTATION_MODELS}"
        )
    
    if model_name == "UNet":
        model = build_unet(
            input_size=input_shape,
            n_classes=num_classes,
            backbone=backbone,
            weights=weights,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )
    elif model_name == "AttentionUNet":
        model = build_attention_unet(
            input_size=input_shape,
            n_classes=num_classes,
            backbone=backbone,
            weights=weights,
            freeze_backbone=freeze_backbone,
            **kwargs,
        )
    elif model_name == "ResUNetPP":
        # ResUNet++ doesn't use pre-trained backbone
        model = build_resunet_pp(
            input_size=input_shape,
            n_classes=num_classes,
            **kwargs,
        )
    elif model_name == "SwinUNet":
        # Swin-UNet has its own transformer architecture
        model = build_swin_unet(
            input_size=input_shape,
            n_classes=num_classes,
            **kwargs,
        )
    
    print(f"   ✅ Parameters: {model.count_params():,}")
    
    return model


def build_model_from_config(config: Dict) -> tf.keras.Model:
    """
    Build a segmentation model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with model settings.
        
    Returns:
        Keras Model.
        
    Example:
        >>> config = load_config("configs/segmentation_config.yaml")
        >>> model = build_model_from_config(config)
    """
    img_size = config.get("img_size", [256, 256])
    input_shape = (*img_size, 3)
    
    return build_segmentation_model(
        model_name=config.get("model_name", "UNet"),
        input_shape=input_shape,
        num_classes=config.get("num_classes", 1),
        backbone=config.get("backbone", "ResNet50"),
        weights=config.get("weights", "imagenet"),
        freeze_backbone=config.get("freeze_backbone", True),
    )


def list_available_models() -> list:
    """
    List all available segmentation model configurations.
    
    Returns:
        List of model names.
    """
    return AVAILABLE_SEGMENTATION_MODELS.copy()
