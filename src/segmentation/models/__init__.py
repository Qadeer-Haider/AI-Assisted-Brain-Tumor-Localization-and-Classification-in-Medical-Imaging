"""
Segmentation model architectures.

Provides factory functions for building various U-Net based
segmentation architectures using keras-unet-collection.
"""

from .architectures import (
    ARCHITECTURE_BUILDERS,
    build_attention_unet,
    build_resunet_pp,
    build_swin_unet,
    build_unet,
)
from .builder import (
    AVAILABLE_SEGMENTATION_MODELS,
    build_model_from_config,
    build_segmentation_model,
    list_available_models,
)

__all__ = [
    # Factory functions
    "build_segmentation_model",
    "build_model_from_config",
    "list_available_models",
    # Architecture builders
    "build_unet",
    "build_attention_unet",
    "build_resunet_pp",
    "build_swin_unet",
    # Constants
    "AVAILABLE_SEGMENTATION_MODELS",
    "ARCHITECTURE_BUILDERS",
]
