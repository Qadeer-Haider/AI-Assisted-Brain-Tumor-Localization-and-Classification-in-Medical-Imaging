"""
Segmentation model architectures using keras-unet-collection.

Provides wrapper functions for building various U-Net architectures
with configurable backbones and settings.
"""

from typing import Optional, Tuple

import tensorflow as tf

try:
    from keras_unet_collection import models as kuc_models
    KERAS_UNET_AVAILABLE = True
except ImportError:
    KERAS_UNET_AVAILABLE = False
    print("⚠️ keras-unet-collection not installed. Please install with: pip install keras-unet-collection")


def build_unet(
    input_size: Tuple[int, int, int] = (256, 256, 3),
    n_classes: int = 1,
    filter_num: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    backbone: str = "ResNet50",
    weights: str = "imagenet",
    freeze_backbone: bool = True,
    freeze_batch_norm: bool = True,
) -> tf.keras.Model:
    """
    Build standard U-Net with pre-trained backbone.
    
    Args:
        input_size: Input image shape (H, W, C).
        n_classes: Number of output classes (1 for binary segmentation).
        filter_num: Number of filters at each depth level.
        backbone: Backbone architecture name.
        weights: Pre-trained weights ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights.
        freeze_batch_norm: Whether to freeze batch normalization layers.
        
    Returns:
        Keras Model.
    """
    if not KERAS_UNET_AVAILABLE:
        raise ImportError("keras-unet-collection is required. Install with: pip install keras-unet-collection")
    
    output_activation = 'Sigmoid' if n_classes == 1 else 'Softmax'
    
    model = kuc_models.unet_2d(
        input_size=input_size,
        filter_num=list(filter_num),
        n_labels=n_classes,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        output_activation=output_activation,
        batch_norm=True,
        pool=True,
        unpool=True,
        backbone=backbone,
        weights=weights,
        freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm,
        name='UNet',
    )
    
    return model


def build_attention_unet(
    input_size: Tuple[int, int, int] = (256, 256, 3),
    n_classes: int = 1,
    filter_num: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    backbone: str = "ResNet50",
    weights: str = "imagenet",
    freeze_backbone: bool = True,
    freeze_batch_norm: bool = True,
) -> tf.keras.Model:
    """
    Build Attention U-Net with pre-trained backbone.
    
    Adds attention gates to skip connections for better feature selection.
    
    Args:
        input_size: Input image shape (H, W, C).
        n_classes: Number of output classes (1 for binary segmentation).
        filter_num: Number of filters at each depth level.
        backbone: Backbone architecture name.
        weights: Pre-trained weights ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights.
        freeze_batch_norm: Whether to freeze batch normalization layers.
        
    Returns:
        Keras Model.
    """
    if not KERAS_UNET_AVAILABLE:
        raise ImportError("keras-unet-collection is required. Install with: pip install keras-unet-collection")
    
    output_activation = 'Sigmoid' if n_classes == 1 else 'Softmax'
    
    model = kuc_models.att_unet_2d(
        input_size=input_size,
        filter_num=list(filter_num),
        n_labels=n_classes,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        atten_activation='ReLU',
        attention='add',
        output_activation=output_activation,
        batch_norm=True,
        pool=True,
        unpool=True,
        backbone=backbone,
        weights=weights,
        freeze_backbone=freeze_backbone,
        freeze_batch_norm=freeze_batch_norm,
        name='AttentionUNet',
    )
    
    return model


def build_resunet_pp(
    input_size: Tuple[int, int, int] = (256, 256, 3),
    n_classes: int = 1,
    filter_num: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    dilation_num: Tuple[int, ...] = (1, 3, 15, 31),
    aspp_num_down: int = 256,
    aspp_num_up: int = 128,
) -> tf.keras.Model:
    """
    Build ResUNet++ architecture.
    
    Combines residual connections with atrous spatial pyramid pooling (ASPP).
    
    Args:
        input_size: Input image shape (H, W, C).
        n_classes: Number of output classes (1 for binary segmentation).
        filter_num: Number of filters at each depth level.
        dilation_num: Dilation rates for ASPP module.
        aspp_num_down: Number of ASPP filters in encoder.
        aspp_num_up: Number of ASPP filters in decoder.
        
    Returns:
        Keras Model.
    """
    if not KERAS_UNET_AVAILABLE:
        raise ImportError("keras-unet-collection is required. Install with: pip install keras-unet-collection")
    
    output_activation = 'Sigmoid' if n_classes == 1 else 'Softmax'
    
    model = kuc_models.resunet_a_2d(
        input_size=input_size,
        filter_num=list(filter_num),
        dilation_num=list(dilation_num),
        n_labels=n_classes,
        aspp_num_down=aspp_num_down,
        aspp_num_up=aspp_num_up,
        activation='ReLU',
        output_activation=output_activation,
        batch_norm=True,
        pool=False,
        unpool=True,
        name='ResUNetPP',
    )
    
    return model


def build_swin_unet(
    input_size: Tuple[int, int, int] = (256, 256, 3),
    n_classes: int = 1,
    filter_num_begin: int = 64,
    depth: int = 4,
    stack_num_down: int = 2,
    stack_num_up: int = 2,
    patch_size: Tuple[int, int] = (4, 4),
    num_heads: Tuple[int, ...] = (4, 8, 8, 8),
    window_size: Tuple[int, ...] = (4, 2, 2, 2),
    num_mlp: int = 512,
) -> tf.keras.Model:
    """
    Build Swin-UNet architecture.
    
    Transformer-based segmentation model using Swin Transformer blocks.
    
    Args:
        input_size: Input image shape (H, W, C).
        n_classes: Number of output classes (1 for binary segmentation).
        filter_num_begin: Initial number of filters.
        depth: Depth of the encoder.
        stack_num_down: Number of transformer blocks in encoder.
        stack_num_up: Number of transformer blocks in decoder.
        patch_size: Patch size for tokenization.
        num_heads: Number of attention heads at each depth.
        window_size: Window size for local attention.
        num_mlp: Hidden dimension of MLP.
        
    Returns:
        Keras Model.
    """
    if not KERAS_UNET_AVAILABLE:
        raise ImportError("keras-unet-collection is required. Install with: pip install keras-unet-collection")
    
    output_activation = 'Sigmoid' if n_classes == 1 else 'Softmax'
    
    model = kuc_models.swin_unet_2d(
        input_size=input_size,
        filter_num_begin=filter_num_begin,
        n_labels=n_classes,
        depth=depth,
        stack_num_down=stack_num_down,
        stack_num_up=stack_num_up,
        patch_size=patch_size,
        num_heads=list(num_heads),
        window_size=list(window_size),
        num_mlp=num_mlp,
        output_activation=output_activation,
        shift_window=True,
        name='SwinUNet',
    )
    
    return model


# Available architectures
ARCHITECTURE_BUILDERS = {
    "UNet": build_unet,
    "AttentionUNet": build_attention_unet,
    "ResUNetPP": build_resunet_pp,
    "SwinUNet": build_swin_unet,
}
