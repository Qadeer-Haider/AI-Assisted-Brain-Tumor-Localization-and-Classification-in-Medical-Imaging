"""
U-Net architecture for segmentation.

Provides U-Net model implementation for brain tumor segmentation.

TODO: Implement when segmentation notebook is ready.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

from ...utils.constants import IMG_SIZE, INPUT_SHAPE


def build_unet(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 1,
    encoder_filters: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build a U-Net model for semantic segmentation.
    
    U-Net architecture:
    - Encoder: Contracting path with convolutions and pooling
    - Bottleneck: Deepest layer
    - Decoder: Expanding path with upsampling and skip connections
    
    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes (1 for binary segmentation).
        encoder_filters: Number of filters in each encoder block.
        dropout_rate: Dropout rate in bottleneck.
        
    Returns:
        Keras Model.
        
    Note:
        This is a placeholder. Implementation pending.
    """
    raise NotImplementedError(
        "U-Net model not yet implemented. "
        "This will be added when the segmentation notebook is ready."
    )


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
) -> tf.Tensor:
    """
    Convolutional block with two Conv-BN-ReLU sequences.
    
    Args:
        x: Input tensor.
        filters: Number of filters.
        kernel_size: Convolution kernel size.
        
    Returns:
        Output tensor.
    """
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    return x


def encoder_block(
    x: tf.Tensor,
    filters: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Encoder block: Conv block + MaxPooling.
    
    Args:
        x: Input tensor.
        filters: Number of filters.
        
    Returns:
        Tuple of (pooled, skip_connection).
    """
    skip = conv_block(x, filters)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return pooled, skip


def decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
) -> tf.Tensor:
    """
    Decoder block: UpSampling + Concatenate + Conv block.
    
    Args:
        x: Input tensor from previous decoder/bottleneck.
        skip: Skip connection from encoder.
        filters: Number of filters.
        
    Returns:
        Output tensor.
    """
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x
