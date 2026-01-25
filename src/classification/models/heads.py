"""
Classification heads and attention mechanisms.

Provides the universal classification head and Squeeze-and-Excitation
attention blocks for enhanced feature learning.
"""

import tensorflow as tf
from tensorflow.keras import layers

from ...utils.constants import NUM_CLASSES


def add_universal_head(
    x: tf.Tensor,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    dense_units: int = 256,
) -> tf.Tensor:
    """
    Add a universal classification head to backbone features.
    
    This head is designed to work with any CNN backbone output,
    providing consistent classification performance across architectures.
    
    Architecture:
    - Global Average Pooling (spatial reduction)
    - BatchNorm + Dense + Dropout (block 1)
    - BatchNorm + Dense + Dropout (block 2)
    - Softmax output
    
    Args:
        x: Feature tensor from backbone (shape: [B, H, W, C]).
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for regularization.
        dense_units: Number of units in dense layers.
        
    Returns:
        Output tensor with shape [B, num_classes].
        
    Example:
        >>> features = backbone(inputs)
        >>> outputs = add_universal_head(features, num_classes=4)
    """
    # 1. Global pooling: (H, W, C) → (C)
    x = layers.GlobalAveragePooling2D()(x)
    
    # 2. First dense block with normalization & dropout
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 3. Second dense block (more regularization)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 4. Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return outputs


def attention_block(x: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    """
    Squeeze-and-Excitation (SE) attention block.
    
    This block adds channel-wise attention by learning which feature
    channels are most important for the classification task.
    
    The mechanism:
    1. Squeeze: Global Average Pooling to get channel descriptors
    2. Excitation: FC bottleneck to learn channel importance
    3. Scale: Multiply original features by learned weights
    
    Reference:
        "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)
    
    Args:
        x: Input feature tensor (shape: [B, H, W, C]).
        ratio: Reduction ratio for the bottleneck layer.
        
    Returns:
        Reweighted feature tensor with same shape as input.
        
    Example:
        >>> features = backbone(inputs)
        >>> features = attention_block(features, ratio=16)
        >>> outputs = add_universal_head(features)
    """
    # Input shape: (Batch, H, W, C)
    channel_axis = -1
    filters = x.shape[channel_axis]
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(x)
    
    # Excitation: Bottleneck → Expansion
    # Reduce channels, then expand back
    se = layers.Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
    )(se)
    se = layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
    )(se)
    
    # Reshape to (Batch, 1, 1, C) for broadcasting
    se = layers.Reshape((1, 1, filters))(se)
    
    # Scale: Reweight the original features
    x = layers.Multiply()([x, se])
    
    return x


def channel_attention(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Channel attention module (CBAM-style).
    
    Uses both average and max pooling for richer channel descriptors.
    
    Args:
        x: Input feature tensor.
        ratio: Reduction ratio.
        
    Returns:
        Channel-attention weighted features.
    """
    channel_axis = -1
    filters = x.shape[channel_axis]
    
    # Shared MLP
    shared_dense1 = layers.Dense(filters // ratio, activation="relu")
    shared_dense2 = layers.Dense(filters)
    
    # Average pooling path
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_out = shared_dense2(shared_dense1(avg_pool))
    
    # Max pooling path
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_out = shared_dense2(shared_dense1(max_pool))
    
    # Combine and activate
    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation("sigmoid")(attention)
    attention = layers.Reshape((1, 1, filters))(attention)
    
    return layers.Multiply()([x, attention])


def spatial_attention(x: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    """
    Spatial attention module (CBAM-style).
    
    Highlights important spatial regions in the feature maps.
    
    Args:
        x: Input feature tensor.
        kernel_size: Convolution kernel size.
        
    Returns:
        Spatially-attention weighted features.
    """
    # Compute channel-wise statistics
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    
    # Concatenate and convolve
    concat = layers.Concatenate()([avg_pool, max_pool])
    attention = layers.Conv2D(
        1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
    )(concat)
    
    return layers.Multiply()([x, attention])
