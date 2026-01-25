"""
Model builder for classification.

Factory function that assembles backbone + head into complete models.
"""

from typing import Optional

import tensorflow as tf

from ...utils.constants import INPUT_SHAPE, NUM_CLASSES, AVAILABLE_MODELS
from .backbones import get_backbone
from .heads import add_universal_head, attention_block


def build_model(
    model_name: str,
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    weights: str = "imagenet",
    trainable_backbone: bool = False,
    use_attention: bool = False,
    dropout_rate: float = 0.3,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    Build a complete classification model with backbone and head.
    
    Creates a model by combining:
    1. Pre-trained backbone (feature extractor)
    2. Optional attention mechanism
    3. Universal classification head
    
    Args:
        model_name: Name of the backbone architecture.
            Options: ResNet152V2, DenseNet201, VGG16, EfficientNetV2,
                     ConvNeXtBase
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        weights: Pre-trained weights ('imagenet' or None).
        trainable_backbone: Whether to fine-tune the backbone.
        use_attention: Whether to add SE attention block.
        dropout_rate: Dropout rate in classification head.
        dense_units: Number of units in dense layers.
        
    Returns:
        Compiled Keras Model.
        
    Example:
        >>> model = build_model("ConvNeXtBase", num_classes=4)
        >>> model.summary()
        
        >>> # With attention

    """
    print(f"🏗️ Building Model: {model_name}")
    
    # Handle attention suffix
    if model_name.endswith("_Attention"):
        base_name = model_name.replace("_Attention", "")
        use_attention = True
    else:
        base_name = model_name
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Get backbone
    backbone = get_backbone(
        base_name,
        input_shape=input_shape,
        weights=weights,
        trainable=trainable_backbone,
    )
    
    # Extract features
    features = backbone(inputs, training=False)
    
    # Apply attention if requested
    if use_attention:
        print("   ↳ Adding SE Attention Block")
        features = attention_block(features)
    
    # Attach classification head
    outputs = add_universal_head(
        features,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
    )
    
    # Create model
    model = tf.keras.Model(inputs, outputs, name=model_name)
    
    print(f"   ↳ Total parameters: {model.count_params():,}")
    print(f"   ↳ Trainable parameters: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    
    return model


def build_model_from_config(config: dict) -> tf.keras.Model:
    """
    Build a model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with model settings.
        
    Returns:
        Keras Model.
        
    Example:
        >>> config = load_config("configs/classification_config.yaml")
        >>> model = build_model_from_config(config)
    """
    return build_model(
        model_name=config.get("model_name", "ConvNeXtBase"),
        input_shape=tuple(config.get("img_size", [224, 224])) + (3,),
        num_classes=config.get("num_classes", NUM_CLASSES),
        weights=config.get("weights", "imagenet"),
        trainable_backbone=config.get("trainable_backbone", False),
        use_attention=config.get("use_attention", False),
        dropout_rate=config.get("dropout_rate", 0.3),
        dense_units=config.get("dense_units", 256),
    )


def list_available_models() -> list:
    """
    List all available model configurations.
    
    Returns:
        List of model names.
    """
    return AVAILABLE_MODELS.copy()
