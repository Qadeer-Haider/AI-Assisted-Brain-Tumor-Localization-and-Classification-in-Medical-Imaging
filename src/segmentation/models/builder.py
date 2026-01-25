"""
Model builder for segmentation.

Factory function that creates segmentation models.

TODO: Implement when segmentation notebook is ready.
"""

import tensorflow as tf

from ...utils.constants import INPUT_SHAPE


def build_segmentation_model(
    model_name: str = "UNet",
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = 1,
    **kwargs,
) -> tf.keras.Model:
    """
    Build a segmentation model.
    
    Args:
        model_name: Name of the model architecture.
        input_shape: Input image shape.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments.
        
    Returns:
        Keras Model.
        
    Note:
        This is a placeholder. Implementation pending.
    """
    raise NotImplementedError(
        "Segmentation model builder not yet implemented. "
        "This will be added when the segmentation notebook is ready."
    )


AVAILABLE_SEGMENTATION_MODELS = [
    "UNet",
    "UNet_ResNet",
    "AttentionUNet",
]
