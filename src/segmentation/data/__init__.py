"""
Data loading and preprocessing for segmentation.

Provides efficient tf.data.Dataset pipelines and augmentation
for image-mask pairs used in brain tumor segmentation.
"""

from .dataset import (
    create_segmentation_datasets,
    get_image_mask_pairs,
    load_and_preprocess,
    make_segmentation_dataset,
    parse_filename,
)
from .preprocessing import (
    SegmentationAugmentation,
    get_augmentation_fn,
    get_segmentation_augmentation,
    get_training_augmentation,
)

__all__ = [
    # Dataset creation
    "make_segmentation_dataset",
    "create_segmentation_datasets",
    "load_and_preprocess",
    # Utilities
    "get_image_mask_pairs",
    "parse_filename",
    # Augmentation
    "get_training_augmentation",
    "get_augmentation_fn",
    "get_segmentation_augmentation",
    "SegmentationAugmentation",
]
