"""
Segmentation module for brain tumor localization.

This module provides components for training and inference of
brain tumor segmentation models (tumor localization).

Example usage:
    >>> from src.segmentation import build_segmentation_model, SegmentationTrainer, TumorSegmentor
    
    # Build a model
    >>> model = build_segmentation_model("UNet", backbone="ResNet50")
    
    # Train using trainer
    >>> trainer = SegmentationTrainer(model_name="UNet", loss_name="bce_tversky")
    >>> trainer.prepare_data("data/train/images", "data/train/masks")
    >>> trainer.build()
    >>> trainer.compile()
    >>> trainer.train(epochs=100)
    
    # Inference
    >>> segmentor = TumorSegmentor("weights/UNet_best.keras")
    >>> mask = segmentor.predict("brain_scan.jpg")
"""

from .data import (
    create_segmentation_datasets,
    get_augmentation_fn,
    get_image_mask_pairs,
    get_training_augmentation,
    make_segmentation_dataset,
)
from .inference import TumorSegmentor, predict_mask
from .models import (
    AVAILABLE_SEGMENTATION_MODELS,
    build_model_from_config,
    build_segmentation_model,
    list_available_models,
)
from .training import (
    LOSS_FUNCTIONS,
    SegmentationTrainer,
    dice_coefficient,
    get_callbacks,
    get_loss_function,
    get_segmentation_metrics,
    iou_score,
    precision_metric,
    sensitivity,
    specificity,
    train_segmentation_model,
)

__all__ = [
    # Trainer
    "SegmentationTrainer",
    "train_segmentation_model",
    # Models
    "build_segmentation_model",
    "build_model_from_config",
    "list_available_models",
    "AVAILABLE_SEGMENTATION_MODELS",
    # Data
    "make_segmentation_dataset",
    "create_segmentation_datasets",
    "get_image_mask_pairs",
    "get_training_augmentation",
    "get_augmentation_fn",
    # Training
    "get_loss_function",
    "get_segmentation_metrics",
    "get_callbacks",
    "LOSS_FUNCTIONS",
    # Metrics
    "dice_coefficient",
    "iou_score",
    "sensitivity",
    "specificity",
    "precision_metric",
    # Inference
    "TumorSegmentor",
    "predict_mask",
]
