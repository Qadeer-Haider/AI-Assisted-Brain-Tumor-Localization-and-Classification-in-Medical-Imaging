"""
Training components for segmentation.

Provides loss functions, metrics, callbacks, and trainer class
for brain tumor segmentation model training.
"""

from .callbacks import get_callbacks, get_callbacks_from_config
from .losses import (
    LOSS_FUNCTIONS,
    bce_tversky_loss,
    dice_bce_loss,
    dice_coefficient,
    dice_loss,
    focal_loss,
    focal_tversky_loss,
    get_loss_function,
    tversky_loss,
)
from .metrics import (
    bce_tversky_loss_metric,
    dice_bce_loss_metric,
    dice_coefficient,
    dice_loss_metric,
    focal_tversky_loss_metric,
    get_basic_metrics,
    get_segmentation_metrics,
    iou_score,
    precision_metric,
    sensitivity,
    specificity,
    tversky_loss_metric,
)
from .trainer import SegmentationTrainer, train_segmentation_model

__all__ = [
    # Trainer
    "SegmentationTrainer",
    "train_segmentation_model",
    # Loss functions
    "get_loss_function",
    "dice_loss",
    "dice_bce_loss",
    "tversky_loss",
    "focal_tversky_loss",
    "bce_tversky_loss",
    "focal_loss",
    "LOSS_FUNCTIONS",
    # Metrics
    "get_segmentation_metrics",
    "get_basic_metrics",
    "dice_coefficient",
    "iou_score",
    "sensitivity",
    "specificity",
    "precision_metric",
    "dice_loss_metric",
    "tversky_loss_metric",
    "focal_tversky_loss_metric",
    "dice_bce_loss_metric",
    "bce_tversky_loss_metric",
    # Callbacks
    "get_callbacks",
    "get_callbacks_from_config",
]
