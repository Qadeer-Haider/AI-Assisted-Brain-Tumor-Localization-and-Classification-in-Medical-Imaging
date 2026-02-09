"""
Inference components for segmentation.

Provides prediction and visualization utilities for trained segmentation models.
"""

from .predictor import TumorSegmentor, predict_mask

__all__ = [
    "TumorSegmentor",
    "predict_mask",
]
